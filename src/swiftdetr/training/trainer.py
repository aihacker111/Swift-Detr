# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Pure PyTorch training orchestrator for Swift-DETR (single-GPU and multi-GPU DDP)."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from swiftdetr.config import ModelConfig, TrainConfig
from swiftdetr.datasets.coco import compute_multi_scale_scales
from swiftdetr.training.callbacks.best_model import BestModelTracker, EarlyStoppingTracker
from swiftdetr.training.callbacks.drop_schedule import DropPathScheduler
from swiftdetr.training.callbacks.ema import EMAManager
from swiftdetr.training.engine import evaluate, print_metrics_table, resolve_precision, train_one_epoch
from swiftdetr.training.module_data import SwiftDetrData
from swiftdetr.training.module_model import SwiftDetrWrapper
from swiftdetr.util.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def _is_ddp() -> bool:
    """Return True when running under torchrun (WORLD_SIZE > 1)."""
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _is_main() -> bool:
    return _rank() == 0


def _setup_ddp() -> torch.device:
    """Initialize NCCL process group and return the local CUDA device."""
    local = _local_rank()
    torch.cuda.set_device(local)
    if not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local}")
    logger.info(
        "DDP initialised: rank=%d / world=%d | device=%s",
        _rank(), _world_size(), device,
    )
    return device


def _cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    """Seed RNGs; per-rank offset ensures different data order on each GPU."""
    seed = seed + _rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(train_config: TrainConfig) -> torch.device:
    """Device for single-GPU / CPU runs (DDP resolves its own device via _setup_ddp)."""
    accelerator = str(train_config.accelerator).lower()
    if accelerator in {"auto", "gpu", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
    if accelerator == "mps" or (accelerator == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# CSV logger (rank-0 only)
# ---------------------------------------------------------------------------

class _CSVLogger:
    def __init__(self, path: str) -> None:
        self._path = path
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: list = []

    def log(self, row: dict) -> None:
        new_keys = [k for k in row if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
            # Collect existing rows and rewrite with updated header
            existing: list[dict] = []
            if self._file is not None:
                self._file.close()
                self._file = None
            if os.path.exists(self._path) and os.path.getsize(self._path) > 0:
                with open(self._path) as f:
                    existing = list(csv.DictReader(f))
            self._file = open(self._path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames, extrasaction="ignore")
            self._writer.writeheader()
            for r in existing:
                self._writer.writerow(r)

        if self._file is None:
            self._file = open(self._path, "a", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames, extrasaction="ignore")
            if not os.path.exists(self._path) or os.path.getsize(self._path) == 0:
                self._writer.writeheader()

        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    output_dir: Path,
    filename: str,
    model_state_dict: dict,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    ema_state_dict: Optional[dict],
    args_dict: object,
    model_name: Optional[str] = None,
) -> None:
    path = output_dir / filename
    payload: dict = {
        "model": model_state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": args_dict,
    }
    if ema_state_dict is not None:
        payload["ema_model"] = ema_state_dict
    if model_name is not None:
        payload["model_name"] = model_name
    torch.save(payload, path)


def _load_resume_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
) -> int:
    """Load checkpoint for resume. Returns starting epoch."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        raw = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        state_dict = raw if raw else {}
    else:
        state_dict = ckpt

    # Unwrap DDP / compile before loading
    raw_model = getattr(model, "module", model)
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    raw_model.load_state_dict(state_dict, strict=False)

    start_epoch = int(ckpt.get("epoch", 0)) + 1

    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            logger.warning("Could not restore optimizer state: %s", exc)

    if scheduler is not None and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as exc:
            logger.warning("Could not restore scheduler state: %s", exc)

    logger.info("Resumed from %s (epoch %d → start %d)", path, ckpt.get("epoch", 0), start_epoch)
    return start_epoch


# ---------------------------------------------------------------------------
# Main fit function
# ---------------------------------------------------------------------------

def fit(
    wrapper: SwiftDetrWrapper,
    data: SwiftDetrData,
    output_dir: str,
    resume: Optional[str] = None,
    fast_dev_run: Optional[int] = None,
) -> None:
    """Full training loop — pure PyTorch, single-GPU or multi-GPU DDP.

    Single GPU::

        python train.py --dataset /data/coco --output ./out

    Multi-GPU (launch with torchrun)::

        torchrun --nproc_per_node=4 train.py --dataset /data/coco --output ./out

    Args:
        wrapper: Model wrapper (model, criterion, postprocess).
        data: Data builder (datasets + dataloaders + kornia pipeline).
        output_dir: Directory for checkpoints and logs.
        resume: Optional checkpoint path to resume from.
        fast_dev_run: If set, run only this many batches (sanity check).
    """
    tc = wrapper.train_config
    mc = wrapper.model_config
    output_path = Path(output_dir)

    # ---- DDP or single-GPU device resolution ----
    ddp_active = _is_ddp()
    if ddp_active:
        device = _setup_ddp()
    else:
        device = _resolve_device(tc)

    world_size = _world_size()
    is_main = _is_main()

    if is_main:
        output_path.mkdir(parents=True, exist_ok=True)

    # ---- Seeding ----
    if tc.seed is not None:
        _seed_everything(tc.seed)

    # ---- Precision + cuDNN ----
    precision = resolve_precision(mc)
    scaler: Optional[torch.amp.GradScaler] = None
    if precision == "fp16":
        scaler = torch.amp.GradScaler(device=device.type)
    torch.backends.cudnn.benchmark = False
    if is_main:
        logger.info("Device: %s | Precision: %s | World size: %d", device, precision, world_size)

    # ---- Model → device ----
    wrapper.to(device)
    model = wrapper.model
    criterion = wrapper.criterion
    postprocess = wrapper.postprocess

    # ---- Sync BatchNorm (optional, good for small batch sizes in DDP) ----
    if ddp_active and tc.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # ---- Wrap with DDP ----
    if ddp_active:
        model = DDP(model, device_ids=[_local_rank()], find_unused_parameters=True)

    # ---- DataLoaders ----
    train_loader = data.train_dataloader(world_size=world_size, rank=_rank())
    val_loader = data.val_dataloader(world_size=world_size, rank=_rank())

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * tc.epochs

    # ---- Optimizer + scheduler ----
    # Build against the unwrapped model so param names match get_param_dict expectations
    wrapper.model = getattr(model, "module", model)
    optimizer, scheduler = wrapper.build_optimizer_and_scheduler(total_steps, steps_per_epoch)
    wrapper.model = model  # restore reference

    # ---- Resume ----
    start_epoch = 0
    if resume:
        start_epoch = _load_resume_checkpoint(resume, model, optimizer, scheduler)
    _barrier()  # all ranks wait until rank-0 finishes loading

    # ---- EMA (main process only to avoid duplicate copies) ----
    ema_manager: Optional[EMAManager] = None
    if tc.use_ema and is_main:
        raw_model = getattr(model, "module", model)
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        ema_manager = EMAManager(raw_model, decay=tc.ema_decay, tau=tc.ema_tau, device=device)

    # ---- Drop-path scheduler ----
    dp_scheduler: Optional[DropPathScheduler] = None
    if tc.drop_path > 0.0:
        dp_scheduler = DropPathScheduler(drop_path=tc.drop_path)
        dp_scheduler.build(tc.epochs, steps_per_epoch)

    # ---- Multi-scale ----
    scales: list = []
    if tc.multi_scale:
        scales = compute_multi_scale_scales(mc.resolution, tc.expanded_scales, mc.patch_size, mc.num_windows)

    # ---- Trackers + loggers (rank-0 only) ----
    best_tracker: Optional[BestModelTracker] = None
    early_stopper: Optional[EarlyStoppingTracker] = None
    csv_logger: Optional[_CSVLogger] = None
    if is_main:
        best_tracker = BestModelTracker(output_dir, use_ema=tc.use_ema)
        if tc.early_stopping:
            early_stopper = EarlyStoppingTracker(
                patience=tc.early_stopping_patience,
                min_delta=tc.early_stopping_min_delta,
                use_ema=tc.early_stopping_use_ema,
            )
        csv_logger = _CSVLogger(str(output_path / "metrics.csv"))

    coco_gt = data.coco_gt
    cat_id_to_name = data.cat_id_to_name
    class_names = data.class_names
    debug_limit_batches = tc.debug_limit_data

    train_config_dump = tc.model_dump() if hasattr(tc, "model_dump") else {}
    if class_names and not train_config_dump.get("class_names"):
        train_config_dump = {**train_config_dump, "class_names": class_names}

    model_name: Optional[str] = None
    config_type = type(mc).__name__
    if config_type.startswith("SwiftDetr") and config_type.endswith("Config"):
        model_name = config_type.removesuffix("Config")

    global_step = start_epoch * steps_per_epoch

    if is_main:
        logger.info(
            "Starting training: epochs=%d start=%d steps/epoch=%d world=%d",
            tc.epochs, start_epoch, steps_per_epoch, world_size,
        )

    should_stop = False

    for epoch in range(start_epoch, tc.epochs):
        # Set epoch on DistributedSampler so each rank shuffles differently
        if ddp_active and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if is_main:
            logger.info("Epoch %d/%d", epoch + 1, tc.epochs)

        dp_schedule = dp_scheduler.dp_schedule if dp_scheduler is not None else None

        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=train_loader,
            device=device,
            epoch=epoch,
            grad_accum=tc.grad_accum_steps,
            clip_max_norm=tc.clip_max_norm,
            precision=precision,
            scaler=scaler,
            ema_model=ema_manager,
            dp_schedule=dp_schedule,
            global_step_start=global_step,
            multi_scale=tc.multi_scale,
            scales=scales,
            do_random_resize_via_padding=tc.do_random_resize_via_padding,
            kornia_pipeline=data._kornia_pipeline,
            kornia_normalize=data._kornia_normalize,
            ema_update_interval=tc.ema_update_interval,
            debug_limit_batches=debug_limit_batches,
        )
        global_step += steps_per_epoch

        if is_main:
            logger.info(
                "  train | loss=%.4f lr=%.6f",
                train_metrics["loss"], train_metrics["lr"],
            )

        row: dict = {"epoch": epoch, **{f"train/{k}": v for k, v in train_metrics.items()}}

        # ---- Validation — all ranks evaluate their shard in parallel ----
        run_eval = (
            fast_dev_run is not None
            or (epoch + 1) % tc.eval_interval == 0
            or epoch + 1 == tc.epochs
        )
        val_metrics: dict = {}
        ema_metrics: dict = {}

        if run_eval:
            # All ranks participate; torchmetrics + distributed_merge_matching_data
            # handle cross-rank aggregation internally via all_gather.
            eval_model = getattr(model, "module", model)
            val_metrics = evaluate(
                model=eval_model,
                postprocess=postprocess,
                loader=val_loader,
                device=device,
                coco_gt=coco_gt,
                max_dets=tc.eval_max_dets,
                segmentation=mc.segmentation_head,
                criterion=criterion,
                weight_dict=criterion.weight_dict,
                compute_loss=tc.compute_val_loss,
                distribute=ddp_active,
                debug_limit_batches=debug_limit_batches,
            )
            if is_main:
                print_metrics_table("val", val_metrics)
                logger.info(
                    "  val   | mAP50:95=%.4f mAP50=%.4f F1=%.4f",
                    val_metrics.get("mAP_50_95", 0),
                    val_metrics.get("mAP_50", 0),
                    val_metrics.get("F1", 0),
                )
                row.update({f"val/{k}": v for k, v in val_metrics.items() if not isinstance(v, str)})

            # EMA exists only on rank 0 — barrier so rank 1 waits, then evaluate
            # locally (distribute=False avoids all_gather with only rank 0 active).
            _barrier()
            if is_main and ema_manager is not None:
                ema_model = ema_manager.module
                ema_model.eval()
                ema_val_loader = data.val_dataloader(world_size=1, rank=0)
                ema_metrics = evaluate(
                    model=ema_model,
                    postprocess=postprocess,
                    loader=ema_val_loader,
                    device=device,
                    coco_gt=coco_gt,
                    max_dets=tc.eval_max_dets,
                    segmentation=mc.segmentation_head,
                    distribute=False,
                    debug_limit_batches=debug_limit_batches,
                )
                print_metrics_table("val (EMA)", ema_metrics)
                row.update({f"val/ema_{k}": v for k, v in ema_metrics.items() if not isinstance(v, str)})
            _barrier()

        # ---- Checkpointing + tracking (rank-0 only) ----
        if is_main:
            if csv_logger:
                csv_logger.log(row)

            # Unwrap for state dict (DDP wraps under .module)
            unwrapped = getattr(model, "module", model)
            unwrapped = getattr(unwrapped, "_orig_mod", unwrapped)
            model_sd = unwrapped.state_dict()
            ema_sd = ema_manager.state_dict() if ema_manager is not None else None

            _save_checkpoint(
                output_path, "last.pth",
                model_sd, optimizer, scheduler,
                epoch, global_step, ema_sd,
                train_config_dump, model_name,
            )
            if (epoch + 1) % tc.checkpoint_interval == 0:
                _save_checkpoint(
                    output_path, f"checkpoint_{epoch}.pth",
                    model_sd, optimizer, scheduler,
                    epoch, global_step, ema_sd,
                    train_config_dump, model_name,
                )

            if run_eval and val_metrics and best_tracker:
                combined = {**val_metrics}
                if ema_metrics:
                    combined["ema_mAP_50_95"] = ema_metrics.get("mAP_50_95", 0.0)
                best_tracker.update(
                    combined, model_sd, train_config_dump,
                    epoch, global_step, ema_sd, model_name,
                )

                if early_stopper is not None:
                    early_stopper.update(combined)
                    if early_stopper.should_stop:
                        logger.info("Early stopping at epoch %d.", epoch + 1)
                        should_stop = True

        # Broadcast early-stop decision to all ranks
        if ddp_active:
            stop_tensor = torch.tensor(int(should_stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

        if should_stop:
            break

        if fast_dev_run is not None:
            if is_main:
                logger.info("fast_dev_run=%d: stopping after 1 epoch.", fast_dev_run)
            break

    if is_main and best_tracker:
        best_tracker.finalize()
    if is_main and csv_logger:
        csv_logger.close()
    if is_main:
        logger.info("Training complete. Outputs at: %s", output_dir)

    _cleanup_ddp()
