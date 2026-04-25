# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Pure PyTorch training orchestrator for Swift-DETR."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(train_config: TrainConfig) -> torch.device:
    accelerator = str(train_config.accelerator).lower()
    if accelerator in {"auto", "gpu", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
    if accelerator == "mps" or (accelerator == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


class _CSVLogger:
    """Minimal CSV metric logger."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._writer: Optional[csv.DictWriter] = None
        self._file = None
        self._fieldnames: list = []

    def log(self, row: dict) -> None:
        new_keys = [k for k in row if k not in self._fieldnames]
        if new_keys:
            # Re-open and rewrite with updated headers if new keys appear
            self._fieldnames.extend(new_keys)
            if self._file is not None:
                self._file.close()
            self._file = open(self._path, "a", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames, extrasaction="ignore")
            if os.path.getsize(self._path) == 0 or new_keys:
                # Write header only once; if keys expanded, rewrite whole file
                self._file.close()
                # Collect existing rows
                existing: list[dict] = []
                if os.path.exists(self._path):
                    with open(self._path) as f:
                        reader = csv.DictReader(f)
                        existing = list(reader)
                self._file = open(self._path, "w", newline="")
                self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames, extrasaction="ignore")
                self._writer.writeheader()
                for r in existing:
                    self._writer.writerow(r)

        if self._file is None:
            self._file = open(self._path, "a", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames, extrasaction="ignore")
            if os.path.getsize(self._path) == 0:
                self._writer.writeheader()

        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


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
    """Load a checkpoint for resume; returns the starting epoch."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Determine where model weights live
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        raw = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        state_dict = raw if raw else {}
    else:
        state_dict = ckpt

    model_for_load = getattr(model, "_orig_mod", model)
    model_for_load.load_state_dict(state_dict, strict=False)

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


def fit(
    wrapper: SwiftDetrWrapper,
    data: SwiftDetrData,
    output_dir: str,
    resume: Optional[str] = None,
    fast_dev_run: Optional[int] = None,
) -> None:
    """Full training loop — pure PyTorch, no Lightning.

    Args:
        wrapper: Model wrapper (model, criterion, postprocess).
        data: Data module (datasets + dataloaders + kornia pipeline).
        output_dir: Where checkpoints and logs are written.
        resume: Optional path to a checkpoint to resume from.
        fast_dev_run: If set, run only this many batches per epoch (sanity check).
    """
    tc = wrapper.train_config
    mc = wrapper.model_config
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Seed
    if tc.seed is not None:
        _seed_everything(tc.seed)

    # Device + precision
    device = _resolve_device(tc)
    precision = resolve_precision(mc)
    scaler: Optional[torch.amp.GradScaler] = None
    if precision == "fp16":
        scaler = torch.amp.GradScaler(device=device.type)
    logger.info("Device: %s | Precision: %s", device, precision)

    # cuDNN
    torch.backends.cudnn.benchmark = False

    # Move model to device
    wrapper.to(device)
    model = wrapper.model
    criterion = wrapper.criterion
    postprocess = wrapper.postprocess

    # DataLoaders
    train_loader = data.train_dataloader(world_size=1)
    val_loader = data.val_dataloader()
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * tc.epochs

    # Optimizer + scheduler
    optimizer, scheduler = wrapper.build_optimizer_and_scheduler(total_steps, steps_per_epoch)

    # Resume
    start_epoch = 0
    if resume:
        start_epoch = _load_resume_checkpoint(resume, model, optimizer, scheduler)

    # EMA
    ema_manager: Optional[EMAManager] = None
    if tc.use_ema:
        raw_model = getattr(model, "_orig_mod", model)
        ema_manager = EMAManager(raw_model, decay=tc.ema_decay, tau=tc.ema_tau, device=device)

    # Drop-path scheduler
    dp_scheduler: Optional[DropPathScheduler] = None
    if tc.drop_path > 0.0:
        dp_scheduler = DropPathScheduler(drop_path=tc.drop_path)
        dp_scheduler.build(tc.epochs, steps_per_epoch)

    # Multi-scale
    scales = []
    if tc.multi_scale:
        scales = compute_multi_scale_scales(mc.resolution, tc.expanded_scales, mc.patch_size, mc.num_windows)

    # Trackers
    best_tracker = BestModelTracker(output_dir, use_ema=tc.use_ema)
    early_stopper: Optional[EarlyStoppingTracker] = (
        EarlyStoppingTracker(
            patience=tc.early_stopping_patience,
            min_delta=tc.early_stopping_min_delta,
            use_ema=tc.early_stopping_use_ema,
        )
        if tc.early_stopping
        else None
    )

    # Logging
    csv_logger = _CSVLogger(str(output_path / "metrics.csv"))
    cat_id_to_name = data.cat_id_to_name
    class_names = data.class_names

    # Resolve args_dict for checkpoint payloads
    train_config_dump = tc.model_dump() if hasattr(tc, "model_dump") else {}
    if class_names and not train_config_dump.get("class_names"):
        train_config_dump = {**train_config_dump, "class_names": class_names}

    model_name: Optional[str] = None
    config_type = type(mc).__name__
    if config_type.startswith("SwiftDetr") and config_type.endswith("Config"):
        model_name = config_type.removesuffix("Config")

    global_step = start_epoch * steps_per_epoch

    logger.info(
        "Starting training: epochs=%d start=%d steps/epoch=%d",
        tc.epochs, start_epoch, steps_per_epoch,
    )

    for epoch in range(start_epoch, tc.epochs):
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
        )
        global_step += steps_per_epoch

        logger.info(
            "  train | loss=%.4f lr=%.6f",
            train_metrics["loss"], train_metrics["lr"],
        )

        row: dict = {"epoch": epoch, **{f"train/{k}": v for k, v in train_metrics.items()}}

        # Validation
        run_eval = (
            fast_dev_run is not None
            or (epoch + 1) % tc.eval_interval == 0
            or epoch + 1 == tc.epochs
        )
        val_metrics: dict = {}
        ema_metrics: dict = {}

        if run_eval:
            val_metrics = evaluate(
                model=model,
                postprocess=postprocess,
                loader=val_loader,
                device=device,
                cat_id_to_name=cat_id_to_name,
                max_dets=tc.eval_max_dets,
                segmentation=mc.segmentation_head,
                criterion=criterion,
                weight_dict=criterion.weight_dict,
                compute_loss=tc.compute_val_loss,
            )
            print_metrics_table("val", val_metrics)
            logger.info(
                "  val   | mAP50:95=%.4f mAP50=%.4f F1=%.4f",
                val_metrics.get("mAP_50_95", 0),
                val_metrics.get("mAP_50", 0),
                val_metrics.get("F1", 0),
            )
            row.update({f"val/{k}": v for k, v in val_metrics.items()})

            if ema_manager is not None:
                ema_model = ema_manager.module
                ema_model.eval()
                ema_metrics = evaluate(
                    model=ema_model,
                    postprocess=postprocess,
                    loader=val_loader,
                    device=device,
                    cat_id_to_name=cat_id_to_name,
                    max_dets=tc.eval_max_dets,
                    segmentation=mc.segmentation_head,
                )
                print_metrics_table("val (EMA)", ema_metrics)
                row.update({f"val/ema_{k}": v for k, v in ema_metrics.items()})

        csv_logger.log(row)

        # Checkpointing
        model_sd = wrapper.state_dict()
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

        if run_eval and val_metrics:
            combined_metrics = {**val_metrics}
            if ema_metrics:
                combined_metrics["ema_mAP_50_95"] = ema_metrics.get("mAP_50_95", 0.0)
            best_tracker.update(
                combined_metrics, model_sd, train_config_dump,
                epoch, global_step, ema_sd, model_name,
            )

            if early_stopper is not None:
                early_stopper.update(combined_metrics)
                if early_stopper.should_stop:
                    logger.info("Early stopping at epoch %d.", epoch + 1)
                    break

        if fast_dev_run is not None:
            logger.info("fast_dev_run=%d: stopping after 1 epoch.", fast_dev_run)
            break

    best_tracker.finalize()
    csv_logger.close()
    logger.info("Training complete. Outputs at: %s", output_dir)
