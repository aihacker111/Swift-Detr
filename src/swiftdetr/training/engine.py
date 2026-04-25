# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Pure PyTorch training and evaluation engine for Swift-DETR."""

from __future__ import annotations

import contextlib
import random
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.detection import MeanAveragePrecision

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from swiftdetr.evaluation.f1_sweep import sweep_confidence_thresholds
from swiftdetr.evaluation.matching import (
    build_matching_data,
    init_matching_accumulator,
    merge_matching_data,
)
from swiftdetr.util.box_ops import box_cxcywh_to_xyxy
from swiftdetr.util.logger import get_logger

logger = get_logger()


def resolve_precision(model_config) -> str:
    """Return AMP precision string: 'bf16', 'fp16', or 'fp32'."""
    if not model_config.amp:
        return "fp32"
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    return "fp32"


@contextlib.contextmanager
def _autocast(device_type: str, precision: str):
    if precision == "bf16":
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            yield
    elif precision == "fp16":
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
            yield
    else:
        yield


def _apply_kornia(samples, targets, kornia_pipeline, kornia_normalize):
    from swiftdetr.datasets.kornia_transforms import collate_boxes, unpack_boxes
    from swiftdetr.util.box_ops import box_xyxy_to_cxcywh
    from swiftdetr.util.tensors import NestedTensor

    img = samples.tensors
    kornia_pipeline.to(img.device)
    kornia_normalize.to(img.device)
    boxes_padded, valid = collate_boxes(targets, img.device)
    img_aug, boxes_aug = kornia_pipeline(img, boxes_padded)
    img_aug = kornia_normalize(img_aug)
    targets = unpack_boxes(boxes_aug, valid, targets, *img_aug.shape[-2:])
    height, width = img_aug.shape[-2:]
    for target in targets:
        boxes = target["boxes"]
        if boxes.numel() == 0:
            continue
        scale = boxes.new_tensor([width, height, width, height])
        target["boxes"] = box_xyxy_to_cxcywh(boxes) / scale
    return NestedTensor(img_aug, samples.mask), targets


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loader,
    device: torch.device,
    epoch: int,
    grad_accum: int,
    clip_max_norm: float,
    precision: str,
    scaler: Optional[torch.amp.GradScaler],
    ema_model: Any,
    dp_schedule: Optional[np.ndarray],
    global_step_start: int,
    multi_scale: bool,
    scales: list,
    do_random_resize_via_padding: bool,
    kornia_pipeline: Any,
    kornia_normalize: Any,
    ema_update_interval: int = 1,
) -> dict:
    model.train()
    weight_dict = criterion.weight_dict
    device_type = device.type
    total_loss = 0.0
    optimizer.zero_grad()
    optimizer_steps = 0

    group_lrs = [pg["lr"] for pg in optimizer.param_groups if "lr" in pg]
    current_lr = group_lrs[0] if group_lrs else 0.0

    pbar = (
        _tqdm(loader, desc=f"Epoch {epoch + 1} [train]", unit="batch", dynamic_ncols=True, leave=True)
        if _HAS_TQDM
        else loader
    )

    for batch_idx, (samples, targets) in enumerate(pbar):
        global_step = global_step_start + batch_idx

        # Drop-path scheduling
        if dp_schedule is not None and global_step < len(dp_schedule):
            model.update_drop_path(dp_schedule[global_step])

        # Transfer to device
        non_blocking = device_type == "cuda"
        samples = samples.to(device, non_blocking=non_blocking)
        targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in targets]

        # Kornia GPU augmentation
        if kornia_pipeline is not None:
            samples, targets = _apply_kornia(samples, targets, kornia_pipeline, kornia_normalize)

        # Multi-scale resize
        if multi_scale and not do_random_resize_via_padding:
            random.seed(global_step)
            scale = random.choice(scales)
            with torch.no_grad():
                samples.tensors = F.interpolate(
                    samples.tensors, size=scale, mode="bilinear", align_corners=False
                )
                samples.mask = (
                    F.interpolate(
                        samples.mask.unsqueeze(1).float(), size=scale, mode="nearest"
                    )
                    .squeeze(1)
                    .bool()
                )

        # Forward + loss
        with _autocast(device_type, precision):
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            loss_scaled = loss / grad_accum

        if scaler is not None:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        is_accum_step = (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader)
        if is_accum_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_steps += 1

            if ema_model is not None and optimizer_steps % ema_update_interval == 0:
                ema_model.update(model)

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        if _HAS_TQDM and hasattr(pbar, "set_postfix"):
            if is_accum_step:
                current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")

    # Epoch-end EMA update
    if ema_model is not None:
        ema_model.update(model)

    group_lrs = [pg["lr"] for pg in optimizer.param_groups if "lr" in pg]
    return {
        "loss": total_loss / max(1, len(loader)),
        "lr": group_lrs[0] if group_lrs else 0.0,
    }


def _convert_preds(preds: list) -> list:
    out = []
    for p in preds:
        entry = dict(p)
        if "masks" in entry and entry["masks"].ndim == 4 and entry["masks"].shape[1] == 1:
            entry["masks"] = entry["masks"].squeeze(1)
        out.append(entry)
    return out


def _convert_targets(targets: list) -> list:
    out = []
    for t in targets:
        h, w = t["orig_size"].tolist()
        scale = t["boxes"].new_tensor([w, h, w, h])
        boxes = box_cxcywh_to_xyxy(t["boxes"]) * scale
        entry: dict = {"boxes": boxes, "labels": t["labels"]}
        if "masks" in t:
            masks = t["masks"].bool()
            if masks.shape[-2:] != (int(h), int(w)):
                masks = (
                    F.interpolate(
                        masks.float().unsqueeze(1), size=(int(h), int(w)), mode="nearest"
                    )
                    .squeeze(1)
                    .bool()
                )
            entry["masks"] = masks
        if "iscrowd" in t:
            entry["iscrowd"] = t["iscrowd"]
        out.append(entry)
    return out


def evaluate(
    model: nn.Module,
    postprocess,
    loader,
    device: torch.device,
    cat_id_to_name: dict,
    max_dets: int = 500,
    segmentation: bool = False,
    criterion: Optional[nn.Module] = None,
    weight_dict: Optional[dict] = None,
    compute_loss: bool = False,
) -> dict:
    model.eval()
    iou_type: Any = ["bbox", "segm"] if segmentation else "bbox"
    map_metric = MeanAveragePrecision(
        iou_type=iou_type,
        class_metrics=True,
        max_detection_thresholds=[1, 10, max_dets],
        backend="faster_coco_eval",
    ).to(device)
    f1_local: dict = init_matching_accumulator()
    total_loss = 0.0

    val_pbar = (
        _tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True, leave=False)
        if _HAS_TQDM
        else loader
    )

    with torch.no_grad():
        for samples, targets in val_pbar:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)

            if compute_loss and criterion is not None and weight_dict is not None:
                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
                total_loss += loss.item()

            orig_sizes = torch.stack([t["orig_size"] for t in targets])
            results = postprocess(outputs, orig_sizes)

            preds = _convert_preds(results)
            gts = _convert_targets(targets)
            map_metric.update(preds, gts)

            iou_t = "segm" if segmentation else "bbox"
            batch_matching = build_matching_data(preds, gts, iou_threshold=0.5, iou_type=iou_t)
            merge_matching_data(f1_local, batch_matching)

    metrics = map_metric.compute()
    pfx = "bbox_" if segmentation else ""
    mar_key = f"{pfx}mar_{max_dets}"

    out: dict = {
        "mAP_50_95": float(metrics[f"{pfx}map"]),
        "mAP_50": float(metrics[f"{pfx}map_50"]),
        "mAP_75": float(metrics[f"{pfx}map_75"]),
        "mAR": float(metrics[mar_key]),
    }
    if compute_loss:
        out["loss"] = total_loss / max(1, len(loader))
    if segmentation:
        out["segm_mAP_50_95"] = float(metrics["segm_map"])
        out["segm_mAP_50"] = float(metrics["segm_map_50"])

    # F1 sweep — evaluate() only runs on rank 0, so no cross-rank gather needed
    merged = f1_local
    if merged:
        sorted_ids = sorted(merged.keys())
        per_class_list = [merged[cid] for cid in sorted_ids]
        classes_with_gt = [i for i, cid in enumerate(sorted_ids) if merged[cid]["total_gt"] > 0]
        f1_results = sweep_confidence_thresholds(per_class_list, np.linspace(0, 1, 101), classes_with_gt)
        best = max(f1_results, key=lambda x: x["macro_f1"])
        out["F1"] = float(best["macro_f1"])
        out["precision"] = float(best["macro_precision"])
        out["recall"] = float(best["macro_recall"])
    else:
        out["F1"] = 0.0
        out["precision"] = 0.0
        out["recall"] = 0.0

    # Per-class AP
    if "classes" in metrics and metrics["classes"].ndim == 0:
        metrics = dict(metrics)
        metrics["classes"] = metrics["classes"].unsqueeze(0)
        for k in list(metrics):
            if isinstance(metrics[k], torch.Tensor) and metrics[k].ndim == 0 and "per_class" in k:
                metrics[k] = metrics[k].unsqueeze(0)

    pc_key = f"{pfx}map_per_class"
    if pc_key in metrics and "classes" in metrics:
        for class_id, ap in zip(metrics["classes"], metrics[pc_key]):
            idx = int(class_id)
            ap_f = float(ap)
            if ap_f < 0:
                continue
            name = cat_id_to_name.get(idx, str(idx))
            out[f"AP/{name}"] = ap_f

    map_metric.reset()
    return out


def print_metrics_table(split: str, metrics: dict, ema_metrics: Optional[dict] = None) -> None:
    """Print a compact metrics table to stdout."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console(force_terminal=True)
        t = Table(title=f"{split.capitalize()} — Overall Metrics", title_style="bold cyan", header_style="bold cyan")
        t.add_column("Metric")
        t.add_column("Regular", justify="right")
        if ema_metrics:
            t.add_column("EMA", justify="right")

        key_labels = [
            ("mAP_50_95", "mAP 50:95"),
            ("mAP_50", "mAP 50"),
            ("mAP_75", "mAP 75"),
            ("mAR", "mAR"),
            ("F1", "F1"),
            ("precision", "Precision"),
            ("recall", "Recall"),
        ]
        for key, label in key_labels:
            val = metrics.get(key, float("nan"))
            row = [label, f"{val:.4f}" if val == val else "—"]
            if ema_metrics:
                ema_val = ema_metrics.get(key, float("nan"))
                row.append(f"{ema_val:.4f}" if ema_val == ema_val else "—")
            t.add_row(*row)
        console.print(t)
    except ImportError:
        keys = ["mAP_50_95", "mAP_50", "F1"]
        parts = " | ".join(f"{k}: {metrics.get(k, float('nan')):.4f}" for k in keys)
        logger.info("[%s] %s", split.upper(), parts)
