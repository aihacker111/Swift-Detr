# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Swift-DETR model wrapper (pure PyTorch, no Lightning)."""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import torch
import torch.nn as nn

from swiftdetr._namespace import _namespace_from_configs
from swiftdetr.config import ModelConfig, TrainConfig
from swiftdetr.models import build_criterion_from_config, build_model_from_config
from swiftdetr.models.weights import apply_lora, load_pretrain_weights
from swiftdetr.training.param_groups import get_param_dict
from swiftdetr.util.logger import get_logger

logger = get_logger()


class SwiftDetrWrapper:
    """Plain Python wrapper around the Swift-DETR model, criterion, and postprocessor.

    Replaces the former ``LightningModule``-based ``SwiftDetrModule``.
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig) -> None:
        self.model_config = model_config
        self.train_config = train_config

        self.model: nn.Module = build_model_from_config(model_config, train_config)

        if model_config.pretrain_weights is not None and model_config.load_detection_pretrain:
            prev_num_classes = self.model_config.num_classes
            load_pretrain_weights(self.model, self.model_config)
            if hasattr(self.model, "num_classes"):
                model_num_classes = getattr(self.model, "num_classes")
                if model_num_classes is not None and model_num_classes != prev_num_classes:
                    self.model_config.num_classes = model_num_classes

        if model_config.backbone_lora:
            apply_lora(self.model)

        self.criterion, self.postprocess = build_criterion_from_config(self.model_config, self.train_config)

        # torch.compile (CUDA only, no multi-scale)
        from swiftdetr.config import DEVICE

        accelerator = str(train_config.accelerator).lower()
        uses_cuda = accelerator in {"auto", "gpu", "cuda"}
        compile_enabled = (
            model_config.compile and DEVICE == "cuda" and uses_cuda and not train_config.multi_scale
        )
        if model_config.compile and train_config.multi_scale:
            logger.info("Disabling torch.compile: multi_scale=True introduces dynamic shapes.")
        if compile_enabled:
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.capture_scalar_outputs = True
            self.model = torch.compile(self.model, dynamic=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "SwiftDetrWrapper":
        self.model = self.model.to(device)
        return self

    @property
    def _use_fused_optimizer(self) -> bool:
        tc = self.train_config
        return (
            self.model_config.fused_optimizer
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ) and tc is not None

    def build_optimizer_and_scheduler(
        self, total_steps: int, steps_per_epoch: int
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        tc = self.train_config
        ns = _namespace_from_configs(self.model_config, tc)

        model_for_params = getattr(self.model, "_orig_mod", self.model)
        param_dicts = get_param_dict(ns, model_for_params)
        param_dicts = [p for p in param_dicts if p["params"].requires_grad]

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=tc.lr,
            weight_decay=tc.weight_decay,
            fused=self._use_fused_optimizer,
        )

        warmup_steps = int(steps_per_epoch * tc.warmup_epochs)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if tc.lr_scheduler == "cosine":
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return tc.lr_min_factor + (1 - tc.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
            if current_step < tc.lr_drop * steps_per_epoch:
                return 1.0
            return 0.1

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, scheduler

    def load_checkpoint(self, path: str) -> int:
        """Load a checkpoint and return the starting epoch."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Handle legacy .pth (no "state_dict" key)
        if "model" in ckpt and "state_dict" not in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            # PTL-format: strip "model." prefix
            raw = {
                k[len("model."):]: v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.")
            }
            state_dict = raw if raw else ckpt.get("model", {})
        else:
            state_dict = ckpt

        model_for_load = getattr(self.model, "_orig_mod", self.model)
        incompatible = model_for_load.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            warnings.warn(
                f"Checkpoint loaded with non-exact key match: "
                f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
                UserWarning,
                stacklevel=2,
            )

        return int(ckpt.get("epoch", 0))

    def state_dict(self) -> dict:
        raw = getattr(self.model, "_orig_mod", self.model)
        return raw.state_dict()

    def reinitialize_detection_head(self, num_classes: int) -> None:
        self.model.reinitialize_detection_head(num_classes)
