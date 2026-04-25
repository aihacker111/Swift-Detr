# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""EMA manager for Swift-DETR (pure PyTorch, no Lightning)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from swiftdetr.training.model_ema import ModelEma


class EMAManager:
    """Manages an exponential moving average copy of a model.

    Wraps ``ModelEma`` and exposes a clean API for use in the training loop.

    Args:
        model: The live model to track.
        decay: EMA decay factor.
        tau: Warm-up time constant (steps). 0 = no warm-up.
        device: Device for the EMA model. Defaults to same device as ``model``.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9997,
        tau: int = 100,
        device: Optional[torch.device] = None,
    ) -> None:
        self._ema = ModelEma(model, decay=decay, tau=float(tau), device=device)

    @property
    def module(self) -> nn.Module:
        """The EMA model (inference-ready copy)."""
        return self._ema.module

    def update(self, model: nn.Module) -> None:
        """Update EMA weights from the current live model."""
        self._ema.update(model)

    def state_dict(self) -> dict:
        return self._ema.module.state_dict()

    def load_state_dict(self, state_dict: dict, strict: bool = False) -> None:
        self._ema.module.load_state_dict(state_dict, strict=strict)
