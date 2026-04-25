# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Drop-path / dropout scheduler (pure PyTorch, no Lightning)."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from swiftdetr.training.drop_schedule import drop_scheduler


class DropPathScheduler:
    """Computes and applies per-step drop-path / dropout rate schedules.

    Args:
        drop_path: Peak drop-path rate.
        dropout: Peak dropout rate.
        cutoff_epoch: Epoch boundary for early/late modes.
        mode: Schedule mode.
        schedule: Schedule shape.
        vit_encoder_num_layers: Passed to ``model.update_drop_path``.
    """

    def __init__(
        self,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        cutoff_epoch: int = 0,
        mode: Literal["standard", "early", "late"] = "standard",
        schedule: Literal["constant", "linear"] = "constant",
        vit_encoder_num_layers: int = 12,
    ) -> None:
        self._drop_path = drop_path
        self._dropout = dropout
        self._cutoff_epoch = cutoff_epoch
        self._mode = mode
        self._schedule = schedule
        self._vit_encoder_num_layers = vit_encoder_num_layers

        self.dp_schedule: Optional[np.ndarray] = None
        self.do_schedule: Optional[np.ndarray] = None

    def build(self, epochs: int, steps_per_epoch: int) -> None:
        """Build the rate arrays. Call once before the training loop."""
        if self._drop_path > 0:
            self.dp_schedule = drop_scheduler(
                self._drop_path, epochs, steps_per_epoch,
                self._cutoff_epoch, self._mode, self._schedule,
            )
        if self._dropout > 0:
            self.do_schedule = drop_scheduler(
                self._dropout, epochs, steps_per_epoch,
                self._cutoff_epoch, self._mode, self._schedule,
            )

    def apply(self, model, global_step: int) -> None:
        """Apply the scheduled rates at ``global_step`` to ``model``."""
        if self.dp_schedule is not None and global_step < len(self.dp_schedule):
            model.update_drop_path(self.dp_schedule[global_step], self._vit_encoder_num_layers)
        if self.do_schedule is not None and global_step < len(self.do_schedule):
            model.update_dropout(self.do_schedule[global_step])
