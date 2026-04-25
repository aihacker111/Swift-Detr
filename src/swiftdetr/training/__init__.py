# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Swift-DETR training package (pure PyTorch, no Lightning).

Exports:
    SwiftDetrWrapper: Model wrapper (model + criterion + postprocess).
    SwiftDetrData: Dataset and dataloader builder.
    fit: Main training orchestrator.
    BestModelTracker: Saves best checkpoints.
    DropPathScheduler: Per-step drop-path schedule.
    EarlyStoppingTracker: Patience-based early stopping.
    EMAManager: Exponential moving average manager.
    convert_legacy_checkpoint: Convert legacy .pth checkpoints.
"""

from swiftdetr.training.callbacks import (
    BestModelTracker,
    DropPathScheduler,
    EarlyStoppingTracker,
    EMAManager,
)
from swiftdetr.training.checkpoint import convert_legacy_checkpoint
from swiftdetr.training.module_data import SwiftDetrData
from swiftdetr.training.module_model import SwiftDetrWrapper
from swiftdetr.training.trainer import fit
from swiftdetr.util.logger import get_logger

_logger = get_logger()

__all__ = [
    "BestModelTracker",
    "DropPathScheduler",
    "EarlyStoppingTracker",
    "EMAManager",
    "SwiftDetrData",
    "SwiftDetrWrapper",
    "convert_legacy_checkpoint",
    "fit",
]
