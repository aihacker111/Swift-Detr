# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Pure-PyTorch training helpers (formerly Lightning callbacks)."""

from swiftdetr.training.callbacks.best_model import BestModelTracker, EarlyStoppingTracker
from swiftdetr.training.callbacks.drop_schedule import DropPathScheduler
from swiftdetr.training.callbacks.ema import EMAManager

__all__ = [
    "BestModelTracker",
    "DropPathScheduler",
    "EarlyStoppingTracker",
    "EMAManager",
]
