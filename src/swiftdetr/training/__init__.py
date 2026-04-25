# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Swift-DETR training package (PyTorch Lightning).

Provides the Lightning module, data module, callbacks, and CLI for
training and evaluation.

Exports:
    SwiftDetrModule: LightningModule wrapping the Swift-DETR model and training loop.
    SwiftDetrDataModule: LightningDataModule wrapping dataset construction and loaders.
    build_trainer: Factory that assembles a PTL Trainer from Swift-DETR configs.
"""

from pytorch_lightning import seed_everything

from swiftdetr.training.callbacks import (
    BestModelCallback,
    COCOEvalCallback,
    DropPathCallback,
    SwiftDetrEarlyStopping,
    SwiftDetrEMACallback,
)
from swiftdetr.training.checkpoint import convert_legacy_checkpoint
from swiftdetr.training.cli import SwiftDetrCli
from swiftdetr.training.module_data import SwiftDetrDataModule
from swiftdetr.training.module_model import SwiftDetrModule
from swiftdetr.training.trainer import build_trainer
from swiftdetr.util.logger import get_logger

_logger = get_logger()

__all__ = [
    "BestModelCallback",
    "COCOEvalCallback",
    "DropPathCallback",
    "SwiftDetrCli",
    "SwiftDetrDataModule",
    "SwiftDetrEMACallback",
    "SwiftDetrEarlyStopping",
    "SwiftDetrModule",
    "build_trainer",
    "convert_legacy_checkpoint",
    "seed_everything",
]
