# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Lightning callbacks for Swift-DETR training."""

from swiftdetr.training.callbacks.best_model import BestModelCallback, SwiftDetrEarlyStopping
from swiftdetr.training.callbacks.coco_eval import COCOEvalCallback
from swiftdetr.training.callbacks.drop_schedule import DropPathCallback
from swiftdetr.training.callbacks.ema import SwiftDetrEMACallback

__all__ = [
    "BestModelCallback",
    "COCOEvalCallback",
    "DropPathCallback",
    "SwiftDetrEMACallback",
    "SwiftDetrEarlyStopping",
]
