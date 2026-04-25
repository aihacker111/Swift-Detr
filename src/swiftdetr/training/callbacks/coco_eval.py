# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""COCO evaluation helpers (pure PyTorch, no Lightning).

The evaluation logic previously lived in ``COCOEvalCallback``.
It is now exposed as plain functions called directly from the training loop.
The ``evaluate()`` function in ``engine.py`` is the main entry point.
"""

# Re-export evaluate and print_metrics_table so existing imports of
# ``from swiftdetr.training.callbacks.coco_eval import ...`` still work.
from swiftdetr.training.engine import evaluate, print_metrics_table

__all__ = ["evaluate", "print_metrics_table"]
