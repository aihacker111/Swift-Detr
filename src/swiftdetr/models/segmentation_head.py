# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Backward-compatibility shim — swiftdetr.models.segmentation_head is deprecated; use swiftdetr.models.heads.segmentation."""

from swiftdetr.util.decorators import _warn_deprecated_module

_warn_deprecated_module("swiftdetr.models.segmentation_head", "swiftdetr.models.heads.segmentation")

from swiftdetr.models.heads.segmentation import DepthwiseConvBlock, MLPBlock, SegmentationHead  # noqa: F401, E402
