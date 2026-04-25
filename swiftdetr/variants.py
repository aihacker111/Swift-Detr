"""Concrete Swift-DETR model variant classes.

Three detection variants:
    SwiftDetrTiny   — swiftnet_tiny  (~6M  backbone params)
    SwiftDetrSmall  — swiftnet_small (~15M backbone params)
    SwiftDetrBase   — swiftnet_base  (~30M backbone params)

All inherit from :class:`swiftdetr.detr.SwiftDetr`.
"""

from __future__ import annotations

__all__ = ["SwiftDetrTiny", "SwiftDetrSmall", "SwiftDetrBase"]

from swiftdetr.config import ModelConfig, SwiftDetrBaseConfig, SwiftDetrSmallConfig, SwiftDetrTinyConfig
from swiftdetr.detr import SwiftDetr


class SwiftDetrTiny(SwiftDetr):
    """Swift-DETR Tiny — optimised for edge / real-time inference."""

    size = "swiftdetr-tiny"
    _model_config_class = SwiftDetrTinyConfig


class SwiftDetrSmall(SwiftDetr):
    """Swift-DETR Small — balanced speed / accuracy tradeoff."""

    size = "swiftdetr-small"
    _model_config_class = SwiftDetrSmallConfig


class SwiftDetrBase(SwiftDetr):
    """Swift-DETR Base — maximum accuracy on server-class edge hardware."""

    size = "swiftdetr-base"
    _model_config_class = SwiftDetrBaseConfig
