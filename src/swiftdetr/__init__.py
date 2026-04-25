"""Swift-DETR: Lightweight DETR with SwiftNet backbone.

Public API:
    SwiftDetrTiny   — ~6M  backbone, real-time edge inference
    SwiftDetrSmall  — ~15M backbone, balanced speed / accuracy
    SwiftDetrBase   — ~30M backbone, high accuracy
"""

from swiftdetr.detr import SwiftDetr, ModelContext
from swiftdetr.variants import SwiftDetrTiny, SwiftDetrSmall, SwiftDetrBase

__all__ = [
    "SwiftDetr",
    "ModelContext",
    "SwiftDetrTiny",
    "SwiftDetrSmall",
    "SwiftDetrBase",
]
