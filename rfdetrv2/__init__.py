# ------------------------------------------------------------------------
# SwiftDetr (RF-DETRv2 codebase)
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import os

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from rfdetrv2.detr import (
    SwiftDetr,
    SwiftDetrBase,
    SwiftDetrSmall,
    SwiftDetrTiny,
    # Deprecated names
    RFDETRBase,
    RFDETRLarge,
    RFDETRNano,
    RFDETRSmall,
    RFDETRV2,
)

__all__ = [
    "SwiftDetr",
    "SwiftDetrTiny",
    "SwiftDetrSmall",
    "SwiftDetrBase",
    "RFDETRV2",
    "RFDETRNano",
    "RFDETRSmall",
    "RFDETRBase",
    "RFDETRLarge",
]
