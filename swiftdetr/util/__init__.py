# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Utility functions and helpers."""

from swiftdetr.util import box_ops
from swiftdetr.util.distributed import (
    all_gather,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    reduce_dict,
    save_on_master,
)
from swiftdetr.util.logger import get_logger
from swiftdetr.util.package import get_sha, get_version
from swiftdetr.util.reproducibility import seed_all
from swiftdetr.util.state_dict import clean_state_dict, strip_checkpoint
from swiftdetr.util.tensors import (
    NestedTensor,
    collate_fn,
    make_collate_fn,
    nested_tensor_from_tensor_list,
)

__all__ = [
    # distributed
    "all_gather",
    "get_rank",
    "get_world_size",
    "is_dist_avail_and_initialized",
    "is_main_process",
    "reduce_dict",
    "save_on_master",
    # tensors
    "NestedTensor",
    "collate_fn",
    "make_collate_fn",
    "nested_tensor_from_tensor_list",
    # box_ops (submodule)
    "box_ops",
    # logger
    "get_logger",
    # package
    "get_sha",
    "get_version",
    # reproducibility
    "seed_all",
    # state_dict
    "clean_state_dict",
    "strip_checkpoint",
]
