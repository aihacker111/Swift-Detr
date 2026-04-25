# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pretrained weight path resolution for :mod:`swiftdetr.models.weights`.

This repository ships a **minimal** implementation: only **local** ``.pth`` / ``.ckpt``
paths are supported.  There is no built-in download from a registry or S3; pass an
on-disk file (for example a checkpoint you already have on Kaggle as a ``/kaggle/...``
path, or a dataset attachment).

A fuller build may replace this module with a version that maps short names to URLs
and downloads into a cache.
"""

from __future__ import annotations

import os

__all__ = ["download_pretrain_weights", "validate_pretrain_weights"]


def _resolved_path(pretrain_weights: str) -> str:
    return os.path.realpath(os.path.expanduser(str(pretrain_weights)))


def download_pretrain_weights(
    pretrain_weights: str,
    *,
    redownload: bool = False,
    validate_md5: bool = True,
) -> None:
    """No-op if *pretrain_weights* already points to an existing file.

    Parameters ``redownload`` and ``validate_md5`` are accepted for API compatibility
    with registry-based implementations; they are ignored in this minimal build.
    """
    p = _resolved_path(pretrain_weights)
    if os.path.isfile(p):
        return
    raise FileNotFoundError(
        f"Pretrained weights file not found: {pretrain_weights!r} (resolved: {p!r}). "
        "This build does not download checkpoints automatically. "
        "Set ``pretrain_weights`` to a local .pth or .ckpt that already exists, "
        "or set ``load_detection_pretrain`` to False and train from scratch."
    )


def validate_pretrain_weights(pretrain_weights: str, strict: bool = False) -> None:
    """Optionally verify *pretrain_weights*; the minimal build performs no extra checks."""
    p = _resolved_path(pretrain_weights)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"Pretrained weights file not found: {pretrain_weights!r} (resolved: {p!r})"
        )
