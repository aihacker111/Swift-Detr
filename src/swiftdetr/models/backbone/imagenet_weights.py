"""Load ImageNet-1K pretrained SWIFTNet weights into the detection backbone.

Checkpoints from classification training may include a ``head`` and keys with
``module.`` / ``model.`` prefixes. This module strips those, keeps only
``stem`` / ``stages`` / ``mergers`` / ``norm`` tensors, and loads them into
:class:`~swiftdetr.models.backbone.backbone.SwiftNetBackbone`.encoder
(or a bare :class:`~swiftdetr.models.backbone.swiftnet.swift_net.SWIFTNet`).

Lateral 1×1 FPN projectors in ``SwiftNetBackbone`` are not in ImageNet
checkpoints; they stay randomly initialised.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys

from swiftdetr.models.backbone.backbone import SwiftNetBackbone
from swiftdetr.models.backbone.swiftnet.swift_net import SWIFTNet
from swiftdetr.util.logger import get_logger

logger = get_logger()

__all__ = [
    "load_swiftnet_imagenet_weights",
    "load_swiftnet_backbone_imagenet_weights",
    "load_swift_detr_encoder_imagenet",
]

# Strip nested Lightning / DDP / attribute prefixes (longest first).
_KEY_PREFIXES: tuple[str, ...] = (
    "model._orig_mod.",
    "backbone.0.encoder.",
    "backbone.0.",
    "backbone.encoder.",
    "model.encoder.",
    "model.backbone.0.encoder.",
    "model.backbone.0.",
    "net.encoder.",
    "net.",
    "module._orig_mod.",
    "module.",
    "model.",
    "encoder.",
    "swiftnet.",
    "backbone.",
)


def _strip_key_prefixes(key: str) -> str:
    k = key
    for _ in range(8):
        before = k
        for p in _KEY_PREFIXES:
            if k.startswith(p):
                k = k[len(p) :]
        if k == before:
            break
    return k


def _unwrap_checkpoint_object(ckpt: Any) -> dict[str, torch.Tensor]:
    """Return a flat ``name -> tensor`` map from a loaded checkpoint file."""
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected a dict checkpoint, got {type(ckpt).__name__}.")

    for name in ("state_dict", "model", "state_dict_ema", "net", "swa"):
        inner = ckpt.get(name)
        if isinstance(inner, dict):
            tensors = {k: v for k, v in inner.items() if isinstance(v, torch.Tensor)}
            if tensors:
                return tensors

    tensors = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    if not tensors:
        raise ValueError(
            "No tensor state dict found (tried 'state_dict', 'model', and top-level tensor keys).",
        )
    return tensors


def _filter_swiftnet_encoder_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Drop classification head and non-encoder keys; keep SWIFTNet trunk keys only."""
    out: dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        k = _strip_key_prefixes(raw_key)
        first = k.split(".", 1)[0] if k else ""
        if first in ("head", "fc", "classifier", "aux_head", "head_dist"):
            continue
        if first not in ("stem", "stages", "mergers", "norm"):
            continue
        out[k] = value
    if not out:
        raise ValueError(
            "After filtering, no SWIFTNet encoder keys remained. "
            "Check that the file matches the same architecture (stem/stages/mergers/norm).",
        )
    return out


def load_swiftnet_imagenet_weights(
    encoder: SWIFTNet,
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> _IncompatibleKeys:
    """Load ImageNet (or other trunk-only) weights into a :class:`SWIFTNet` module.

    Args:
        encoder: A :class:`SWIFTNet` instance.
        checkpoint_path: Path to ``.pth`` / ``.pt`` / ``.ckpt`` file.
        map_location: Device for tensors before :meth:`~torch.nn.Module.load_state_dict`.
        strict: If ``True``, :meth:`load_state_dict` raises on missing/unexpected keys.

    Returns:
        The result of :meth:`~torch.nn.Module.load_state_dict` (missing/unexpected keys).
    """
    path = os.path.realpath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {path}")

    blob = torch.load(path, map_location=map_location, weights_only=False)
    raw = _unwrap_checkpoint_object(blob)
    filtered = _filter_swiftnet_encoder_state_dict(raw)

    incompat = encoder.load_state_dict(filtered, strict=strict)
    n_miss = len(incompat.missing_keys)
    n_unexp = len(incompat.unexpected_keys)
    if n_miss or n_unexp:
        logger.info(
            "ImageNet encoder load (strict=%s): missing %d, unexpected %d keys.",
            strict,
            n_miss,
            n_unexp,
        )
        if not strict and (n_miss or n_unexp):
            if n_miss:
                logger.debug("First missing keys: %s", incompat.missing_keys[:6])
            if n_unexp:
                logger.debug("First unexpected keys: %s", incompat.unexpected_keys[:6])
    else:
        logger.info("Loaded ImageNet SWIFTNet weights from %s (all keys matched).", path)
    return incompat


def load_swiftnet_backbone_imagenet_weights(
    backbone: SwiftNetBackbone,
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> _IncompatibleKeys:
    """Load ImageNet weights into ``backbone.encoder`` (faster than passing SWIFTNet)."""
    return load_swiftnet_imagenet_weights(
        backbone.encoder,
        checkpoint_path,
        map_location=map_location,
        strict=strict,
    )


def load_swift_detr_encoder_imagenet(
    model: nn.Module,
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> _IncompatibleKeys:
    """Load into ``model.backbone[0]``, the :class:`SwiftNetBackbone` inside :class:`Joiner`.

    Accepts a :class:`SwiftDetrModel` (or any module whose ``.backbone[0]`` is a
    :class:`SwiftNetBackbone`).
    """
    joiner = model.backbone
    bb = joiner[0]
    if not isinstance(bb, SwiftNetBackbone):
        raise TypeError(
            f"model.backbone[0] must be SwiftNetBackbone, got {type(bb).__name__}.",
        )
    return load_swiftnet_backbone_imagenet_weights(
        bb,
        checkpoint_path,
        map_location=map_location,
        strict=strict,
    )
