"""Backbone package for Swift-DETR.

Public API:
    build_backbone  — assemble Joiner(SwiftNetBackbone, PositionEmbedding)
    Joiner          — Sequential wrapper that adds positional embeddings
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from swiftdetr.models.backbone.backbone import SwiftNetBackbone
from swiftdetr.models.backbone.imagenet_weights import (
    load_swift_detr_encoder_imagenet,
    load_swiftnet_backbone_imagenet_weights,
    load_swiftnet_imagenet_weights,
)
from swiftdetr.models.position_encoding import build_position_encoding
from swiftdetr.util.tensors import NestedTensor

__all__ = [
    "build_backbone",
    "Joiner",
    "load_swift_detr_encoder_imagenet",
    "load_swiftnet_backbone_imagenet_weights",
    "load_swiftnet_imagenet_weights",
]


class Joiner(nn.Sequential):
    """Sequential backbone + position embedding.

    Args:
        backbone: SwiftNetBackbone instance.
        position_embedding: Sinusoidal or learnable PE module.
    """

    def __init__(self, backbone: SwiftNetBackbone, position_embedding: nn.Module) -> None:
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(self, tensor_list: NestedTensor) -> tuple[list[NestedTensor], list]:
        features = self[0](tensor_list)
        pos = [self[1](feat, align_dim_orders=False).to(feat.tensors.dtype) for feat in features]
        return features, pos

    def export(self) -> None:
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export  # type: ignore[method-assign]
        for _name, m in self.named_modules():
            if (
                hasattr(m, "export")
                and isinstance(m.export, Callable)
                and hasattr(m, "_export")
                and not m._export
            ):
                m.export()

    def forward_export(self, inputs: torch.Tensor) -> tuple[list, None, list]:
        feats, masks = self[0](inputs)
        poss = [self[1](mask, align_dim_orders=False).to(feat.dtype) for feat, mask in zip(feats, masks)]
        return feats, None, poss


def build_backbone(
    encoder: str,
    drop_path: float,
    out_channels: int,
    projector_scale: list[str],
    hidden_dim: int,
    position_embedding: str,
    freeze_encoder: bool,
    gradient_checkpointing: bool,
    positional_encoding_size: int,
    projector_num_blocks: int = 3,
    projector_expand_ratio: float = 8 / 3,
    projector_layer_scale_init: float = 1e-6,
) -> Joiner:
    """Build a Joiner combining SwiftNetBackbone with sinusoidal position encoding.

    Args:
        encoder: SwiftNet variant name (e.g. ``"swiftnet_tiny"``).
        drop_path: Stochastic-depth rate for SwiftNet blocks.
        out_channels: Channel dimension after lateral projection (= hidden_dim).
        projector_scale: FPN levels to produce, e.g. ``["P3","P4","P5"]``.
        hidden_dim: Transformer hidden dimension (used for PE construction).
        position_embedding: PE type — currently only ``"sine"`` is supported.
        freeze_encoder: Freeze SwiftNet weights when ``True``.
        gradient_checkpointing: Reserved for future gradient-checkpointing support.
        positional_encoding_size: Grid side length for the sinusoidal PE.
        projector_num_blocks: ConvNeXtBlock count per fusion level.
        projector_expand_ratio: SwiGLU expand ratio in ConvNeXtBlock.
        projector_layer_scale_init: LayerScale init value; 0 disables it.

    Returns:
        ``Joiner(SwiftNetBackbone, PositionEmbedding)``.
    """
    pe = build_position_encoding(hidden_dim, position_embedding)
    backbone = SwiftNetBackbone(
        variant=encoder,
        out_channels=out_channels,
        drop_path_rate=drop_path,
        freeze_encoder=freeze_encoder,
        projector_scale=projector_scale,
        projector_num_blocks=projector_num_blocks,
        projector_expand_ratio=projector_expand_ratio,
        projector_layer_scale_init=projector_layer_scale_init,
    )
    return Joiner(backbone, pe)
