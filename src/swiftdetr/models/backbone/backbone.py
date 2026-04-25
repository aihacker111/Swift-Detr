"""SwiftNet backbone wrapper for Swift-DETR.

Extracts multi-scale spatial feature maps from SwiftNet stages 1-3
(strides 8, 16, 32 → P3, P4, P5) and projects them to a uniform
hidden dimension via lateral 1×1 convolutions + LayerNorm.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from swiftdetr.models.backbone.base import BackboneBase
from swiftdetr.models.backbone.swiftnet.swift_net import SWIFTNet, swift_net_tiny, swift_net_small, swift_net_base
from swiftdetr.util.tensors import NestedTensor

__all__ = ["SwiftNetBackbone"]

# Maps projector-scale label → SwiftNet stage index (0-indexed)
_SCALE_TO_STAGE: dict[str, int] = {"P3": 1, "P4": 2, "P5": 3}

# Spatial stride for each SwiftNet stage
_STAGE_STRIDES: list[int] = [4, 8, 16, 32]

_VARIANT_FACTORIES = {
    "swiftnet_tiny": swift_net_tiny,
    "swiftnet_small": swift_net_small,
    "swiftnet_base": swift_net_base,
}


def _build_swiftnet(variant: str, drop_path_rate: float) -> SWIFTNet:
    factory = _VARIANT_FACTORIES[variant]
    return factory(pretrained=False, drop_path_rate=drop_path_rate)


class _ChannelLastLayerNorm(nn.Module):
    """LayerNorm applied in BCHW format (normalises over the channel dim)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)   # → [B, H, W, C]
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  # → [B, C, H, W]


class SwiftNetBackbone(BackboneBase):
    """SwiftNet backbone that produces P3/P4/P5 feature pyramids for Swift-DETR.

    Stage-to-scale mapping:
        Stage 0  stride  4  →  (not used)
        Stage 1  stride  8  →  P3
        Stage 2  stride 16  →  P4
        Stage 3  stride 32  →  P5

    Args:
        variant: One of ``"swiftnet_tiny"``, ``"swiftnet_small"``, ``"swiftnet_base"``.
        out_channels: Output channel dim (DETR hidden dim).
        drop_path_rate: Stochastic-depth rate for SwiftNet blocks.
        freeze_encoder: Freeze all backbone parameters when ``True``.
        projector_scale: Ordered list of FPN levels to produce (e.g. ``["P3","P4","P5"]``).
    """

    def __init__(
        self,
        variant: str,
        out_channels: int = 256,
        drop_path_rate: float = 0.0,
        freeze_encoder: bool = False,
        projector_scale: list[str] | None = None,
    ) -> None:
        super().__init__()

        assert variant in _VARIANT_FACTORIES, (
            f"Unknown SwiftNet variant '{variant}'. "
            f"Choose from: {list(_VARIANT_FACTORIES)}"
        )

        self.encoder = _build_swiftnet(variant, drop_path_rate)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale: list[str] = projector_scale or ["P3", "P4", "P5"]

        assert sorted(self.projector_scale) == self.projector_scale, (
            "projector_scale must be in ascending order: P3 < P4 < P5."
        )

        self.stage_indices: list[int] = [_SCALE_TO_STAGE[s] for s in self.projector_scale]

        # Lateral projections: SwiftNet stage channels → out_channels
        backbone_dims = self.encoder.config.dims  # [C0, C1, C2, C3]
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(backbone_dims[i], out_channels, kernel_size=1, bias=False)
            for i in self.stage_indices
        ])
        self.lateral_norms = nn.ModuleList([
            _ChannelLastLayerNorm(out_channels)
            for _ in self.stage_indices
        ])

        self._export = False

    def _tokens_to_2d(self, tokens: Tensor, h: int, w: int) -> Tensor:
        """Reshape [B, H*W, C] token tensor to [B, C, H, W] spatial map."""
        B, _N, C = tokens.shape
        return tokens.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()

    def _project(self, feat_2d: Tensor, lateral: nn.Conv2d, norm: _ChannelLastLayerNorm) -> Tensor:
        return norm(lateral(feat_2d))

    def forward(self, tensor_list: NestedTensor) -> list[NestedTensor]:
        x = tensor_list.tensors
        B, _C, H, W = x.shape

        # all_features: [stage0, stage1, stage2, stage3], each [B, N_i, C_i]
        all_features = self.encoder.get_feature_maps(x)

        # Spatial dims for each stage derived from input resolution
        stage_hw = [
            (H // 4,  W // 4),    # stage 0 – stride 4
            (H // 8,  W // 8),    # stage 1 – stride 8  → P3
            (H // 16, W // 16),   # stage 2 – stride 16 → P4
            (H // 32, W // 32),   # stage 3 – stride 32 → P5
        ]

        m = tensor_list.mask
        out: list[NestedTensor] = []

        for stage_i, lateral, norm in zip(self.stage_indices, self.lateral_convs, self.lateral_norms):
            h, w = stage_hw[stage_i]
            feat_2d = self._tokens_to_2d(all_features[stage_i], h, w)
            feat_proj = self._project(feat_2d, lateral, norm)
            mask = F.interpolate(m[None].float(), size=(h, w)).to(torch.bool)[0]
            out.append(NestedTensor(feat_proj, mask))

        return out

    def export(self) -> None:
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export  # type: ignore[method-assign]

    def forward_export(self, tensors: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        B, _C, H, W = tensors.shape
        all_features = self.encoder.get_feature_maps(tensors)
        stage_hw = [
            (H // 4,  W // 4),
            (H // 8,  W // 8),
            (H // 16, W // 16),
            (H // 32, W // 32),
        ]
        out_feats: list[Tensor] = []
        out_masks: list[Tensor] = []
        for stage_i, lateral, norm in zip(self.stage_indices, self.lateral_convs, self.lateral_norms):
            h, w = stage_hw[stage_i]
            feat_2d = self._tokens_to_2d(all_features[stage_i], h, w)
            feat_proj = self._project(feat_2d, lateral, norm)
            out_feats.append(feat_proj)
            out_masks.append(torch.zeros((B, h, w), dtype=torch.bool, device=feat_proj.device))
        return out_feats, out_masks

    def get_named_param_lr_pairs(self, args: object, prefix: str = "backbone.0") -> dict:
        """Return per-parameter LR/WD dicts with stage-based decay for the encoder.

        Earlier stages (stem, stage 0) receive a stronger LR decay than later
        stages — analogous to the layer-wise decay used for ViT backbones.
        """
        # Assign a depth index to each parameter based on its stage
        _stage_key_to_depth: dict[str, int] = {
            "stem":     0,
            "stages.0": 1,
            "mergers.0": 2,
            "stages.1": 2,
            "mergers.1": 3,
            "stages.2": 3,
            "mergers.2": 4,
            "stages.3": 4,
        }
        num_depths = 5  # 0 (earliest) … 4 (latest)
        backbone_key = f"{prefix}.encoder"
        named_param_lr_pairs: dict = {}

        lr_encoder = getattr(args, "lr_encoder", 1.5e-4)
        lr_decay = getattr(args, "lr_vit_layer_decay", 0.8)
        comp_decay = getattr(args, "lr_component_decay", 0.7)
        weight_decay = getattr(args, "weight_decay", 1e-4)

        for n, p in self.named_parameters():
            full_n = f"{prefix}.{n}"
            if backbone_key not in full_n or not p.requires_grad:
                continue
            depth = num_depths  # default: last stage (no extra decay)
            for key, d in _stage_key_to_depth.items():
                if key in n:
                    depth = d
                    break
            lr = lr_encoder * (lr_decay ** (num_depths - depth)) * (comp_decay ** 2)
            wd = 0.0 if _is_no_decay_param(n) else weight_decay
            named_param_lr_pairs[full_n] = {"params": p, "lr": lr, "weight_decay": wd}

        return named_param_lr_pairs


def _is_no_decay_param(name: str) -> bool:
    """Return True for parameters that should have zero weight decay."""
    return any(k in name for k in ("bias", "norm", "bn", ".ls1", ".ls2", ".alpha"))
