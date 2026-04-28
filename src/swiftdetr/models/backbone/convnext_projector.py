# ------------------------------------------------------------------------
# Swift-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ConvNeXt multi-scale projector for SwiftNet CNN backbone.

Replaces the simple lateral 1×1 convs with cross-scale ConvNeXt fusion:
each output FPN level (P3/P4/P5) is formed by resampling ALL input backbone
stages to the target resolution, concatenating along channels, then refining
with a ConvNeXtFusion block (1×1 proj + N ConvNeXtBlocks with SwiGLU).

Scale matrix (stage stride → target stride):
           stage1(s=8)  stage2(s=16)  stage3(s=32)
P3 (s=8):    1.0          2.0           4.0
P4 (s=16):   0.5          1.0           2.0
P5 (s=32):   0.25         0.5           1.0

Adapted from rfdetrv2/models/backbone/convnext_projector.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SwiftNetConvNextProjector"]


# ---------------------------------------------------------------------------
# LayerNorm (channel-first, for BCHW tensors)
# ---------------------------------------------------------------------------

class _LayerNorm(nn.Module):
    """Channel-first LayerNorm for (B, C, H, W) tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.size(3),), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# SwiGLU — gated activation used inside ConvNeXtBlock FFN
# ---------------------------------------------------------------------------

class _SwiGLU(nn.Module):
    """SwiGLU channel-mixing FFN (channel-last input/output).

    Architecture: fc1 → split → SiLU(gate) ⊙ value → LayerNorm → fc2.
    expand_ratio=8/3 is param-equivalent to GELU with expand_ratio=4.
    """

    def __init__(self, dim: int, expand_ratio: float = 8 / 3) -> None:
        super().__init__()
        hidden = int(dim * expand_ratio)
        self.fc1  = nn.Linear(dim, hidden * 2, bias=False)
        self.norm = nn.LayerNorm(hidden, eps=1e-6)
        self.fc2  = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(gate) * value
        x = self.norm(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# ConvNeXtBlock — spatial + channel mixing with optional LayerScale
# ---------------------------------------------------------------------------

class _ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block: depthwise 7×7 → LayerNorm → SwiGLU → LayerScale + residual."""

    def __init__(
        self,
        dim: int,
        expand_ratio: float = 8 / 3,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm    = nn.LayerNorm(dim, eps=1e-6)
        self.ffn     = _SwiGLU(dim, expand_ratio=expand_ratio)
        self.gamma   = (
            nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)
            if layer_scale_init > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)          # (B,C,H,W) → (B,H,W,C)
        x = self.norm(x)
        x = self.ffn(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)          # (B,H,W,C) → (B,C,H,W)
        return residual + x


# ---------------------------------------------------------------------------
# ConvNeXtFusion — project concatenated features then refine with N blocks
# ---------------------------------------------------------------------------

class _ConvNeXtFusion(nn.Module):
    """Fuse multi-scale features: 1×1 proj → LayerNorm → N ConvNeXtBlocks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_blocks: int = 3,
        expand_ratio: float = 8 / 3,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        self.proj   = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.norm0  = _LayerNorm(out_dim)
        self.blocks = nn.Sequential(*[
            _ConvNeXtBlock(out_dim, expand_ratio=expand_ratio, layer_scale_init=layer_scale_init)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm0(self.proj(x))
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Sampling helpers — resample a feature map to a target spatial scale
# ---------------------------------------------------------------------------

def _out_dim_after_scale(in_dim: int, scale: float) -> int:
    """Output channel count after the sampling block for this scale."""
    if scale == 4.0:  return in_dim // 4   # 2× ConvTranspose per step
    if scale == 2.0:  return in_dim // 2   # 1× ConvTranspose
    if scale == 1.0:  return in_dim        # identity
    if scale == 0.5:  return in_dim        # DW stride-2, channels kept
    if scale == 0.25: return in_dim        # DW stride-4 (two DW stride-2), channels kept
    raise NotImplementedError(f"Unsupported scale: {scale}")


def _make_sampling_block(in_dim: int, scale: float) -> nn.Sequential:
    """Build a resampling block:

    scale=4.0 → 4× upsample  (2× ConvTranspose → LN+GELU, twice)
    scale=2.0 → 2× upsample  (ConvTranspose → LN+GELU)
    scale=1.0 → identity
    scale=0.5 → 2× downsample (DW stride-2 → LN+GELU)
    scale=0.25→ 4× downsample (two DW stride-2 → LN+GELU each)
    """
    if scale == 4.0:
        mid_dim = in_dim // 2
        out_dim = in_dim // 4
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim,  mid_dim, kernel_size=2, stride=2),
            _LayerNorm(mid_dim), nn.GELU(),
            nn.ConvTranspose2d(mid_dim, out_dim, kernel_size=2, stride=2),
            _LayerNorm(out_dim), nn.GELU(),
        )
    elif scale == 2.0:
        out_dim = in_dim // 2
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
            _LayerNorm(out_dim), nn.GELU(),
        )
    elif scale == 1.0:
        return nn.Sequential()  # identity
    elif scale == 0.5:
        return nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2,
                      padding=1, groups=in_dim, bias=False),
            _LayerNorm(in_dim), nn.GELU(),
        )
    elif scale == 0.25:
        return nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2,
                      padding=1, groups=in_dim, bias=False),
            _LayerNorm(in_dim), nn.GELU(),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2,
                      padding=1, groups=in_dim, bias=False),
            _LayerNorm(in_dim), nn.GELU(),
        )
    else:
        raise NotImplementedError(f"Unsupported scale: {scale}")


# ---------------------------------------------------------------------------
# SwiftNetConvNextProjector — cross-scale FPN projector for SwiftNet stages
# ---------------------------------------------------------------------------

# Scale matrix: entry [i][j] is the spatial scale applied to stage j
# when producing output level i.  Rows = [P3, P4, P5], cols = [stage1, stage2, stage3].
#   stage strides = [8, 16, 32]; target strides = [8, 16, 32]
_SCALE_MATRIX: list[list[float]] = [
    [1.0,  2.0,  4.0],   # P3 (stride 8):  upsample stage2/stage3
    [0.5,  1.0,  2.0],   # P4 (stride 16): downsample stage1, upsample stage3
    [0.25, 0.5,  1.0],   # P5 (stride 32): downsample stage1/stage2
]


class SwiftNetConvNextProjector(nn.Module):
    """Cross-scale ConvNeXt projector replacing lateral_convs in SwiftNetBackbone.

    For each target FPN level, all input backbone stages are resampled to the
    target spatial resolution, concatenated along the channel axis, then refined
    by a ConvNeXtFusion block (1×1 proj + N ConvNeXtBlocks with SwiGLU FFN).

    Args:
        in_channels: channel dims for each input stage, e.g. [96, 192, 384] for small.
        out_channels: output hidden dim (= DETR hidden_dim), e.g. 256.
        num_blocks: ConvNeXtBlock count per fusion block (default 3).
        expand_ratio: SwiGLU expand ratio (default 8/3 ≈ param-equiv to GELU×4).
        layer_scale_init: LayerScale init value; 0 disables it (default 1e-6).
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 256,
        num_blocks: int = 3,
        expand_ratio: float = 8 / 3,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()

        n_targets = len(_SCALE_MATRIX)
        n_stages  = len(in_channels)

        sampling_all: list[nn.ModuleList] = []
        fusions: list[_ConvNeXtFusion] = []

        for t_idx in range(n_targets):
            scales = _SCALE_MATRIX[t_idx][:n_stages]

            stage_samplers = nn.ModuleList([
                _make_sampling_block(in_channels[s_idx], scales[s_idx])
                for s_idx in range(n_stages)
            ])
            sampling_all.append(stage_samplers)

            fused_dim = sum(
                _out_dim_after_scale(in_channels[s_idx], scales[s_idx])
                for s_idx in range(n_stages)
            )
            fusions.append(_ConvNeXtFusion(
                fused_dim, out_channels,
                num_blocks=num_blocks,
                expand_ratio=expand_ratio,
                layer_scale_init=layer_scale_init,
            ))

        self.sampling = nn.ModuleList(sampling_all)
        self.fusions  = nn.ModuleList(fusions)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            features: list of [B, Ci, Hi, Wi] tensors for each backbone stage.
        Returns:
            list of [B, out_channels, Hi, Wi] tensors for each FPN level.
        """
        results: list[torch.Tensor] = []
        for samplers, fusion in zip(self.sampling, self.fusions):
            resampled = [samplers[j](features[j]) for j in range(len(features))]
            fused = torch.cat(resampled, dim=1) if len(resampled) > 1 else resampled[0]
            results.append(fusion(fused))
        return results
