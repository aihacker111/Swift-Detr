# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Projector — ConvNeXt + SwiGLU, tối ưu cho DINOv3 backbone.

Thay đổi so với bản cũ
-----------------------
ConvNeXtBlock:
  [FIX-1] Bỏ register_hook không cần thiết trên dw_conv.weight
  [FIX-2] GELU → SwiGLU (nhất quán với DINOv3 FFN activation)
  [FIX-3] expand_ratio = 8/3 (param-equivalent với GELU expand=4)
  [FIX-4] layer_scale_init=0 tắt layer scale, >0 bật
  [FIX-5] Thêm LayerNorm bên trong SwiGLU trước fc2 để stable

ConvNeXtFusion:
  [FIX-6] Bỏ norm1 cuối (double norm thừa — ConvNeXtBlock đã có LN)
  [FIX-7] Truyền expand_ratio và layer_scale_init xuống block

Upsample blocks (_make_upsample_block):
  [FIX-8]  scale=2.0 (P3): thêm LayerNorm + GELU sau ConvTranspose
  [FIX-9]  scale=4.0 (P2): thêm LayerNorm sau ConvTranspose thứ hai
  [FIX-10] scale=0.5 (P5): depthwise stride-2 + LN thay BN (nhất quán)

MultiScaleProjector:
  [FIX-11] P6 (scale=0.25) tách khỏi loop chính → in_dim không bị sai
  [FIX-12] extra_pool dùng kernel_size=2 stride=2 (thay kernel=1 stride=2)
  [FIX-13] _out_dim_after_scale(): tính output dim đúng per-scale
  [FIX-14] Expose expand_ratio và layer_scale_init ra ngoài để tune
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Channel-first LayerNorm cho tensor (B, C, H, W)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.size(3),), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


def get_norm(norm, out_channels):
    if norm is None:
        return None
    if isinstance(norm, str):
        if not norm:
            return None
        norm = {"LN": lambda c: LayerNorm(c)}[norm]
    return norm(out_channels)


def get_activation(name: str, inplace: bool = False) -> nn.Module:
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name in ("LeakyReLU", "leakyrelu", "lrelu"):
        return nn.LeakyReLU(0.1, inplace=inplace)
    if name is None:
        return nn.Identity()
    raise AttributeError(f"Unsupported act type: {name}")


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, groups=1,
                 dilation=1, act="relu", layer_norm=False, rms_norm=False):
        super().__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel,
            stride=stride, padding=padding, groups=groups,
            dilation=dilation, bias=False,
        )
        if rms_norm:
            self.bn = nn.RMSNorm(out_planes)
        elif layer_norm:
            self.bn = LayerNorm(out_planes)
        else:
            self.bn = nn.BatchNorm2d(out_planes)
        self.act = get_activation(act, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x.contiguous())))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5,
                 act="silu", layer_norm=False, rms_norm=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvX(c1, c_, k[0], 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(c_, c2, k[1], 1, groups=g, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,
                 act="silu", layer_norm=False, rms_norm=False):
        super().__init__()
        self.c   = int(c2 * e)
        self.cv1 = ConvX(c1, 2 * self.c, 1, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX((2 + n) * self.c, c2, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.m   = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0,
                       act=act, layer_norm=layer_norm, rms_norm=rms_norm)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ---------------------------------------------------------------------------
# SwiGLU  [FIX-2, FIX-3, FIX-5]
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU activation cho ConvNeXtBlock channel-mixing FFN.

    Kiến trúc:
        fc1: dim → hidden*2  (gate + value trong 1 linear)
        SiLU(gate) ⊙ value   (gating)
        LayerNorm(hidden)     (stable trước project back)
        fc2: hidden → dim

    Tại sao SwiGLU tốt hơn GELU ở đây:
    - DINOv3 backbone (vitb16, vits16plus) dùng SwiGLU trong FFN
      → projector nhận features từ SwiGLU-shaped distribution
      → dùng SwiGLU giảm distribution shift giữa backbone và neck
    - Gating mechanism giúp gradient flow tốt hơn qua nhiều blocks
    - Cùng param count với GELU khi dùng expand_ratio = 8/3 ≈ 2.67

    Args:
        dim:          input/output dimension
        expand_ratio: hidden = int(dim * expand_ratio)
                      Default 8/3 ≈ param-equivalent với GELU expand=4
    """

    def __init__(self, dim: int, expand_ratio: float = 8 / 3):
        super().__init__()
        hidden = int(dim * expand_ratio)
        # Gate + value gộp vào 1 linear → tiết kiệm 1 matmul so với 2 linear riêng
        self.fc1  = nn.Linear(dim, hidden * 2, bias=False)
        self.norm = nn.LayerNorm(hidden, eps=1e-6)   # [FIX-5] stable trước fc2
        self.fc2  = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C) — channel-last (sau permute trong ConvNeXtBlock)
        gate, value = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(gate) * value    # SwiGLU = Swish(gate) ⊙ value
        x = self.norm(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# ConvNeXtBlock  [FIX-1, FIX-2, FIX-3, FIX-4]
# ---------------------------------------------------------------------------

class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block với SwiGLU, tối ưu cho DINOv3 patch features.

    Kiến trúc:
        depthwise Conv2d 7×7   → spatial mixing
        LayerNorm              → normalize trước channel mixing
        SwiGLU FFN             → channel mixing với gating
        LayerScale (optional)  → stable training với pretrained backbone
        residual +

    So với bản cũ:
        GELU → SwiGLU                        [FIX-2]
        register_hook thừa đã bỏ             [FIX-1]
        expand_ratio configurable (8/3)      [FIX-3]
        layer_scale_init=0 → tắt hẳn        [FIX-4]

    Args:
        dim:              số channels
        expand_ratio:     SwiGLU hidden = int(dim * expand_ratio)
        layer_scale_init: >0 bật LayerScale với giá trị init này
                          0  tắt LayerScale hoàn toàn
    """

    def __init__(
        self,
        dim: int,
        expand_ratio: float = 8 / 3,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        # [FIX-1] Không có register_hook
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm    = nn.LayerNorm(dim, eps=1e-6)
        self.ffn     = SwiGLU(dim, expand_ratio=expand_ratio)   # [FIX-2, FIX-3]
        # [FIX-4] layer_scale_init=0 → không tạo param
        self.gamma   = (
            nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)
            if layer_scale_init > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)      # (B, C, H, W) → (B, H, W, C)
        x = self.norm(x)
        x = self.ffn(x)                 # SwiGLU channel mixing
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)      # (B, H, W, C) → (B, C, H, W)
        return residual + x


# ---------------------------------------------------------------------------
# ConvNeXtFusion  [FIX-6, FIX-7]
# ---------------------------------------------------------------------------

class ConvNeXtFusion(nn.Module):
    """Fuse multi-scale features sau khi concat, thay thế C2f.

    Kiến trúc:
        1×1 Conv  →  LayerNorm  →  N × ConvNeXtBlock

    So với bản cũ:
        Bỏ norm1 cuối (double norm thừa)                          [FIX-6]
        expand_ratio và layer_scale_init truyền xuống block       [FIX-7]

    Args:
        in_dim:           channels sau khi concat các scale features
        out_dim:          output channels (= hidden_dim của detector)
        num_blocks:       số ConvNeXtBlock
        expand_ratio:     truyền xuống SwiGLU trong mỗi block
        layer_scale_init: truyền xuống LayerScale trong mỗi block
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_blocks: int = 3,
        expand_ratio: float = 8 / 3,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.proj  = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.norm0 = LayerNorm(out_dim)
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(
                out_dim,
                expand_ratio=expand_ratio,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(num_blocks)
        ])
        # [FIX-6] Không có norm1 cuối

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm0(self.proj(x))
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Upsample / Downsample helpers  [FIX-8, FIX-9, FIX-10]
# ---------------------------------------------------------------------------

def _out_dim_after_scale(in_dim: int, scale: float) -> int:
    """Tính output channel sau sampling layer.  [FIX-13]"""
    if scale == 4.0: return in_dim // 4
    if scale == 2.0: return in_dim // 2
    if scale == 1.0: return in_dim
    if scale == 0.5: return in_dim
    raise NotImplementedError(f"Unsupported scale: {scale}")


def _make_sampling_block(in_dim: int, scale: float) -> nn.Sequential:
    """Tạo sampling block nhất quán (LN ở tất cả scales).

    scale=4.0 (P2): upsample 4×  — 2 ConvTranspose, mỗi cái có LN+GELU  [FIX-9]
    scale=2.0 (P3): upsample 2×  — ConvTranspose + LN + GELU             [FIX-8]
    scale=1.0 (P4): identity     — không làm gì
    scale=0.5 (P5): downsample 2× — depthwise stride-2 + LN + GELU       [FIX-10]

    P6 (scale=0.25) được handle riêng bằng extra_pool trong forward(),
    không đi qua hàm này.
    """
    if scale == 4.0:
        mid_dim = in_dim // 2
        out_dim = in_dim // 4
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, mid_dim, kernel_size=2, stride=2),
            LayerNorm(mid_dim),
            nn.GELU(),
            nn.ConvTranspose2d(mid_dim, out_dim, kernel_size=2, stride=2),
            LayerNorm(out_dim),      # [FIX-9] thêm norm sau ConvTranspose thứ hai
            nn.GELU(),
        )
    elif scale == 2.0:
        out_dim = in_dim // 2
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
            LayerNorm(out_dim),      # [FIX-8] bản cũ thiếu norm ở đây
            nn.GELU(),
        )
    elif scale == 1.0:
        return nn.Sequential()       # identity — không làm gì
    elif scale == 0.5:
        # [FIX-10] depthwise stride-2 + LN thay vì ConvX với BN
        # Nhất quán với ConvNeXt pipeline (không dùng BN)
        return nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2,
                      padding=1, groups=in_dim, bias=False),
            LayerNorm(in_dim),
            nn.GELU(),
        )
    else:
        raise NotImplementedError(f"Unsupported scale_factor: {scale}")


# ---------------------------------------------------------------------------
# MultiScaleProjector  [FIX-11, FIX-12, FIX-13, FIX-14]
# ---------------------------------------------------------------------------

class MultiScaleProjector(nn.Module):
    """Multi-scale feature projector với ConvNeXtFusion.

    Flow cho mỗi scale:
        backbone features (×N_layers)
            │
            ▼ sampling layers (upsample/downsample per feature)
        resampled features (×N_layers, cùng H×W)
            │
            ▼ concat theo channel
        fused tensor (B, sum_dims, H, W)
            │
            ▼ ConvNeXtFusion
        output (B, out_channels, H, W)

    Args:
        in_channels:       list channel dims từ backbone (1 per out_feature_index)
        out_channels:      output channels = hidden_dim của transformer
        scale_factors:     list float, ví dụ [2.0, 1.0, 0.5] cho P3/P4/P5
        num_blocks:        số ConvNeXtBlock trong mỗi ConvNeXtFusion
        expand_ratio:      SwiGLU expand ratio trong ConvNeXtBlock
        layer_scale_init:  LayerScale init value (0 = tắt)
        survival_prob:     stochastic depth probability (1.0 = tắt)
        use_convnext:      True → ConvNeXtFusion, False → C2f (backward compat)
    """

    def __init__(
        self,
        in_channels,
        out_channels: int,
        scale_factors,
        num_blocks: int = 3,
        layer_norm: bool = False,
        rms_norm: bool = False,
        survival_prob: float = 1.0,
        force_drop_last_n_features: int = 0,
        use_convnext: bool = True,
        expand_ratio: float = 8 / 3,        # [FIX-14]
        layer_scale_init: float = 1e-6,     # [FIX-14]
    ):
        super().__init__()

        self.scale_factors              = scale_factors
        self.survival_prob              = survival_prob
        self.force_drop_last_n_features = force_drop_last_n_features
        self.use_extra_pool             = False

        stages_sampling: list[nn.ModuleList] = []
        stages: list[nn.Module] = []

        for scale in scale_factors:

            # ----------------------------------------------------------------
            # [FIX-11] P6 tách khỏi loop chính để tránh in_dim sai
            # ----------------------------------------------------------------
            if scale == 0.25:
                self.use_extra_pool = True
                # P6 = max_pool(P5 output) — không cần sampling layers riêng
                # Không append vào stages_sampling / stages
                continue

            # Sampling: đưa mỗi backbone feature về cùng H×W của scale này
            sampling = nn.ModuleList([
                _make_sampling_block(in_dim, scale)
                for in_dim in in_channels
            ])
            stages_sampling.append(sampling)

            # [FIX-13] Tính fused input dim đúng dựa trên output của sampling
            fused_in_dim = sum(
                _out_dim_after_scale(c, scale) for c in in_channels
            )

            if use_convnext:
                stage = ConvNeXtFusion(
                    fused_in_dim,
                    out_channels,
                    num_blocks=num_blocks,
                    expand_ratio=expand_ratio,          # [FIX-14]
                    layer_scale_init=layer_scale_init,  # [FIX-14]
                )
            else:
                # C2f path — giữ nguyên để backward compat
                stage = nn.Sequential(
                    C2f(fused_in_dim, out_channels, num_blocks,
                        layer_norm=layer_norm, rms_norm=rms_norm),
                    get_norm("LN", out_channels),
                )
            stages.append(stage)

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages          = nn.ModuleList(stages)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        num_features = len(x)

        # ------------------------------------------------------------------
        # Stochastic depth / feature drop (training only)
        # ------------------------------------------------------------------
        if self.survival_prob < 1.0 and self.training:
            final_drop_prob = 1.0 - self.survival_prob
            drop_p = np.random.uniform()
            for i in range(1, num_features):
                critical_drop_prob = i * (final_drop_prob / (num_features - 1))
                if drop_p < critical_drop_prob:
                    x[i] = torch.zeros_like(x[i])
        elif self.force_drop_last_n_features > 0:
            for i in range(self.force_drop_last_n_features):
                x[-(i + 1)] = torch.zeros_like(x[-(i + 1)])

        # ------------------------------------------------------------------
        # Forward qua từng scale
        # ------------------------------------------------------------------
        results: list[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            # Resample mỗi backbone feature về resolution của scale này
            feat_list = [
                self.stages_sampling[i][j](x[j])
                for j in range(len(self.stages_sampling[i]))
            ]
            # Concat theo channel rồi fuse
            fused = torch.cat(feat_list, dim=1) if len(feat_list) > 1 else feat_list[0]
            results.append(stage(fused))

        # ------------------------------------------------------------------
        # P6 — [FIX-12] kernel_size=2 stride=2 (bản cũ: kernel=1 stride=2)
        # kernel=1 stride=2 chỉ lấy 1 pixel mỗi 2×2 patch → aliasing
        # kernel=2 stride=2 average trên 2×2 patch → smooth downsample
        # ------------------------------------------------------------------
        if self.use_extra_pool:
            results.append(
                F.max_pool2d(results[-1], kernel_size=2, stride=2, padding=0)
            )

        return results


# ---------------------------------------------------------------------------
# SimpleProjector — giữ nguyên
# ---------------------------------------------------------------------------

class SimpleProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, factor_kernel: bool = False):
        super().__init__()
        if not factor_kernel:
            self.convx1 = ConvX(in_dim, in_dim * 2, layer_norm=True, act="silu")
            self.convx2 = ConvX(in_dim * 2, out_dim, layer_norm=True, act="silu")
        else:
            self.convx1 = ConvX(in_dim, out_dim, kernel=(3, 1), layer_norm=True, act="silu")
            self.convx2 = ConvX(out_dim, out_dim, kernel=(1, 3), layer_norm=True, act="silu")
        self.ln = get_norm("LN", out_dim)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.ln(self.convx2(self.convx1(x[0])))]