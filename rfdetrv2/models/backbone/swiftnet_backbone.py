"""
SwiftNet backbone encoder for RF-DETR.

Wraps SWIFTNet (CNN stem + Hybrid Transformer) as a feature extractor:
  - Classification head is removed; only stem, stages and mergers are used.
  - forward() extracts features from all 4 stages and resamples them to a
    common spatial resolution (stage-2: H/16, W/16) so the existing
    MultiScaleProjector can consume them unchanged.
  - _out_feature_channels lists the channel count per stage.
"""

import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetrv2.models.backbone.swiftnet.swift_net import SWIFTNet
from rfdetrv2.models.backbone.swiftnet.config import SWIFTNetConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Size configurations — mirrors SWIFTNet factory variants
# ---------------------------------------------------------------------------

SWIFTNET_CONFIGS = {
    "tiny": dict(
        dims=[32, 64, 128, 256],
        depths=[2, 2, 6, 2],
        num_heads=[1, 2, 4, 8],
        drop_path_rate=0.05,
    ),
    "small": dict(
        dims=[48, 96, 192, 384],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 12],
        drop_path_rate=0.1,
    ),
    "base": dict(
        dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 3],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.25,
    ),
}


# ---------------------------------------------------------------------------
# SwiftNetEncoder
# ---------------------------------------------------------------------------

class SwiftNetEncoder(nn.Module):
    """SwiftNet feature extractor for RF-DETR.

    Extracts feature maps from all 4 stages of SwiftNet and resamples
    them to a common spatial resolution (stage-2 output: H/16, W/16 for a
    standard 640-px input) so the MultiScaleProjector can process them
    identically to DINOv3 intermediate layer features.

    The classification head is stripped; weights from an ImageNet-1k
    pretrained checkpoint can be loaded via *pretrained_encoder*.
    """

    def __init__(
        self,
        size: str = "base",
        pretrained_encoder: Optional[str] = None,
        freeze: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        if size not in SWIFTNET_CONFIGS:
            raise ValueError(
                f"Unsupported SwiftNet size '{size}'. "
                f"Choose from: {sorted(SWIFTNET_CONFIGS)}."
            )

        cfg_kw = dict(SWIFTNET_CONFIGS[size])  # copy to avoid mutation
        # num_classes=0 → SWIFTNetHead uses Identity; we replace head anyway
        config = SWIFTNetConfig(**cfg_kw, num_classes=0)
        model  = SWIFTNet(config)

        # ── Strip the classification head ──────────────────────────────
        model.head = nn.Identity()

        self.stem       = model.stem
        self.stages     = model.stages
        self.mergers    = model.mergers
        self.num_stages = len(model.stages)

        # Stage 2 is the reference: H/16 × W/16 for a 640-px input
        self.ref_stage = 2

        # One channel count per stage (consumed by MultiScaleProjector)
        self._out_feature_channels: List[int] = list(config.dims)

        self._export = False

        if pretrained_encoder is not None:
            self._load_pretrained(pretrained_encoder)
        else:
            logger.info("SwiftNetEncoder: no pretrained weights — using random init.")

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            logger.info("SwiftNetEncoder: all encoder weights frozen.")

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------

    def _load_pretrained(self, path: str) -> None:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"SwiftNet pretrained weights not found: {p}"
            )
        checkpoint = torch.load(str(p), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)

        # Drop classification head weights — shape will not match (num_classes differs)
        head_keys = [k for k in list(state_dict.keys()) if k.startswith("head.")]
        for k in head_keys:
            del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info(
                "SwiftNetEncoder: missing keys after pretrained load "
                "(expected for removed head): %s …", missing[:5]
            )
        if unexpected:
            logger.warning(
                "SwiftNetEncoder: unexpected keys in checkpoint: %s …", unexpected[:5]
            )
        logger.info("SwiftNetEncoder: loaded pretrained weights from '%s'.", path)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self) -> None:
        self._export = True  # SwiftNet is TorchScript-compatible out of the box

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run SwiftNet and return per-stage feature maps at H/16 resolution.

        Parameters
        ----------
        x : Tensor[B, 3, H, W]

        Returns
        -------
        List of Tensor[B, dims[i], H/16, W/16], one per stage (4 total).
        """
        B = x.shape[0]
        tokens, H, W = self.stem(x)  # [B, H/4·W/4, dims[0]]

        stage_feats: List[torch.Tensor] = []
        stage_shapes: List[tuple]       = []

        for si, stage in enumerate(self.stages):
            for block in stage:
                tokens = block(tokens, H, W)

            # Tokens → spatial feature map (B, C, H_i, W_i)
            C    = tokens.shape[-1]
            feat = tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            stage_feats.append(feat)
            stage_shapes.append((H, W))

            if si < self.num_stages - 1:
                tokens, H, W = self.mergers[si](tokens, H, W)

        # Resample all stages to stage-2 resolution (H/16)
        H_ref, W_ref = stage_shapes[self.ref_stage]
        out: List[torch.Tensor] = []
        for feat in stage_feats:
            if feat.shape[2] != H_ref or feat.shape[3] != W_ref:
                feat = F.interpolate(
                    feat,
                    size=(H_ref, W_ref),
                    mode="bilinear",
                    align_corners=False,
                )
            out.append(feat)
        return out
