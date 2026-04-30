# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules — SwiftNet edition.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetrv2.models.backbone.base import BackboneBase
from rfdetrv2.models.backbone.swiftnet_backbone import SwiftNetEncoder
from rfdetrv2.models.backbone.convnext_projector import MultiScaleProjector
from rfdetrv2.util.misc import NestedTensor

logger = logging.getLogger(__name__)

__all__ = ["Backbone"]

# ---------------------------------------------------------------------------
# Supported encoder names
# ---------------------------------------------------------------------------

_SUPPORTED_ENCODERS = frozenset({
    "swiftnet_tiny",
    "swiftnet_small",
    "swiftnet_base",
})


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class Backbone(BackboneBase):
    """RF-DETR backbone: SwiftNet encoder + multi-scale ConvNeXt projector.

    SwiftNet extracts hierarchical feature maps from 4 stages
    (H/4, H/8, H/16, H/32), resamples them to a common H/16 resolution,
    and feeds them into the existing MultiScaleProjector which produces
    the P3/P4/P5(/P6) feature pyramid consumed by the transformer.
    """

    def __init__(
        self,
        name: str,
        pretrained_encoder: str = None,
        drop_path: float = 0.0,
        out_channels: int = 256,
        projector_scale: list = None,
        freeze_encoder: bool = False,
        layer_norm: bool = False,
        rms_norm: bool = False,
        gradient_checkpointing: bool = False,
        use_convnext_projector: bool = True,
    ):
        super().__init__()

        if name not in _SUPPORTED_ENCODERS:
            raise ValueError(
                f"Unsupported encoder '{name}'. "
                f"Expected one of: {sorted(_SUPPORTED_ENCODERS)}."
            )

        size = name.split("_", maxsplit=1)[1]  # "tiny" | "small" | "base"

        self.encoder = SwiftNetEncoder(
            size=size,
            pretrained_encoder=pretrained_encoder,
            freeze=freeze_encoder,
            gradient_checkpointing=gradient_checkpointing,
        )

        # ------------------------------------------------------------------
        # Projector
        # ------------------------------------------------------------------
        self.projector_scale = (
            [projector_scale]
            if isinstance(projector_scale, str)
            else list(projector_scale)
        )
        assert len(self.projector_scale) > 0, "projector_scale must not be empty"
        assert sorted(self.projector_scale) == self.projector_scale, (
            "projector_scale must be in ascending order (e.g. ['P3','P4','P5'])."
        )

        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
            use_convnext=use_convnext_projector,
        )

        self._export = False

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        self.encoder.export()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, tensor_list: NestedTensor):
        feats = self.encoder(tensor_list.tensors)
        feats = self.projector(feats)
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(
                m[None].float(), size=feat.shape[-2:]
            ).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors: torch.Tensor):
        feats = self.encoder(tensors)
        feats = self.projector(feats)
        out_feats, out_masks = [], []
        for feat in feats:
            b, _, h, w = feat.shape
            out_masks.append(
                torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
            )
            out_feats.append(feat)
        return out_feats, out_masks

    # ------------------------------------------------------------------
    # Per-parameter learning-rate / weight-decay
    # ------------------------------------------------------------------

    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        """Build per-parameter LR / weight-decay dicts for the SwiftNet encoder.

        Applies stage-wise layer-decay: earlier stages receive lower LR so
        fine-grained spatial features are updated more conservatively than
        the semantically richer later stages.
        """
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}

        num_stages = len(self.encoder.stages)

        def _stage_lr_scale(full_name: str) -> float:
            if "encoder.stem" in full_name:
                return args.lr_vit_layer_decay ** num_stages
            for si in range(num_stages):
                if f"encoder.stages.{si}." in full_name or \
                   f"encoder.mergers.{si}." in full_name:
                    # Later stages → decay exponent closer to 0 → higher LR
                    return args.lr_vit_layer_decay ** (num_stages - si)
            return 1.0  # projector and other params: no decay

        _NO_WD = frozenset({"bias", "norm", "bn", "ls1", "ls2", "alpha", "log_tau"})

        def _no_wd(full_name: str) -> bool:
            return any(tok in full_name for tok in _NO_WD)

        for n, p in self.named_parameters():
            full_name = f"{prefix}.{n}"
            if not p.requires_grad:
                continue
            if backbone_key not in full_name:
                continue
            lr = (
                args.lr_encoder
                * _stage_lr_scale(full_name)
                * args.lr_component_decay ** 2
            )
            wd = 0.0 if _no_wd(full_name) else args.weight_decay
            named_param_lr_pairs[full_name] = {
                "params": p,
                "lr": lr,
                "weight_decay": wd,
            }

        return named_param_lr_pairs
