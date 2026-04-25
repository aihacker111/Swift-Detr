"""Hardcoded architectural constants not exposed in ModelConfig or TrainConfig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True, slots=True)
class ModelDefaults:
    """Architectural constants for Swift-DETR.

    These values are consumed by builder functions but are not configurable
    per-variant.  Promoting a field to ``ModelConfig`` or ``TrainConfig``
    is the correct path if it needs to be tunable.

    Attributes:
        position_embedding: PE type (``"sine"``).
        dim_feedforward: FFN hidden dim in decoder layers.
        decoder_norm: Norm type in the decoder (``"LN"``).
        aux_loss: Whether to compute auxiliary losses at intermediate layers.
        focal_alpha: Focal-loss alpha parameter.
        set_cost_class: Matcher classification cost weight.
        set_cost_bbox: Matcher L1 bbox cost weight.
        set_cost_giou: Matcher GIoU cost weight.
        bbox_loss_coef: Bbox regression loss coefficient.
        giou_loss_coef: GIoU loss coefficient.
        sum_group_losses: Sum (vs. average) group-DETR losses.
        use_varifocal_loss: Use varifocal loss instead of focal loss.
        use_position_supervised_loss: Use position-supervised loss.
        dropout: Dropout rate in the decoder.
        force_no_pretrain: Force-disable pretrain weight loading.
        backbone_only: Build backbone only (no decoder).
        encoder_only: Build encoder only.
        pretrained_encoder: Path/URL to a pretrained encoder (unused for SwiftNet).
    """

    position_embedding: str = "sine"
    dim_feedforward: int = 2048
    decoder_norm: str = "LN"
    aux_loss: bool = True
    focal_alpha: float = 0.25
    set_cost_class: float = 2.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    sum_group_losses: bool = False
    use_varifocal_loss: bool = False
    use_position_supervised_loss: bool = False
    dropout: float = 0.0
    force_no_pretrain: bool = False
    backbone_only: bool = False
    encoder_only: bool = False
    pretrained_encoder: Optional[str] = None


MODEL_DEFAULTS = ModelDefaults()
