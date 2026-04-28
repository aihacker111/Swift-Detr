"""Swift-DETR model and builder functions.

Architecture: SwiftNetBackbone → MultiScale FPN → DETR Transformer
(Group DETR v3, RoPE 2D decoder self-attention).
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Callable, Optional

import torch
from torch import nn

if TYPE_CHECKING:
    from swiftdetr.config import ModelConfig, TrainConfig

from swiftdetr.models._defaults import MODEL_DEFAULTS, ModelDefaults
from swiftdetr.models._types import BuilderArgs
from swiftdetr.models.backbone import build_backbone
from swiftdetr.models.criterion import (  # noqa: F401 — re-exported for backward compat
    SetCriterion,
    dice_loss,
    dice_loss_jit,
    position_supervised_loss,
    sigmoid_ce_loss,
    sigmoid_ce_loss_jit,
    sigmoid_focal_loss,
    sigmoid_varifocal_loss,
)
from swiftdetr.models.matcher import build_matcher
from swiftdetr.models.math import MLP
from swiftdetr.models.postprocess import PostProcess
from swiftdetr.models.transformer import build_transformer
from swiftdetr.util.tensors import NestedTensor, nested_tensor_from_tensor_list

__all__ = ["SwiftDetrModel", "build_model", "build_criterion_and_postprocessors"]


def _resize_linear(linear: nn.Linear, num_classes: int) -> nn.Linear:
    """Resize a Linear layer to num_classes outputs (tile or truncate weights)."""
    base = linear.weight.shape[0]
    num_repeats = int(math.ceil(num_classes / base))
    new_weight = linear.weight.detach().repeat(num_repeats, 1)[:num_classes]
    new_bias = linear.bias.detach().repeat(num_repeats)[:num_classes] if linear.bias is not None else None
    new_linear = nn.Linear(linear.in_features, num_classes, bias=new_bias is not None)
    with torch.no_grad():
        new_linear.weight.copy_(new_weight)
        if new_bias is not None and new_linear.bias is not None:
            new_linear.bias.copy_(new_bias)
    new_linear.weight.requires_grad = linear.weight.requires_grad
    if linear.bias is not None and new_linear.bias is not None:
        new_linear.bias.requires_grad = linear.bias.requires_grad
    return new_linear


class SwiftDetrModel(nn.Module):
    """Group DETR v3 detection model with SwiftNet backbone.

    Args:
        backbone: Joiner(SwiftNetBackbone, PositionEmbedding).
        transformer: DETR transformer encoder+decoder.
        segmentation_head: Optional segmentation head (``None`` for detection-only).
        num_classes: Number of classes including background.
        num_queries: Number of object queries.
        aux_loss: Whether to return auxiliary decoder outputs.
        group_detr: Number of query groups for Group DETR training.
        two_stage: Enable two-stage DETR (encoder proposals as initial queries).
        lite_refpoint_refine: Use lightweight reference-point refinement.
        bbox_reparam: Reparameterize bounding-box predictions.
    """

    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        segmentation_head: Optional[nn.Module],
        num_classes: int,
        num_queries: int,
        aux_loss: bool = False,
        group_detr: int = 1,
        two_stage: bool = False,
        lite_refpoint_refine: bool = False,
        bbox_reparam: bool = False,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.segmentation_head = segmentation_head

        self.refpoint_embed = nn.Embedding(num_queries * group_detr, 4)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr

        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam
        self.two_stage = two_stage

        # Focal-loss bias initialisation
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)]
            )
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)]
            )

        self._export = False

    def reinitialize_detection_head(self, num_classes: int) -> None:
        """Resize the detection head to a new number of classes."""
        self.class_embed = _resize_linear(self.class_embed, num_classes)
        if self.two_stage:
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [_resize_linear(m, num_classes) for m in self.transformer.enc_out_class_embed]
            )

    def export(self) -> None:
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export  # type: ignore[method-assign]
        for _name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    def forward(self, samples: NestedTensor | list | torch.Tensor, targets=None) -> dict:
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, poss = self.backbone(samples)
        srcs, masks = [], []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)

        if self.training:
            refpoint_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            refpoint_weight = self.refpoint_embed.weight[: self.num_queries]
            query_feat_weight = self.query_feat.weight[: self.num_queries]

        if self.segmentation_head is not None:
            seg_fwd = self.segmentation_head.sparse_forward if self.training else self.segmentation_head.forward

        hs, ref_unsigmoid, memory_ts, boxes_ts = self.transformer(
            srcs, masks, poss, refpoint_weight, query_feat_weight
        )

        out: dict = {}

        if hs is not None:
            if self.bbox_reparam:
                delta = self.bbox_embed(hs)
                cxcy = delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                wh = delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.cat([cxcy, wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            outputs_class = self.class_embed(hs)

            if self.segmentation_head is not None:
                outputs_masks = seg_fwd(features[0].tensors, hs, samples.tensors.shape[-2:])

            out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
            if self.segmentation_head is not None:
                out["pred_masks"] = outputs_masks[-1]
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(
                    outputs_class,
                    outputs_coord,
                    outputs_masks if self.segmentation_head is not None else None,
                )

        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            memory_ts_list = memory_ts.chunk(group_detr, dim=1)
            cls_enc = torch.cat(
                [self.transformer.enc_out_class_embed[g](memory_ts_list[g]) for g in range(group_detr)],
                dim=1,
            )
            if hs is not None:
                out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": boxes_ts}
            else:
                out = {"pred_logits": cls_enc, "pred_boxes": boxes_ts}

        return out

    def forward_export(self, tensors: torch.Tensor) -> tuple:
        srcs, _, poss = self.backbone(tensors)
        refpoint_weight = self.refpoint_embed.weight[: self.num_queries]
        query_feat_weight = self.query_feat.weight[: self.num_queries]
        hs, ref_unsigmoid, memory_ts, boxes_ts = self.transformer(
            srcs, None, poss, refpoint_weight, query_feat_weight
        )
        if hs is not None:
            if self.bbox_reparam:
                delta = self.bbox_embed(hs)
                cxcy = delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                wh = delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.cat([cxcy, wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
            outputs_class = self.class_embed(hs)
        else:
            outputs_class = self.transformer.enc_out_class_embed[0](memory_ts)
            outputs_coord = boxes_ts
        return outputs_coord, outputs_class

    @torch.jit.unused
    def _set_aux_loss(
        self,
        outputs_class: torch.Tensor,
        outputs_coord: torch.Tensor,
        outputs_masks: Optional[torch.Tensor],
    ) -> list[dict]:
        if outputs_masks is not None:
            return [
                {"pred_logits": a, "pred_boxes": b, "pred_masks": c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_masks[:-1])
            ]
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def update_dropout(self, drop_rate: float) -> None:
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate


# ── Builder functions ──────────────────────────────────────────────────────


def _load_encoder_imagenet_if_set(
    args: "BuilderArgs",
    backbone: nn.Module,
    full_model: SwiftDetrModel | None,
) -> None:
    """Load optional SWIFTNet ImageNet trunk weights (see ``encoder_imagenet_weights``)."""
    path = getattr(args, "encoder_imagenet_weights", None)
    if not path:
        return
    from swiftdetr.models.backbone.imagenet_weights import (
        load_swift_detr_encoder_imagenet,
        load_swiftnet_backbone_imagenet_weights,
        load_swiftnet_imagenet_weights,
    )

    if full_model is not None:
        load_swift_detr_encoder_imagenet(full_model, path)
        return
    snb = backbone[0]
    if getattr(args, "encoder_only", False):
        load_swiftnet_imagenet_weights(snb.encoder, path)
    elif getattr(args, "backbone_only", False):
        load_swiftnet_backbone_imagenet_weights(snb, path)


def build_model(args: "BuilderArgs") -> SwiftDetrModel:
    """Assemble a SwiftDetrModel from a builder-args namespace.

    Args:
        args: Namespace produced by ``_namespace_from_configs``.

    Returns:
        Fully initialised :class:`SwiftDetrModel`.
    """
    num_classes = args.num_classes + 1  # +1 for background
    torch.device(args.device)

    backbone = build_backbone(
        encoder=args.encoder,
        drop_path=args.drop_path,
        out_channels=args.hidden_dim,
        projector_scale=args.projector_scale,
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        gradient_checkpointing=args.gradient_checkpointing,
        positional_encoding_size=args.positional_encoding_size,
        projector_num_blocks=getattr(args, "projector_num_blocks", 3),
        projector_expand_ratio=getattr(args, "projector_expand_ratio", 8 / 3),
        projector_layer_scale_init=getattr(args, "projector_layer_scale_init", 1e-6),
    )

    if args.encoder_only:
        _load_encoder_imagenet_if_set(args, backbone, full_model=None)
        return backbone[0].encoder, None, None  # type: ignore[return-value]
    if args.backbone_only:
        _load_encoder_imagenet_if_set(args, backbone, full_model=None)
        return backbone, None, None  # type: ignore[return-value]

    args.num_feature_levels = len(args.projector_scale)
    transformer = build_transformer(args)

    segmentation_head = None
    if args.segmentation_head:
        from swiftdetr.models.heads.segmentation import SegmentationHead

        segmentation_head = SegmentationHead(
            args.hidden_dim,
            args.dec_layers,
            downsample_ratio=args.mask_downsample_ratio,
        )

    model = SwiftDetrModel(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
    )
    _load_encoder_imagenet_if_set(args, backbone, full_model=model)
    return model


def build_criterion_and_postprocessors(args: "BuilderArgs") -> tuple[SetCriterion, PostProcess]:
    """Build loss criterion and detection postprocessor.

    Args:
        args: Namespace produced by ``_namespace_from_configs``.

    Returns:
        ``(SetCriterion, PostProcess)`` tuple.
    """
    device = torch.device(args.device)
    matcher = build_matcher(args)

    weight_dict = {
        "loss_ce": args.cls_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
    }
    if args.segmentation_head:
        weight_dict["loss_mask_ce"] = args.mask_ce_loss_coef
        weight_dict["loss_mask_dice"] = args.mask_dice_loss_coef

    if args.aux_loss:
        aux_weight_dict: dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({f"{k}_{i}": v for k, v in weight_dict.items()})
        if args.two_stage:
            aux_weight_dict.update({f"{k}_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.segmentation_head:
        losses.append("masks")

    criterion_kwargs = dict(
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        losses=losses,
        group_detr=args.group_detr,
        sum_group_losses=args.sum_group_losses,
        use_varifocal_loss=args.use_varifocal_loss,
        use_position_supervised_loss=args.use_position_supervised_loss,
        ia_bce_loss=args.ia_bce_loss,
    )
    if args.segmentation_head:
        criterion_kwargs["mask_point_sample_ratio"] = args.mask_point_sample_ratio

    criterion = SetCriterion(args.num_classes + 1, **criterion_kwargs)
    criterion.to(device)

    postprocess = PostProcess(num_select=args.num_select)
    return criterion, postprocess


def build_model_from_config(
    model_config: "ModelConfig",
    train_config: Optional["TrainConfig"] = None,
    defaults: ModelDefaults = MODEL_DEFAULTS,
) -> SwiftDetrModel:
    """Build a SwiftDetrModel directly from a ModelConfig.

    Args:
        model_config: Architecture configuration.
        train_config: Training configuration (optional; a minimal dummy is used if None).
        defaults: Hardcoded architectural constants.

    Returns:
        Fully initialised :class:`SwiftDetrModel`.
    """
    from swiftdetr._namespace import _namespace_from_configs

    if defaults.encoder_only or defaults.backbone_only:
        raise ValueError(
            "build_model_from_config() requires defaults.encoder_only=False and defaults.backbone_only=False."
        )
    if train_config is None:
        from swiftdetr.config import TrainConfig

        train_config = TrainConfig(dataset_dir=".", output_dir=".")

    ns = _namespace_from_configs(model_config, train_config, defaults)
    return build_model(ns)


def build_criterion_from_config(
    model_config: "ModelConfig",
    train_config: "TrainConfig",
    defaults: ModelDefaults = MODEL_DEFAULTS,
) -> tuple[SetCriterion, PostProcess]:
    """Build criterion and postprocessor from config objects.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.
        defaults: Hardcoded architectural constants.

    Returns:
        ``(SetCriterion, PostProcess)`` tuple.
    """
    from swiftdetr._namespace import _namespace_from_configs

    ns = _namespace_from_configs(model_config, train_config, defaults)
    return build_criterion_and_postprocessors(ns)
