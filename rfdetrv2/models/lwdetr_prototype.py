# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
LW-DETR model and criterion classes.
"""

import copy
import math
from typing import Callable, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

from rfdetrv2.models.backbone import build_backbone
from rfdetrv2.models.matcher import build_matcher
from rfdetrv2.models.segmentation_head import (
    SegmentationHead,
    calculate_uncertainty,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from rfdetrv2.models.transformers_cdn import build_transformer
from rfdetrv2.util import box_ops
from rfdetrv2.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)


# ---------------------------------------------------------------------------
# Prototype memory (EMA class prototypes for alignment loss)
# ---------------------------------------------------------------------------

class PrototypeMemory(nn.Module):
    """Maintains L2-normalized class prototypes via EMA over matched query features."""

    def __init__(self, num_classes: int, feat_dim: int, momentum: float = 0.999):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(num_classes, feat_dim))
        self.register_buffer("proto_mask", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        """feats: (N, D), detached, matching labels; labels in [0, num_classes-1]."""
        if feats.numel() == 0:
            return
        device = feats.device
        dtype = feats.dtype
        d = feats.shape[1]
        sum_buf = torch.zeros(self.num_classes, d, device=device, dtype=dtype)
        cnt_buf = torch.zeros(self.num_classes, device=device, dtype=dtype)
        for c in labels.unique():
            ci = int(c.item())
            if ci < 0 or ci >= self.num_classes:
                continue
            m = labels == c
            sum_buf[ci] = feats[m].sum(0)
            cnt_buf[ci] = m.sum().to(dtype)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(sum_buf, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(cnt_buf, op=torch.distributed.ReduceOp.SUM)
        for c in range(self.num_classes):
            if cnt_buf[c] < 0.5:
                continue
            cls_feat = sum_buf[c] / cnt_buf[c]
            cls_feat = F.normalize(cls_feat.unsqueeze(0), dim=-1).squeeze(0)
            if not self.proto_mask[c]:
                self.prototypes[c].copy_(cls_feat)
                self.proto_mask[c] = True
            else:
                p = self.momentum * self.prototypes[c] + (1 - self.momentum) * cls_feat
                p = F.normalize(p.unsqueeze(0), dim=-1).squeeze(0)
                self.prototypes[c].copy_(p)


# ---------------------------------------------------------------------------
# LWDETR
# ---------------------------------------------------------------------------

class LWDETR(nn.Module):
    """RF-DETR main model."""

    def __init__(self,
                 backbone,
                 transformer,
                 segmentation_head,
                 num_classes,
                 num_queries,
                 aux_loss=False,
                 group_detr=1,
                 two_stage=False,
                 lite_refpoint_refine=False,
                 bbox_reparam=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed  = MLP(hidden_dim, hidden_dim, 4, 3)
        self.segmentation_head = segmentation_head

        query_dim = 4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat     = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)
        self.backbone   = backbone
        self.aux_loss   = aux_loss
        self.group_detr = group_detr

        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)])
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)])

        self._export = False
        
        
    # ------------------------------------------------------------------
    # Standard methods
    # ------------------------------------------------------------------

    def reinitialize_detection_head(self, num_classes):
        base = self.class_embed.weight.shape[0]
        num_repeats = int(math.ceil(num_classes / base))
        self.class_embed.weight.data = self.class_embed.weight.data.repeat(num_repeats, 1)[:num_classes]
        self.class_embed.bias.data   = self.class_embed.bias.data.repeat(num_repeats)[:num_classes]
        if self.two_stage:
            for enc_out_class_embed in self.transformer.enc_out_class_embed:
                enc_out_class_embed.weight.data = enc_out_class_embed.weight.data.repeat(num_repeats, 1)[:num_classes]
                enc_out_class_embed.bias.data   = enc_out_class_embed.bias.data.repeat(num_repeats)[:num_classes]

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, samples: NestedTensor, targets=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs, masks = [], []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight     = self.query_feat.weight
        else:
            refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
            query_feat_weight     = self.query_feat.weight[:self.num_queries]

        if self.segmentation_head is not None:
            seg_head_fwd = (self.segmentation_head.sparse_forward
                            if self.training else self.segmentation_head.forward)

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight)

        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy  = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh    = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord       = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            outputs_class = self.class_embed(hs)

            if self.segmentation_head is not None:
                outputs_masks = seg_head_fwd(features[0].tensors, hs, samples.tensors.shape[-2:])

            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

            if self.segmentation_head is not None:
                out['pred_masks'] = outputs_masks[-1]
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(
                    outputs_class, outputs_coord,
                    outputs_masks if self.segmentation_head is not None else None,
                )
            if self.training:
                out['pred_queries'] = hs[-1]

        if self.two_stage:
            group_detr  = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc     = torch.cat([
                self.transformer.enc_out_class_embed[g](hs_enc_list[g])
                for g in range(group_detr)
            ], dim=1)

            if self.segmentation_head is not None:
                masks_enc = seg_head_fwd(
                    features[0].tensors, [hs_enc,],
                    samples.tensors.shape[-2:], skip_blocks=True,
                )[0]

            if hs is not None:
                out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
                if self.segmentation_head is not None:
                    out['enc_outputs']['pred_masks'] = masks_enc
            else:
                out = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
                if self.segmentation_head is not None:
                    out['pred_masks'] = masks_enc

        return out

    def forward_export(self, tensors):
        srcs, _, poss = self.backbone(tensors)
        refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
        query_feat_weight     = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, None, poss, refpoint_embed_weight, query_feat_weight)

        outputs_masks = None
        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy  = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh    = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord       = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
            outputs_class = self.class_embed(hs)
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(srcs[0], [hs,], tensors.shape[-2:])[0]
        else:
            assert self.two_stage
            outputs_class = self.transformer.enc_out_class_embed[0](hs_enc)
            outputs_coord = ref_enc
            if self.segmentation_head is not None:
                outputs_masks = self.segmentation_head(
                    srcs[0], [hs_enc,], tensors.shape[-2:], skip_blocks=True)[0]

        if outputs_masks is not None:
            return outputs_coord, outputs_class, outputs_masks
        return outputs_coord, outputs_class

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_masks):
        if outputs_masks is not None:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_masks[:-1])]
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)]
        for i in range(vit_encoder_num_layers):
            if hasattr(self.backbone[0].encoder, 'blocks'):
                if hasattr(self.backbone[0].encoder.blocks[i].drop_path, 'drop_prob'):
                    self.backbone[0].encoder.blocks[i].drop_path.drop_prob = dp_rates[i]
            else:
                if hasattr(self.backbone[0].encoder.trunk.blocks[i].drop_path, 'drop_prob'):
                    self.backbone[0].encoder.trunk.blocks[i].drop_path.drop_prob = dp_rates[i]

    def update_dropout(self, drop_rate):
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate


# ---------------------------------------------------------------------------
# SetCriterion
# ---------------------------------------------------------------------------

class SetCriterion(nn.Module):
    def __init__(self,
                 num_classes,
                 matcher,
                 weight_dict,
                 focal_alpha,
                 losses,
                 group_detr=1,
                 sum_group_losses=False,
                 use_varifocal_loss=False,
                 use_position_supervised_loss=False,
                 ia_bce_loss=False,
                 mask_point_sample_ratio: int = 16,
                 prototype_memory: Optional[PrototypeMemory] = None,
                 prototype_warmup_epochs: float = 0.0,
                 prototype_repulsion_coef: float = 0.0,
                 ):
        super().__init__()
        self.num_classes                  = num_classes
        self.matcher                      = matcher
        self.weight_dict                  = weight_dict
        self.losses                       = losses
        self.focal_alpha                  = focal_alpha
        self.group_detr                   = group_detr
        self.sum_group_losses             = sum_group_losses
        self.use_varifocal_loss           = use_varifocal_loss
        self.use_position_supervised_loss = use_position_supervised_loss
        self.ia_bce_loss                  = ia_bce_loss
        self.mask_point_sample_ratio      = mask_point_sample_ratio
        self.prototype_memory             = prototype_memory
        self.prototype_warmup_epochs      = prototype_warmup_epochs
        self.prototype_repulsion_coef     = prototype_repulsion_coef
        self.register_buffer("_epoch_buf", torch.tensor(0, dtype=torch.long))

    def set_epoch(self, epoch: int) -> None:
        self._epoch_buf.fill_(int(epoch))

    @property
    def current_epoch(self) -> int:
        return int(self._epoch_buf.item())

    # ------------------------------------------------------------------
    # Classification loss
    # ------------------------------------------------------------------

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:
            alpha = self.focal_alpha
            gamma = 2
            src_boxes    = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            iou_targets  = torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious    = iou_targets.clone().detach()
            prob        = src_logits.sigmoid()
            pos_weights = torch.zeros_like(src_logits)
            neg_weights = prob ** gamma
            pos_ind = [id for id in idx]
            pos_ind.append(target_classes_o)
            t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()
            pos_weights[pos_ind] = t.to(pos_weights.dtype)
            neg_weights[pos_ind] = 1 - t.to(neg_weights.dtype)
            loss_ce = (neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)).sum() / num_boxes

        elif self.use_position_supervised_loss:
            src_boxes    = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            iou_targets  = torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious_func = iou_targets.clone().detach()
            cls_iou_func_targets = torch.zeros(
                (src_logits.shape[0], src_logits.shape[1], self.num_classes),
                dtype=src_logits.dtype, device=src_logits.device)
            pos_ind = [id for id in idx]
            pos_ind.append(target_classes_o)
            cls_iou_func_targets[pos_ind] = pos_ious_func
            norm_cls = cls_iou_func_targets / (
                cls_iou_func_targets.view(cls_iou_func_targets.shape[0], -1, 1).amax(1, True) + 1e-8)
            loss_ce = position_supervised_loss(
                src_logits, norm_cls, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        elif self.use_varifocal_loss:
            src_boxes    = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            iou_targets  = torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            cls_iou_targets = torch.zeros(
                (src_logits.shape[0], src_logits.shape[1], self.num_classes),
                dtype=src_logits.dtype, device=src_logits.device)
            pos_ind = [id for id in idx]
            pos_ind.append(target_classes_o)
            cls_iou_targets[pos_ind] = pos_ious
            loss_ce = sigmoid_varifocal_loss(
                src_logits, cls_iou_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        else:
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_ce = sigmoid_focal_loss(
                src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    # ------------------------------------------------------------------
    # Box losses
    # ------------------------------------------------------------------

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device      = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred   = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        return {'cardinality_error': F.l1_loss(card_pred.float(), tgt_lengths.float())}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx          = self._get_src_permutation_idx(indices)
        src_boxes    = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        return {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_giou': loss_giou.sum() / num_boxes,
        }

    # ------------------------------------------------------------------
    # Mask loss
    # ------------------------------------------------------------------

    def loss_masks(self, outputs, targets, indices, num_boxes):
        assert 'pred_masks' in outputs
        idx        = self._get_src_permutation_idx(indices)
        pred_masks = outputs['pred_masks']

        if isinstance(pred_masks, torch.Tensor):
            src_masks = pred_masks[idx]
        else:
            spatial_features = outputs["pred_masks"]["spatial_features"]
            query_features   = outputs["pred_masks"]["query_features"]
            bias             = outputs["pred_masks"]["bias"]
            if idx[0].numel() == 0:
                src_masks = torch.tensor([], device=spatial_features.device)
            else:
                batched = []
                pbc = idx[0].unique(return_counts=True)[1]
                bi  = torch.cat((torch.zeros_like(pbc[:1]), pbc), 0).cumsum(0)
                for i in range(pbc.shape[0]):
                    bi_i = idx[0][bi[i]:bi[i+1]]
                    bx_i = idx[1][bi[i]:bi[i+1]]
                    q_i  = query_features[(bi_i, bx_i)]
                    sf_i = spatial_features[idx[0][bi[i+1]-1]]
                    batched.append(torch.einsum("chw,nc->nhw", sf_i, q_i) + bias)
                src_masks = torch.cat(batched)

        if src_masks.numel() == 0:
            return {'loss_mask_ce': src_masks.sum(), 'loss_mask_dice': src_masks.sum()}

        target_masks = torch.cat([t['masks'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        src_masks    = src_masks.unsqueeze(1)
        target_masks = target_masks.unsqueeze(1).float()

        num_points = max(
            src_masks.shape[-2],
            src_masks.shape[-2] * src_masks.shape[-1] // self.mask_point_sample_ratio,
        )
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks, lambda logits: calculate_uncertainty(logits), num_points, 3, 0.75)

        point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)
        with torch.no_grad():
            point_labels = point_sample(
                target_masks, point_coords, align_corners=False, mode="nearest").squeeze(1)

        losses = {
            "loss_mask_ce":   sigmoid_ce_loss_jit(point_logits, point_labels, num_boxes),
            "loss_mask_dice": dice_loss_jit(point_logits, point_labels, num_boxes),
        }
        del src_masks, target_masks
        return losses

    def loss_prototype_align(self, outputs, targets, indices, num_boxes, **kwargs):
        if self.prototype_memory is None or 'pred_queries' not in outputs:
            ref = outputs.get('pred_logits', outputs.get('pred_boxes'))
            dev, dt = ref.device, ref.dtype
            return {'loss_proto_align': torch.zeros((), device=dev, dtype=dt)}
        pred_q = outputs['pred_queries']
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {'loss_proto_align': torch.zeros((), device=pred_q.device, dtype=pred_q.dtype)}
        batch_idx, src_idx = idx
        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        q = pred_q[batch_idx, src_idx]
        q = F.normalize(q, dim=-1)
        self.prototype_memory.update(q.detach(), target_classes)
        mask = self.prototype_memory.proto_mask[target_classes]
        if not mask.any():
            return {'loss_proto_align': (q * 0).mean()}
        proto = self.prototype_memory.prototypes[target_classes].detach()
        cos = (q * proto).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        loss = ((1.0 - cos) * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        if self.prototype_repulsion_coef > 0 and self.prototype_memory.proto_mask.sum() > 1:
            P = self.prototype_memory.prototypes[self.prototype_memory.proto_mask]
            Pn = F.normalize(P, dim=-1)
            sim = Pn @ Pn.T
            c = sim.shape[0]
            off = ~torch.eye(c, dtype=torch.bool, device=sim.device)
            rep = (sim * off.float()).pow(2).sum() / off.float().sum().clamp(min=1.0)
            loss = loss + self.prototype_repulsion_coef * rep
        wu = self.prototype_warmup_epochs
        if wu > 0:
            scale = min(1.0, (self.current_epoch + 1) / wu)
            loss = loss * scale
        return {'loss_proto_align': loss}

    # ------------------------------------------------------------------
    # Loss dispatch
    # ------------------------------------------------------------------

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx   = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx   = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels':           self.loss_labels,
            'cardinality':      self.loss_cardinality,
            'boxes':            self.loss_boxes,
            'masks':            self.loss_masks,
            'prototype_align':  self.loss_prototype_align,
        }
        assert loss in loss_map, f'Unknown loss: {loss}'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        group_detr          = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices             = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes *= group_detr
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            if loss == 'prototype_align':
                continue
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if 'prototype_align' in self.losses and self.prototype_memory is not None:
            losses.update(self.get_loss('prototype_align', outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_i = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.losses:
                    if loss == 'prototype_align':
                        continue
                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_i, num_boxes, **kwargs)
                    losses.update({k + f'_{i}': v for k, v in l_dict.items()})

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices_e   = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                if loss == 'prototype_align':
                    continue
                kwargs = {'log': False} if loss == 'labels' else {}
                l_dict = self.get_loss(loss, enc_outputs, targets, indices_e, num_boxes, **kwargs)
                losses.update({k + '_enc': v for k, v in l_dict.items()})

        return losses


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob    = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t     = prob * targets + (1 - prob) * (1 - targets)
    loss    = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss    = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


def sigmoid_varifocal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob         = inputs.sigmoid()
    focal_weight = (targets * (targets > 0.0).float()
                    + (1 - alpha) * (prob - targets).abs().pow(gamma) * (targets <= 0.0).float())
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return (ce_loss * focal_weight).mean(1).sum() / num_boxes


def position_supervised_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob    = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss    = ce_loss * (torch.abs(targets - prob) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * (targets > 0.0).float() + (1 - alpha) * (targets <= 0.0).float()
        loss    = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    inputs      = inputs.sigmoid().flatten(1)
    numerator   = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss        = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)


# ---------------------------------------------------------------------------
# PostProcess
# ---------------------------------------------------------------------------

class PostProcess(nn.Module):
    def __init__(self, num_select=300) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_masks = outputs.get('pred_masks', None)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores     = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels     = topk_indexes % out_logits.shape[2]
        boxes      = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes      = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct    = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes        = boxes * scale_fct[:, None, :]

        results = []
        if out_masks is not None:
            for i in range(out_masks.shape[0]):
                k_idx   = topk_boxes[i]
                masks_i = torch.gather(
                    out_masks[i], 0,
                    k_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, out_masks.shape[-2], out_masks.shape[-1]))
                h, w    = target_sizes[i].tolist()
                masks_i = F.interpolate(masks_i.unsqueeze(1), size=(int(h), int(w)),
                                        mode='bilinear', align_corners=False)
                results.append({'scores': scores[i], 'labels': labels[i],
                                'boxes': boxes[i], 'masks': masks_i > 0.0})
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b}
                       for s, l, b in zip(scores, labels, boxes)]
        return results


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# build_model / build_criterion_and_postprocessors
# ---------------------------------------------------------------------------

def build_model(args):
    num_classes = args.num_classes + 1
    torch.device(args.device)

    backbone = build_backbone(
        encoder=args.encoder,
        pretrained_encoder=args.pretrained_encoder,
        drop_path=args.drop_path,
        out_channels=args.hidden_dim,
        projector_scale=args.projector_scale,
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        rms_norm=args.rms_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        use_convnext_projector=getattr(args, "use_convnext_projector", True),
    )
    if args.encoder_only:
        return backbone[0].encoder, None, None
    if args.backbone_only:
        return backbone, None, None

    args.num_feature_levels = len(args.projector_scale)
    transformer = build_transformer(args)

    segmentation_head = (
        SegmentationHead(args.hidden_dim, args.dec_layers,
                         downsample_ratio=args.mask_downsample_ratio)
        if args.segmentation_head else None
    )

    model = LWDETR(
        backbone, transformer, segmentation_head,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
    )
    return model


def build_criterion_and_postprocessors(args):
    device  = torch.device(args.device)
    matcher = build_matcher(args)
 
    weight_dict = {
        'loss_ce':   args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
    }
    if args.segmentation_head:
        weight_dict['loss_mask_ce']   = args.mask_ce_loss_coef
        weight_dict['loss_mask_dice'] = args.mask_dice_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        if args.two_stage:
            aux_weight_dict.update({k + '_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if getattr(args, 'use_prototype_align', False):
        weight_dict['loss_proto_align'] = getattr(args, 'prototype_loss_coef', 1.0)

    losses = ['labels', 'boxes', 'cardinality']
    if args.segmentation_head:
        losses.append('masks')
    if getattr(args, 'use_prototype_align', False):
        losses.append('prototype_align')

    prototype_memory = None
    if getattr(args, 'use_prototype_align', False):
        prototype_memory = PrototypeMemory(
            num_classes=args.num_classes,
            feat_dim=args.hidden_dim,
            momentum=getattr(args, 'prototype_momentum', 0.999),
        )

    criterion_kwargs = dict(
        focal_alpha=args.focal_alpha,
        losses=losses,
        group_detr=args.group_detr,
        sum_group_losses=getattr(args, 'sum_group_losses', False),
        use_varifocal_loss=args.use_varifocal_loss,
        use_position_supervised_loss=args.use_position_supervised_loss,
        ia_bce_loss=args.ia_bce_loss,
        prototype_memory=prototype_memory,
        prototype_warmup_epochs=getattr(args, 'prototype_warmup_epochs', 0.0),
        prototype_repulsion_coef=getattr(args, 'prototype_repulsion_coef', 0.0),
    )
    if args.segmentation_head:
        criterion_kwargs['mask_point_sample_ratio'] = args.mask_point_sample_ratio
 
    criterion = SetCriterion(
        args.num_classes + 1,
        matcher=matcher,
        weight_dict=weight_dict,
        **criterion_kwargs,
    )
    criterion.to(device)
    postprocess = PostProcess(num_select=args.num_select)
    return criterion, postprocess
