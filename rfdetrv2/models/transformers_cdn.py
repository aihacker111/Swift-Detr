# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Transformer class — RoPE 2D Self-Attention only.

Thay đổi so với bản gốc
------------------------
RoPE 2D Self-Attention:
  - _rotate_half: LLaMA style (even/odd pairs thay vì chunk nửa).
  - forward cos/sin: repeat_interleave(2) thay vì repeat.
  - Training path: group_detr split → RoPE → self_attn → merge.
  - Inference path: full attention, RoPE trực tiếp.

Bỏ RoPE Cross-Attention:
  - rope_ca đã được xóa khỏi TransformerDecoderLayer.
  - Cross-attention dùng with_pos_embed(tgt, query_pos) như bản gốc.

Bỏ CDN/attn_mask:
  - Transformer.forward: không có attn_mask param.
  - TransformerDecoderLayer.forward_post: chỉ còn training/inference path.
"""

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rfdetrv2.models.ops.modules import MSDeformAttn


# ---------------------------------------------------------------------------
# RoPE 2D
# ---------------------------------------------------------------------------

class RoPE2D(nn.Module):
    """2D Rotary Position Embedding cho decoder self-attention.

    Thay vì cộng sine_embed vào Q/K (absolute position encoding),
    RoPE *xoay* Q và K bằng rotation matrix phụ thuộc vào (cx, cy)
    của reference box tương ứng với mỗi query.

    Tính chất cốt lõi:
        Q_rot · K_rot  ∝  cos(θ(cx_i) - θ(cx_j)) × cos(θ(cy_i) - θ(cy_j))
        → dot product chỉ phụ thuộc RELATIVE distance (Δcx, Δcy).

    Lợi ích:
        ✓ Relative encoding: model học spatial relationship chính xác hơn.
        ✓ Extrapolate tốt hơn sang resolution khác (không cố định max_pos).
        ✓ Zero thêm trainable params (freqs là buffer cố định).
        ✓ Stable với group_detr: mỗi group có coordinate space riêng biệt.

    Dim layout (D = hidden_dim, phải chia hết cho 4):
        Dims [0 .. D//2-1]:   encode x-coordinate (cx).
        Dims [D//2 .. D-1]:   encode y-coordinate (cy).
        Trong mỗi nửa: pairs (2i, 2i+1) xoay bởi cùng frequency band
        (LLaMA style — gradient locality tốt hơn GPT-NeoX style).
    """

    def __init__(self, dim: int, max_freq: float = 10000.0):
        super().__init__()
        assert dim % 4 == 0, (
            f"RoPE2D cần dim chia hết cho 4 (x-half + y-half, mỗi half dùng "
            f"rotate_half). Nhận được dim={dim}."
        )
        quarter = dim // 4
        freqs = 1.0 / (max_freq ** (torch.arange(0, quarter).float() / quarter))
        self.register_buffer("freqs", freqs)   # (dim//4,)
        self.dim = dim

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """LLaMA style: rotate từng cặp liền kề (2i, 2i+1).

        [x0, x1, x2, x3, ...] → [-x1, x0, -x3, x2, ...]
        """
        x1 = x[..., 0::2]   # even indices: 0, 2, 4, ...
        x2 = x[..., 1::2]   # odd  indices: 1, 3, 5, ...
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(
        self,
        q: torch.Tensor,       # (..., N, D)
        k: torch.Tensor,       # (..., N, D)
        boxes: torch.Tensor,   # (..., N, 4)  cxcywh, sigmoid ∈ [0, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cx = boxes[..., 0]   # (..., N)
        cy = boxes[..., 1]   # (..., N)

        angle_x = 2 * math.pi * cx.unsqueeze(-1) * self.freqs
        angle_y = 2 * math.pi * cy.unsqueeze(-1) * self.freqs

        # repeat_interleave(2): [f0,f0, f1,f1, ...] — khớp với LLaMA _rotate_half
        cos_x = angle_x.cos().repeat_interleave(2, dim=-1)   # (..., N, dim//2)
        sin_x = angle_x.sin().repeat_interleave(2, dim=-1)
        cos_y = angle_y.cos().repeat_interleave(2, dim=-1)
        sin_y = angle_y.sin().repeat_interleave(2, dim=-1)

        cos = torch.cat([cos_x, cos_y], dim=-1)
        sin = torch.cat([sin_x, sin_y], dim=-1)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor, dim=128):
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes, unsigmoid=True):
    N_, S_, C_ = memory.shape
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        if memory_padding_mask is not None:
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        else:
            valid_H = torch.tensor([H_ for _ in range(N_)], device=memory.device)
            valid_W = torch.tensor([W_ for _ in range(N_)], device=memory.device)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
        )
        grid  = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid  = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        wh    = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)

    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class Transformer(nn.Module):

    def __init__(self, d_model=512, sa_nhead=8, ca_nhead=8, num_queries=300,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, group_detr=1,
                 two_stage=False,
                 num_feature_levels=4, dec_n_points=4,
                 lite_refpoint_refine=False,
                 decoder_norm_type='LN',
                 bbox_reparam=False,
                 use_rope=True):
        super().__init__()
        self.encoder = None

        decoder_layer = TransformerDecoderLayer(
            d_model, sa_nhead, ca_nhead, dim_feedforward,
            dropout, activation, normalize_before,
            group_detr=group_detr,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            skip_self_attn=False,
            use_rope=use_rope,
        )
        assert decoder_norm_type in ['LN', 'Identity']
        norm = {
            "LN":       lambda c: nn.LayerNorm(c),
            "Identity": lambda c: nn.Identity(),
        }
        decoder_norm = norm[decoder_norm_type](d_model)

        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            lite_refpoint_refine=lite_refpoint_refine,
            bbox_reparam=bbox_reparam,
        )

        self.two_stage = two_stage
        if two_stage:
            self.enc_output      = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(group_detr)])
            self.enc_output_norm = nn.ModuleList([nn.LayerNorm(d_model)       for _ in range(group_detr)])

        self._reset_parameters()
        self.num_queries        = num_queries
        self.d_model            = d_model
        self.dec_layers         = num_decoder_layers
        self.group_detr         = group_detr
        self.num_feature_levels = num_feature_levels
        self.bbox_reparam       = bbox_reparam
        self._export            = False

    def export(self):
        self._export = True

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def forward(self, srcs, masks, pos_embeds, refpoint_embed, query_feat):
        src_flatten           = []
        mask_flatten          = [] if masks is not None else None
        lvl_pos_embed_flatten = []
        spatial_shapes        = []
        valid_ratios          = [] if masks is not None else None

        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src       = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed_flatten.append(pos_embed)
            src_flatten.append(src)
            if masks is not None:
                mask_flatten.append(masks[lvl].flatten(1))

        memory = torch.cat(src_flatten, 1)
        if masks is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes        = torch.as_tensor(spatial_shapes, dtype=torch.long, device=memory.device)
        level_start_index     = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))

        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, unsigmoid=not self.bbox_reparam
            )
            refpoint_embed_ts, memory_ts, boxes_ts = [], [], []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                out_mem_g  = self.enc_output_norm[g_idx](self.enc_output[g_idx](output_memory))
                cls_g      = self.enc_out_class_embed[g_idx](out_mem_g)

                if self.bbox_reparam:
                    coord_delta_g = self.enc_out_bbox_embed[g_idx](out_mem_g)
                    cxcy_g = coord_delta_g[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2]
                    wh_g   = coord_delta_g[..., 2:].exp() * output_proposals[..., 2:]
                    coord_g = torch.cat([cxcy_g, wh_g], dim=-1)
                else:
                    coord_g = self.enc_out_bbox_embed[g_idx](out_mem_g) + output_proposals

                topk       = min(self.num_queries, cls_g.shape[-2])
                topk_idx_g = torch.topk(cls_g.max(-1)[0], topk, dim=1)[1]

                ref_g_undetach = torch.gather(coord_g, 1, topk_idx_g.unsqueeze(-1).repeat(1, 1, 4))
                ref_g          = ref_g_undetach.detach()
                tgt_g          = torch.gather(out_mem_g, 1, topk_idx_g.unsqueeze(-1).repeat(1, 1, self.d_model))

                refpoint_embed_ts.append(ref_g)
                memory_ts.append(tgt_g)
                boxes_ts.append(ref_g_undetach)

            refpoint_embed_ts = torch.cat(refpoint_embed_ts, dim=1)
            memory_ts         = torch.cat(memory_ts, dim=1)
            boxes_ts          = torch.cat(boxes_ts, dim=1)

        if self.dec_layers > 0:
            if query_feat.dim() == 2:
                tgt            = query_feat.unsqueeze(0).repeat(bs, 1, 1)
                refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)
            else:
                tgt            = query_feat
                refpoint_embed = refpoint_embed

            if self.two_stage:
                ts_len     = refpoint_embed_ts.shape[-2]
                ref_ts_sub = refpoint_embed[..., :ts_len, :]
                ref_sub    = refpoint_embed[..., ts_len:, :]

                if self.bbox_reparam:
                    cxcy       = ref_ts_sub[..., :2] * refpoint_embed_ts[..., 2:] + refpoint_embed_ts[..., :2]
                    wh         = ref_ts_sub[..., 2:].exp() * refpoint_embed_ts[..., 2:]
                    ref_ts_sub = torch.cat([cxcy, wh], dim=-1)
                else:
                    ref_ts_sub = ref_ts_sub + refpoint_embed_ts

                refpoint_embed = torch.cat([ref_ts_sub, ref_sub], dim=-2)

            hs, references = self.decoder(
                tgt, memory,
                memory_key_padding_mask=mask_flatten,
                pos=lvl_pos_embed_flatten,
                refpoints_unsigmoid=refpoint_embed,
                level_start_index=level_start_index,
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios.to(memory.dtype) if valid_ratios is not None else valid_ratios,
            )
        else:
            assert self.two_stage
            hs = references = None

        if self.two_stage:
            if self.bbox_reparam:
                return hs, references, memory_ts, boxes_ts
            else:
                return hs, references, memory_ts, boxes_ts.sigmoid()
        return hs, references, None, None


# ---------------------------------------------------------------------------
# TransformerDecoder
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False, d_model=256,
                 lite_refpoint_refine=False, bbox_reparam=False):
        super().__init__()
        self.layers               = _get_clones(decoder_layer, num_layers)
        self.num_layers           = num_layers
        self.d_model              = d_model
        self.norm                 = norm
        self.return_intermediate  = return_intermediate
        self.lite_refpoint_refine = lite_refpoint_refine
        self.bbox_reparam         = bbox_reparam
        self.ref_point_head       = MLP(2 * d_model, d_model, d_model, 2)
        self._export              = False

    def export(self):
        self._export = True

    def refpoints_refine(self, refpoints_unsigmoid, new_refpoints_delta):
        if self.bbox_reparam:
            cxcy = new_refpoints_delta[..., :2] * refpoints_unsigmoid[..., 2:] + refpoints_unsigmoid[..., :2]
            wh   = new_refpoints_delta[..., 2:].exp() * refpoints_unsigmoid[..., 2:]
            return torch.cat([cxcy, wh], dim=-1)
        return refpoints_unsigmoid + new_refpoints_delta

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,
                level_start_index: Optional[Tensor] = None,
                spatial_shapes: Optional[Tensor] = None,
                valid_ratios: Optional[Tensor] = None):
        output = tgt

        intermediate           = []
        hs_refpoints_unsigmoid = [refpoints_unsigmoid]

        def get_reference(refpoints):
            obj_center = refpoints[..., :4]
            if self._export:
                query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model / 2)
                refpoints_input  = obj_center[:, :, None]
            else:
                refpoints_input  = obj_center[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                query_sine_embed = gen_sineembed_for_position(
                    refpoints_input[:, :, 0, :], self.d_model / 2)
            query_pos = self.ref_point_head(query_sine_embed)
            return obj_center, refpoints_input, query_pos, query_sine_embed

        if self.lite_refpoint_refine:
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
            else:
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

        for layer_id, layer in enumerate(self.layers):
            if not self.lite_refpoint_refine:
                if self.bbox_reparam:
                    obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid)
                else:
                    obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid())

            query_pos = query_pos * 1  # pos_transformation = 1

            output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos, query_pos=query_pos,
                query_sine_embed=query_sine_embed,
                is_first=(layer_id == 0),
                reference_points=refpoints_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                ref_boxes=obj_center,    # ← RoPE SA: pass box positions
            )

            if not self.lite_refpoint_refine:
                new_refpoints_delta     = self.bbox_embed(output)
                new_refpoints_unsigmoid = self.refpoints_refine(refpoints_unsigmoid, new_refpoints_delta)
                if layer_id != self.num_layers - 1:
                    hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
                refpoints_unsigmoid = new_refpoints_unsigmoid.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self._export:
                hs  = intermediate[-1]
                ref = hs_refpoints_unsigmoid[-1] if self.bbox_embed is not None else refpoints_unsigmoid
                return hs, ref
            if self.bbox_embed is not None:
                return [torch.stack(intermediate), torch.stack(hs_refpoints_unsigmoid)]
            else:
                return [torch.stack(intermediate), refpoints_unsigmoid.unsqueeze(0)]

        return output.unsqueeze(0)


# ---------------------------------------------------------------------------
# TransformerDecoderLayer — RoPE 2D Self-Attention only
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, sa_nhead, ca_nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, group_detr=1,
                 num_feature_levels=4, dec_n_points=4, skip_self_attn=False,
                 use_rope: bool = True):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, sa_nhead, dropout=dropout, batch_first=True)
        self.dropout1   = nn.Dropout(dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformAttn(d_model, n_levels=num_feature_levels,
                                       n_heads=ca_nhead, n_points=dec_n_points)
        self.nhead      = ca_nhead
        self.linear1    = nn.Linear(d_model, dim_feedforward)
        self.dropout    = nn.Dropout(dropout)
        self.linear2    = nn.Linear(dim_feedforward, d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout2   = nn.Dropout(dropout)
        self.dropout3   = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.group_detr = group_detr

        # RoPE 2D cho self-attention only
        # rope_ca đã được xóa — cross-attention dùng with_pos_embed như gốc
        self.rope = RoPE2D(d_model) if use_rope else None

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, query_pos=None, query_sine_embed=None,
                     is_first=False, reference_points=None,
                     spatial_shapes=None, level_start_index=None,
                     ref_boxes=None):
        """
        ref_boxes: (B, N, 4) sigmoid cxcywh — box positions cho RoPE SA.
        """
        bs, num_queries, _ = tgt.shape

        q = tgt + query_pos
        k = tgt + query_pos
        v = tgt

        # ------------------------------------------------------------------
        # Self-Attention với RoPE 2D
        # Training: group_detr split → RoPE → self_attn → merge
        # Inference: full attention (1 group) → RoPE → self_attn
        # ------------------------------------------------------------------
        if self.training:
            g          = self.group_detr
            split_size = num_queries // g

            q_s = torch.cat(q.split(split_size, dim=1), dim=0)   # (B*g, N//g, D)
            k_s = torch.cat(k.split(split_size, dim=1), dim=0)
            v_s = torch.cat(v.split(split_size, dim=1), dim=0)

            if self.rope is not None and ref_boxes is not None:
                rb_s = torch.cat(ref_boxes.split(split_size, dim=1), dim=0)  # (B*g, N//g, 4)
                q_s, k_s = self.rope(q_s, k_s, rb_s)

            tgt2 = self.self_attn(
                q_s, k_s, v_s,
                attn_mask=None,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
            )[0]
            tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)

        else:
            # Inference: 1 group, không split
            if self.rope is not None and ref_boxes is not None:
                q, k = self.rope(q, k, ref_boxes)

            tgt2 = self.self_attn(
                q, k, v,
                attn_mask=None,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
            )[0]

        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # ------------------------------------------------------------------
        # Cross-Attention — không có RoPE, giống bản gốc
        # ------------------------------------------------------------------
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points, memory,
            spatial_shapes, level_start_index,
            memory_key_padding_mask,
        )

        tgt  = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt  = self.norm3(tgt + self.dropout3(tgt2))
        return tgt

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None, query_sine_embed=None,
                is_first=False, reference_points=None,
                spatial_shapes=None, level_start_index=None,
                ref_boxes=None):
        return self.forward_post(
            tgt, memory, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask,
            pos, query_pos, query_sine_embed, is_first,
            reference_points, spatial_shapes, level_start_index,
            ref_boxes=ref_boxes,
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_transformer(args):
    try:
        two_stage = args.two_stage
    except Exception:
        two_stage = False

    return Transformer(
        d_model=args.hidden_dim,
        sa_nhead=args.sa_nheads,
        ca_nhead=args.ca_nheads,
        num_queries=args.num_queries,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
        group_detr=args.group_detr,
        two_stage=two_stage,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        lite_refpoint_refine=args.lite_refpoint_refine,
        decoder_norm_type=args.decoder_norm,
        bbox_reparam=args.bbox_reparam,
        use_rope=getattr(args, "use_rope", True),
    )


def _get_activation_fn(activation):
    if activation == "relu":  return F.relu
    if activation == "gelu":  return F.gelu
    if activation == "glu":   return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")