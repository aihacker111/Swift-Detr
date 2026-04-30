"""
Visualize RoPE2D decoder self-attention — thể hiện tương tác query-to-query theo vị trí 2D.

RoPE2D: attention score phụ thuộc khoảng cách tương đối (Δcx, Δcy) giữa các query.
Visualization:
  - Panel 1: Ref points (vị trí 2D mỗi query) + lines (attention links) giữa queries
  - Panel 2: Ma trận N×N attention, sắp theo thứ tự không gian → block structure = local attention

Usage:
    python scripts/visualize_attention.py \\
        --weights output/checkpoint_best_regular.pth \\
        --model-size base --image path/to/image.jpg \\
        --layer 1 --top-k-links 12
"""

from pathlib import Path
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import supervision as sv

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfdetrv2 import SwiftDetrBase, SwiftDetrSmall, SwiftDetrTiny
from rfdetrv2.util.coco_classes import COCO_CLASSES
from rfdetrv2.util.swiftnet_pretrained import (
    SWIFTNET_WEIGHTS_BY_SIZE,
    resolve_pretrained_encoder_path,
)


# ---------------------------------------------------------------------------
# Hook manager
# ---------------------------------------------------------------------------

class AttentionHook:
    """
    Captures self-attention weights from a specific decoder layer.

    The hook stores the raw attention weight matrix (B, nheads, N, N)
    produced by nn.MultiheadAttention when need_weights=True.
    """

    def __init__(self):
        self.attn_weights = None
        self._hook = None

    def register(self, attn_module: torch.nn.Module):
        """Register forward hook on a MultiheadAttention module."""
        def _hook_fn(module, inp, out):
            # out = (attn_output, attn_weights)
            # attn_weights shape: (B, N, N) — averaged over heads by default
            # To get per-head: pass average_attn_weights=False (PyTorch >= 1.13)
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                self.attn_weights = out[1].detach().cpu()

        self._hook = attn_module.register_forward_hook(_hook_fn)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()

    def get(self):
        return self.attn_weights


# ---------------------------------------------------------------------------
# Patch MultiheadAttention to return weights
# ---------------------------------------------------------------------------

def _patch_self_attn_to_return_weights(layer):
    """
    Monkey-patch the layer's self_attn so it returns attention weights.
    PyTorch MHA with batch_first=True defaults need_weights=False in RF-DETR.
    """
    orig_forward = layer.self_attn.forward

    def patched_forward(query, key, value, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = True   # (B, N, N) averaged over heads
        return orig_forward(query, key, value, **kwargs)

    layer.self_attn.forward = patched_forward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_class_name(cls_id, class_names):
    raw = int(cls_id)
    for key in (raw, raw + 1):
        val = class_names.get(key)
        if isinstance(val, str):
            return val
    if raw in COCO_CLASSES:
        return COCO_CLASSES[raw]
    if (raw + 1) in COCO_CLASSES:
        return COCO_CLASSES[raw + 1]
    return str(raw)


def _get_transformer(model):
    """Return the Transformer module regardless of DDP wrapping.
    SwiftDetrBase: model.model = main.Model, model.model.model = LWDetr (has .transformer).
    """
    m = model.model
    if hasattr(m, 'module'):
        m = m.module
    # main.Model wraps LWDetr; LWDetr has .transformer
    if hasattr(m, 'model') and not hasattr(m, 'transformer'):
        m = m.model
    if hasattr(m, 'module'):
        m = m.module
    return m.transformer


def _run_inference_with_hook(model, image_path, threshold, layer_idx, device):
    """
    Run a single forward pass, capture self-attention weights and ref_boxes of layer_idx.

    Returns
    -------
    detections : sv.Detections
    attn       : (N, N) numpy array — attention weights averaged over heads
    ref_boxes  : (N, 4) numpy array — cxcywh normalized [0,1] per query (RoPE2D positions)
    """
    transformer = _get_transformer(model)
    decoder_layers = transformer.decoder.layers
    n_layers = len(decoder_layers)

    # -1 = last layer
    if layer_idx < 0:
        layer_idx = n_layers + layer_idx

    if layer_idx >= n_layers or layer_idx < 0:
        raise ValueError(
            f"--layer {layer_idx} out of range, model has {n_layers} decoder layers (0 to {n_layers - 1})."
        )

    target_layer = decoder_layers[layer_idx]
    _patch_self_attn_to_return_weights(target_layer)

    # Capture ref_boxes (RoPE2D positions) when this layer runs
    ref_boxes_capture = [None]

    orig_forward = target_layer.forward

    def _patched_forward(tgt, memory, tgt_mask=None, memory_mask=None,
                        tgt_key_padding_mask=None, memory_key_padding_mask=None,
                        pos=None, query_pos=None, query_sine_embed=None,
                        is_first=False, reference_points=None,
                        spatial_shapes=None, level_start_index=None,
                        ref_boxes=None):
        if ref_boxes is not None:
            ref_boxes_capture[0] = ref_boxes.detach().cpu()
        return orig_forward(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
            is_first=is_first, reference_points=reference_points,
            spatial_shapes=spatial_shapes, level_start_index=level_start_index,
            ref_boxes=ref_boxes,
        )

    target_layer.forward = _patched_forward
    hook = AttentionHook()
    hook.register(target_layer.self_attn)

    try:
        detections = model.predict(image_path, threshold=threshold)
    finally:
        target_layer.forward = orig_forward
        hook.remove()

    attn = hook.get()  # (B, N, N) or None
    if attn is None:
        raise RuntimeError(
            "Failed to capture attention weights. "
            "Check that the decoder layer uses nn.MultiheadAttention."
        )

    attn_np = attn[0].numpy()  # (N, N)
    ref_np = ref_boxes_capture[0]
    if ref_np is not None:
        ref_np = ref_np[0].numpy()  # (N, 4) cxcywh
    else:
        ref_np = np.zeros((attn_np.shape[0], 4), dtype=np.float32)

    return detections, attn_np, ref_np


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_attention(
    image_path: str,
    detections: sv.Detections,
    attn: np.ndarray,
    ref_boxes: np.ndarray,
    class_names: dict,
    query_idx: int,
    layer_idx: int,
    save_path: str,
    threshold: float,
    num_queries: int = 300,
    top_k_links: int = 8,
):
    """
    Visualize RoPE2D query-to-query attention — rõ ràng hơn với contrast cao.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]

    n_det = len(detections)
    if n_det == 0:
        print("No detections above threshold — nothing to visualize.")
        return

    N_q = attn.shape[0]
    cx_norm = ref_boxes[:, 0]
    cy_norm = ref_boxes[:, 1]
    cx_px = cx_norm * W
    cy_px = cy_norm * H

    if query_idx >= 0:
        query_indices = [query_idx] if query_idx < n_det else []
    else:
        query_indices = list(range(n_det))

    if not query_indices:
        print(f"query_idx={query_idx} out of range ({n_det} detections).")
        return

    # 3 panels: (1) image+links, (2) matrix, (3) attention vs distance
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_img, ax_mat, ax_dist = axes

    colors = plt.cm.Set1(np.linspace(0, 1, max(n_det, 1)))

    # --- Panel 1: Image — làm tối nền để overlay nổi bật ---
    ax_img.imshow(image_rgb, alpha=0.5)
    # Ref points: chấm lớn hơn, viền đen để rõ trên mọi nền
    ax_img.scatter(cx_px, cy_px, c='lime', s=12, alpha=0.9, edgecolors='black',
                   linewidths=0.3, label='query positions', zorder=5)
    # Detected queries: chấm to, màu box
    for i in query_indices:
        ax_img.scatter([cx_px[i]], [cy_px[i]], c=[colors[i % len(colors)]], s=80,
                       alpha=0.95, edgecolors='white', linewidths=2, zorder=6)

    for i, (xyxy, conf, cls_id) in enumerate(
        zip(detections.xyxy, detections.confidence, detections.class_id)
    ):
        x1, y1, x2, y2 = xyxy
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=colors[i], facecolor='none', zorder=4,
        )
        ax_img.add_patch(rect)
        label = f"Q{i}: {_get_class_name(int(cls_id), class_names)} {conf:.2f}"
        ax_img.text(x1, y1 - 6, label, color='white', fontsize=8, fontweight='bold',
                    bbox=dict(facecolor=colors[i], alpha=0.9, pad=2), zorder=7)

    # Attention links: dày hơn, alpha mạnh hơn
    row_max = attn.max()
    for plot_i, qi in enumerate(query_indices):
        row = attn[qi]
        top_j = np.argsort(row)[::-1][:top_k_links + 1]
        xi, yi = cx_px[qi], cy_px[qi]
        for j in top_j:
            if j == qi:
                continue
            xj, yj = cx_px[j], cy_px[j]
            w = row[j]
            alpha = 0.5 + 0.5 * (w / (row_max + 1e-8))
            lw = 1.0 + 2.5 * (w / (row_max + 1e-8))
            ax_img.plot(
                [xi, xj], [yi, yj],
                color=colors[plot_i % len(colors)], alpha=alpha,
                linewidth=lw, solid_capstyle='round', zorder=3,
            )

    ax_img.set_title(
        f"Query positions + top-{top_k_links} links\n"
        f"Line thickness ∝ attention weight",
        fontsize=10, fontweight='bold',
    )
    ax_img.axis('off')

    # --- Panel 2: Attention matrix — colormap rõ, vmax từ percentile ---
    order = np.lexsort((cx_norm, cy_norm))
    attn_sorted = attn[np.ix_(order, order)]

    vmax = np.percentile(attn, 99) if attn.max() > 0 else 1e-6
    im = ax_mat.imshow(attn_sorted, cmap='hot', aspect='auto', vmin=0, vmax=vmax)
    # Vẽ band "local" dọc đường chéo
    band = 30
    for off in [-band, 0, band]:
        ax_mat.axline((0, off), (N_q, N_q + off), color='cyan', ls='--', alpha=0.4, linewidth=1)
    ax_mat.text(N_q * 0.02, N_q * 0.08, '↔ Local (RoPE2D)\n  queries gần nhau',
                fontsize=8, color='cyan', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5))
    ax_mat.set_xlabel('target query (spatial order)')
    ax_mat.set_ylabel('source query (spatial order)')
    ax_mat.set_title("Attention matrix — band = local attention", fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax_mat, shrink=0.5, label='weight')

    # --- Panel 3: Attention vs spatial distance (RoPE2D proof) ---
    # Khoảng cách trong sorted order ~ khoảng cách không gian
    dist_vals = []
    attn_vals = []
    for i in range(N_q):
        for j in range(N_q):
            if i != j:
                dist_vals.append(abs(i - j))  # trong sorted order
                attn_vals.append(attn_sorted[i, j])

    # Bin theo distance
    bins = np.linspace(0, N_q, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (np.array(dist_vals) >= lo) & (np.array(dist_vals) < hi)
        if mask.any():
            bin_means.append(np.mean(np.array(attn_vals)[mask]))
        else:
            bin_means.append(0)
    bin_means = np.array(bin_means)

    bin_means_safe = np.maximum(bin_means, 1e-10)
    ax_dist.fill_between(bin_centers, bin_means_safe, alpha=0.6, color='orange')
    ax_dist.plot(bin_centers, bin_means_safe, color='red', linewidth=2)
    ax_dist.set_xlabel('Spatial distance |i − j| (in sorted order)')
    ax_dist.set_ylabel('Mean attention weight')
    ax_dist.set_title("RoPE2D: Gần nhau → attend mạnh hơn", fontsize=10, fontweight='bold')
    ax_dist.set_yscale('log')
    ax_dist.grid(True, alpha=0.3)

    plt.suptitle(
        f"RoPE2D Decoder Self-Attention — Layer {layer_idx}",
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()

    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_p), dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Attention visualization saved to: {save_p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize decoder self-attention to demonstrate RoPE2D."
    )
    parser.add_argument("--weights",            required=True, type=str)
    parser.add_argument("--model-size",         default="base",
                        choices=["tiny", "small", "base"])
    parser.add_argument("--pretrained-encoder", default=None, type=str)
    parser.add_argument("--image",              required=True, type=str)
    parser.add_argument("--threshold",          default=0.35, type=float)
    parser.add_argument("--device",             default="cuda",
                        choices=["cuda", "cpu", "mps"])
    parser.add_argument("--save",               default="output/attention_vis.png", type=str)
    parser.add_argument("--layer",              default=1, type=int,
                        help="Decoder layer: 0=early/diffuse, 1=middle (rõ nhất), 2=last. -1=last layer.")
    parser.add_argument("--query-idx",          default=-1, type=int,
                        help="-1 = all detected queries, else specific query index.")
    parser.add_argument("--num-queries",        default=300, type=int)
    parser.add_argument("--top-k-links",       default=12, type=int,
                        help="Số link attention (query→query) vẽ từ mỗi query. Default 12.")
    args = parser.parse_args()

    # Resolve pretrained encoder
    pretrained = resolve_pretrained_encoder_path(
        project_root,
        args.model_size,
        explicit=args.pretrained_encoder if args.pretrained_encoder else None,
        weights_by_size=SWIFTNET_WEIGHTS_BY_SIZE,
    )

    model_cls = {
        "tiny": SwiftDetrTiny,
        "small": SwiftDetrSmall,
        "base": SwiftDetrBase,
    }[args.model_size]

    model = model_cls(
        pretrain_weights=args.weights,
        pretrained_encoder=pretrained,
        device=args.device,
    )
    # SwiftDetr/SwiftDetrBase is a wrapper; the actual nn.Module is model.model.model
    model.model.model.eval()

    class_names = model.class_names or COCO_CLASSES

    # Resolve -1 (last layer) before use
    transformer = _get_transformer(model)
    n_layers = len(transformer.decoder.layers)
    layer_idx = args.layer if args.layer >= 0 else n_layers + args.layer

    print(f"Running inference + attention capture (layer={layer_idx})…")
    detections, attn, ref_boxes = _run_inference_with_hook(
        model, args.image, args.threshold, layer_idx, args.device
    )
    print(f"Detections: {len(detections)}  |  Attention: {attn.shape}  |  Ref boxes: {ref_boxes.shape}")

    visualize_attention(
        image_path   = args.image,
        detections   = detections,
        attn         = attn,
        ref_boxes    = ref_boxes,
        class_names  = class_names,
        query_idx    = args.query_idx,
        layer_idx    = layer_idx,
        save_path    = args.save,
        threshold    = args.threshold,
        num_queries  = args.num_queries,
        top_k_links  = args.top_k_links,
    )


if __name__ == "__main__":
    main()