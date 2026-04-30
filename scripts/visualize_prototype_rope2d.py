"""
Mô phỏng Prototype EMA Memory và mối liên hệ với RoPE2D.

Hai cơ chế hoạt động ở hai chiều:
  - RoPE2D: không gian (cx, cy) — quyết định query nào attend query nào
  - Prototype: feature space — kéo query feature về prototype của class

Usage:
    # Simulation thuần (không cần model)
    python scripts/visualize_prototype_rope2d.py --mode simulate --save output/prototype_rope2d_sim.png

    # Từ model thật (cần weights + image)
    python scripts/visualize_prototype_rope2d.py --mode real --weights ... --image ... --save output/prototype_rope2d_real.png
"""

from pathlib import Path
import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.colors as mcolors

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def _get_class_name(cls_id: int, class_names: dict, use_coco_fallback: bool = True) -> str:
    """Resolve class name — theo inference.py.

    model.class_names có thể:
      A) {0: "person", ...} — 0-indexed  → dùng trực tiếp
      B) {1: "person", ...} — 1-indexed  → dùng trực tiếp
      C) {0: 1, 1: 2, ...} — remapping  → fallback COCO_CLASSES với raw id

    Với format C, raw id là COCO category_id: raw=18 → "dog".
    """
    from rfdetrv2.util.coco_classes import COCO_CLASSES

    raw = int(cls_id)
    for key in (raw, raw + 1):
        val = class_names.get(key)
        if isinstance(val, str):
            return val
    if use_coco_fallback:
        if raw in COCO_CLASSES:
            return COCO_CLASSES[raw]
        if (raw + 1) in COCO_CLASSES:
            return COCO_CLASSES[raw + 1]
    return str(raw)


# ---------------------------------------------------------------------------
# 1. EMA Simulation (thuần numpy)
# ---------------------------------------------------------------------------

def simulate_prototype_ema(
    num_classes: int = 3,
    feat_dim: int = 64,
    num_queries_per_class: int = 20,
    num_steps: int = 50,
    momentum: float = 0.999,
):
    """
    Mô phỏng PrototypeMemory EMA update với dữ liệu synthetic.
    Returns: prototypes (C, D), query features (M, D), labels (M,), history
    """
    np.random.seed(42)

    # Class centers trong feature space — mỗi class có distribution riêng
    class_centers = np.random.randn(num_classes, feat_dim) * 0.5
    class_centers = class_centers / np.linalg.norm(class_centers, axis=1, keepdims=True)

    # Query features: gần class center + noise
    all_feats = []
    all_labels = []
    for c in range(num_classes):
        center = class_centers[c]
        feats = center + np.random.randn(num_queries_per_class, feat_dim) * 0.3
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        all_feats.append(feats)
        all_labels.append(np.full(num_queries_per_class, c))

    all_feats = np.vstack(all_feats)
    all_labels = np.hstack(all_labels)

    # Prototype EMA simulation
    prototypes = np.zeros((num_classes, feat_dim))
    initialized = np.zeros(num_classes, dtype=bool)
    history = []

    for step in range(num_steps):
        # Shuffle và lấy batch (simulate training)
        perm = np.random.permutation(len(all_feats))
        feats = all_feats[perm]
        labels = all_labels[perm]

        for c in range(num_classes):
            mask = labels == c
            if not mask.any():
                continue
            cls_feat = feats[mask].mean(0)
            cls_feat = cls_feat / np.linalg.norm(cls_feat)

            if not initialized[c]:
                prototypes[c] = cls_feat
                initialized[c] = True
            else:
                prototypes[c] = momentum * prototypes[c] + (1 - momentum) * cls_feat
                prototypes[c] = prototypes[c] / np.linalg.norm(prototypes[c])

        history.append(prototypes.copy())

    return prototypes, all_feats, all_labels, class_centers, history


# ---------------------------------------------------------------------------
# 2. Visualization
# ---------------------------------------------------------------------------

def plot_schematic(ax):
    """Sơ đồ luồng: RoPE2D vs Prototype."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.5, 'RoPE2D vs Prototype EMA — hai chiều bổ trợ nhau', fontsize=12, ha='center', fontweight='bold')

    # Left branch: RoPE2D (spatial)
    ax.text(2, 4.5, 'Không gian 2D (cx, cy)', fontsize=9, ha='center', color='blue')
    ax.text(2, 4, 'Ref boxes', fontsize=8, ha='center')
    ax.text(2, 3.5, '→ RoPE2D', fontsize=8, ha='center')
    ax.text(2, 3, 'Self-attention: query gần\nnhau → attend mạnh', fontsize=7, ha='center', wrap=True)
    rect1 = mpatches.FancyBboxPatch((0.5, 2.5), 3, 2.5, boxstyle="round,pad=0.1",
                                    facecolor='lightblue', edgecolor='blue', alpha=0.7)
    ax.add_patch(rect1)

    # Right branch: Prototype (semantic)
    ax.text(7, 4.5, 'Feature space (D-dim)', fontsize=9, ha='center', color='green')
    ax.text(7, 4, 'Query features', fontsize=8, ha='center')
    ax.text(7, 3.5, '→ Prototype EMA', fontsize=8, ha='center')
    ax.text(7, 3, 'Kéo feature về\nprototype[class]', fontsize=7, ha='center', wrap=True)
    rect2 = mpatches.FancyBboxPatch((5.5, 2.5), 3, 2.5, boxstyle="round,pad=0.1",
                                    facecolor='lightgreen', edgecolor='green', alpha=0.7)
    ax.add_patch(rect2)

    # Middle: Query
    ax.text(5, 1.5, 'Mỗi query có: (cx, cy) + feature', fontsize=9, ha='center', fontweight='bold')
    ax.annotate('', xy=(2, 3), xytext=(4.5, 1.8), arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.annotate('', xy=(8, 3), xytext=(5.5, 1.8), arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.text(5, 0.5, 'RoPE2D: local interaction  |  Prototype: class alignment', fontsize=8, ha='center')


def plot_ema_simulation(ax, prototypes, feats, labels, class_centers):
    """PCA 2D: query features + prototypes trong feature space."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        ax.text(0.5, 0.5, 'Need sklearn for PCA', ha='center', va='center', transform=ax.transAxes)
        return

    n_classes = prototypes.shape[0]
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    # PCA
    all_pts = np.vstack([prototypes, feats])
    pca = PCA(n_components=2)
    pts_2d = pca.fit_transform(all_pts)

    proto_2d = pts_2d[:n_classes]
    feat_2d = pts_2d[n_classes:]

    # Plot query features
    for c in range(n_classes):
        mask = labels == c
        ax.scatter(feat_2d[mask, 0], feat_2d[mask, 1], c=[colors[c]], s=20, alpha=0.6, label=f'Q class {c}')

    # Plot prototypes (lớn, viền đậm)
    for c in range(n_classes):
        ax.scatter(proto_2d[c, 0], proto_2d[c, 1], c=[colors[c]], s=200, marker='*',
                   edgecolors='black', linewidths=2, label=f'Prototype {c}', zorder=5)

    # Arrows: feature → prototype (kéo về)
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            for i in np.where(mask)[0][:3]:  # vẽ 3 mũi tên mỗi class
                ax.annotate('', xy=proto_2d[c], xytext=feat_2d[i],
                            arrowprops=dict(arrowstyle='->', color=colors[c], alpha=0.4, lw=1))

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Feature space: Prototype kéo query về')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def plot_predict_image(ax, image_path, detections, class_names):
    """Vẽ ảnh với boxes + labels từ predict (giống inference.py --save)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        ax.text(0.5, 0.5, 'Could not read image', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return
    img = img_bgr.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, (xyxy, conf, cid) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        color = colors[i % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        lbl = f"{_get_class_name(int(cid), class_names)} {float(conf):.2f}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, lbl, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title('Predict: boxes + labels')
    ax.axis('off')


def plot_dual_view(ax_spatial, ax_feat, ref_boxes, feats, labels, class_names=None):
    """
    Dual view: cùng màu = cùng class. RoPE2D (trái) vs Feature space (phải).
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        ax_spatial.text(0.5, 0.5, 'Need sklearn', ha='center', va='center', transform=ax_spatial.transAxes)
        return

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_classes, 1)))
    label_to_idx = {c: i for i, c in enumerate(unique_labels)}

    cx, cy = ref_boxes[:, 0], ref_boxes[:, 1]

    # Spatial — chấm to hơn, cùng màu theo class
    for i, c in enumerate(unique_labels):
        mask = labels == c
        if mask.any():
            ax_spatial.scatter(cx[mask], cy[mask], c=[colors[i]], s=60, alpha=0.85,
                              label=_get_class_name(int(c), class_names), edgecolors='white')

    ax_spatial.set_xlabel('cx (RoPE2D)')
    ax_spatial.set_ylabel('cy (RoPE2D)')
    ax_spatial.set_title('(1) RoPE2D: vị trí 2D — queries gần nhau attend mạnh')
    ax_spatial.set_xlim(0, 1)
    ax_spatial.set_ylim(0, 1)
    ax_spatial.set_aspect('equal')
    ax_spatial.invert_yaxis()
    ax_spatial.legend()

    # Feature space — PCA + pseudo-prototype (mean mỗi class)
    pts_2d = PCA(n_components=2).fit_transform(feats)

    for i, c in enumerate(unique_labels):
        mask = labels == c
        if mask.any():
            lbl = _get_class_name(int(c), class_names)
            ax_feat.scatter(pts_2d[mask, 0], pts_2d[mask, 1], c=[colors[i]], s=60, label=lbl, alpha=0.85,
                            edgecolors='white')
            # Pseudo-prototype = mean của class (mũi tên kéo về)
            mean_pt = pts_2d[mask].mean(0)
            ax_feat.scatter(mean_pt[0], mean_pt[1], marker='*', s=300, c=[colors[i]], edgecolors='black',
                            linewidths=1.5, zorder=5)

    ax_feat.set_xlabel('PCA 1')
    ax_feat.set_ylabel('PCA 2')
    ax_feat.set_title('(2) Feature space — cùng class gần nhau, kéo về prototype (★)')
    ax_feat.legend()
    ax_feat.set_aspect('equal')
    ax_feat.grid(True, alpha=0.3)


def run_real_model_forward(weights_path, image_path, model_size='base', threshold=0.35):
    """Dùng predict() như inference.py để lấy labels đúng, rồi forward để lấy ref_boxes + pred_queries."""
    import torch
    import torchvision.transforms.functional as F_tv

    from rfdetrv2 import SwiftDetrBase, SwiftDetrSmall, SwiftDetrTiny
    from rfdetrv2.util.coco_classes import COCO_CLASSES
    from rfdetrv2.util.swiftnet_pretrained import (
        SWIFTNET_WEIGHTS_BY_SIZE,
        resolve_pretrained_encoder_path,
    )

    pretrained = resolve_pretrained_encoder_path(
        project_root,
        model_size,
        explicit=None,
        weights_by_size=SWIFTNET_WEIGHTS_BY_SIZE,
    )

    model_cls = {
        "tiny": SwiftDetrTiny,
        "small": SwiftDetrSmall,
        "base": SwiftDetrBase,
    }[model_size]
    model = model_cls(pretrain_weights=weights_path, pretrained_encoder=pretrained, device='cuda')

    # 1. Predict như inference.py → labels đúng (dog, cat)
    detections = model.predict(image_path, threshold=threshold)
    class_names = model.class_names or COCO_CLASSES
    if hasattr(class_names, 'items'):
        pass
    elif isinstance(class_names, (list, tuple)):
        class_names = {i + 1: n for i, n in enumerate(class_names)}
    else:
        class_names = COCO_CLASSES

    # Lấy class_ids từ predict output (chính xác)
    if len(detections.xyxy) > 0:
        pred_class_ids = set(int(cid) for cid in detections.class_id)
    else:
        pred_class_ids = set()

    # Preprocess cho forward
    means = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    resolution = getattr(model.model, 'resolution', 640)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = F_tv.resize(img_tensor.unsqueeze(0), (resolution, resolution)).cuda()
    img_tensor = (img_tensor - means) / stds

    ref_boxes_capture = [None]
    pred_queries_capture = [None]

    def _hook_ref_boxes(module, args, kwargs):
        rb = kwargs.get('ref_boxes')
        if rb is not None:
            ref_boxes_capture[0] = rb.detach().cpu()

    lwdetr = model.model.model
    lwdetr.eval()

    transformer = lwdetr.transformer
    handle_ref = transformer.decoder.layers[-1].register_forward_pre_hook(_hook_ref_boxes, with_kwargs=True)

    orig_transformer_fwd = transformer.forward
    def _capture_hs_forward(*args, **kwargs):
        out_tuple = orig_transformer_fwd(*args, **kwargs)
        hs = out_tuple[0]
        if hs is not None:
            pred_queries_capture[0] = hs[-1].detach().cpu()
        return out_tuple
    transformer.forward = _capture_hs_forward

    try:
        with torch.no_grad():
            out = lwdetr([img_tensor.squeeze(0)])
    finally:
        transformer.forward = orig_transformer_fwd
        handle_ref.remove()

    ref_boxes = ref_boxes_capture[0]
    pred_queries = pred_queries_capture[0]

    if ref_boxes is None:
        ref_boxes = torch.zeros(1, 300, 4)
    if pred_queries is None:
        pred_queries = torch.randn(1, 300, 256) * 0.1

    pred_logits = out['pred_logits'][0].softmax(-1)
    scores_np, labels_pred = pred_logits[..., :-1].max(-1)
    scores_np = scores_np.cpu().numpy()
    labels_np = labels_pred.cpu().numpy()

    # Dùng class_ids từ predict() thay vì top2 từ raw logits
    if pred_class_ids:
        top_classes = pred_class_ids
    else:
        unique_labels = np.unique(labels_np)
        class_scores = [(c, scores_np[labels_np == c].sum()) for c in unique_labels]
        class_scores.sort(key=lambda x: -x[1])
        top_classes = {c for c, _ in class_scores[:2]}

    mask = np.isin(labels_np, list(top_classes)) & (scores_np > 0.1)
    idx = np.where(mask)[0]
    if len(idx) < 5:
        idx = np.argsort(scores_np)[::-1][:50]

    ref_np = ref_boxes[0].numpy()
    feat_np = pred_queries[0].cpu().numpy()

    return ref_np[idx], feat_np[idx], labels_np[idx], class_names, detections


def main():
    parser = argparse.ArgumentParser(description="Mô phỏng Prototype EMA + RoPE2D")
    parser.add_argument('--mode', choices=['simulate', 'real'], default='simulate')
    parser.add_argument('--save', default='output/prototype_rope2d.png')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--image', type=str)
    parser.add_argument('--model-size', default='base', choices=['tiny', 'small', 'base'])
    args = parser.parse_args()

    if args.mode == 'real' and (not args.weights or not args.image):
        print("Mode 'real' cần --weights và --image")
        return

    fig = plt.figure(figsize=(18, 6))

    if args.mode == 'simulate':
        # Schematic
        ax0 = fig.add_subplot(131)
        plot_schematic(ax0)

        # EMA simulation
        prototypes, feats, labels, _, history = simulate_prototype_ema(
            num_classes=3, feat_dim=64, num_queries_per_class=25, num_steps=100, momentum=0.999
        )
        ax1 = fig.add_subplot(132)
        plot_ema_simulation(ax1, prototypes, feats, labels, None)

        # EMA convergence
        ax2 = fig.add_subplot(133)
        try:
            from sklearn.decomposition import PCA
            n_cls = prototypes.shape[0]
            colors = plt.cm.Set1(np.linspace(0, 1, n_cls))
            for c in range(n_cls):
                for step in [0, 33, 66, 99]:
                    if step < len(history):
                        proto_c = history[step][c]
                        pts = np.vstack([[proto_c], feats[labels == c]])
                        pca = PCA(n_components=2)
                        pts_2d = pca.fit_transform(pts)
                        ax2.scatter(pts_2d[0, 0], pts_2d[0, 1], s=60 + step // 33 * 20, marker='*',
                                    c=[colors[c]], alpha=0.5 + 0.2 * (step == 99), edgecolors='black')
            ax2.set_title('Prototype convergence (EMA steps 0→99)')
            ax2.set_xlabel('PCA 1')
            ax2.set_ylabel('PCA 2')
            ax2.grid(True, alpha=0.3)
        except ImportError:
            ax2.text(0.5, 0.5, 'Need sklearn', ha='center', va='center', transform=ax2.transAxes)

    else:
        ref_boxes, feats, labels, class_names, detections = run_real_model_forward(
            args.weights, args.image, args.model_size
        )
        # 3 cột: ảnh predict | RoPE2D | Feature space
        ax_img = fig.add_subplot(131)
        plot_predict_image(ax_img, args.image, detections, class_names)
        ax0 = fig.add_subplot(132)
        ax1 = fig.add_subplot(133)
        plot_dual_view(ax0, ax1, ref_boxes, feats, labels, class_names)

    plt.suptitle('Prototype EMA Memory & RoPE2D — Cùng màu = cùng class', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_p = Path(args.save)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_p), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_p}")


if __name__ == '__main__':
    main()
