from pathlib import Path
import argparse
import os
import sys

import cv2
import numpy as np
import supervision as sv

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfdetrv2 import SwiftDetrBase, SwiftDetrSmall, SwiftDetrTiny
from rfdetrv2.util.swiftnet_pretrained import (
    SWIFTNET_WEIGHTS_BY_SIZE,
    resolve_pretrained_encoder_path,
)
DEFAULT_PRETRAINED = os.environ.get("PRETRAINED_ENCODER")
from rfdetrv2.util.coco_classes import COCO_CLASSES


def _get_class_name(cls_id: int, class_names: dict, use_coco_fallback: bool = True) -> str:
    """Resolve a human-readable class name from a raw model class id.

    model.class_names can be in three formats:
      A) {0: "person", 1: "bicycle", ...}  — 0-indexed string values  → use directly
      B) {1: "person", 2: "bicycle", ...}  — 1-indexed string values  → use directly
      C) {0: 1, 1: 2, 17: 18, 18: 19, ...} — int remapping dict       → IGNORE,
         fall through to COCO_CLASSES with the raw id instead

    For format C (RF-DETR default), the raw id IS already the correct COCO 1-indexed key:
      raw=18 → COCO_CLASSES[18] = "dog"  ✓
    """
    raw = int(cls_id)

    # Only trust class_names values when they are actual name strings
    for key in (raw, raw + 1):
        val = class_names.get(key)
        if isinstance(val, str):
            return val

    # class_names has int values (remapping dict) or no match — use COCO directly
    if use_coco_fallback:
        if raw in COCO_CLASSES:
            return COCO_CLASSES[raw]        # e.g. 18 → "dog"
        if (raw + 1) in COCO_CLASSES:
            return COCO_CLASSES[raw + 1]

    return str(raw)


def _merge_overlapping_detections(detections: sv.Detections, labels: list, box_eps: float = 2.0):
    """Merge detections with identical boxes so both labels are visible (stacked)."""
    if len(detections) == 0:
        return detections, labels
    xyxy = np.asarray(detections.xyxy)
    conf = np.asarray(detections.confidence)
    cls_id = np.asarray(detections.class_id)
    merged_xyxy, merged_conf, merged_cls, merged_labels = [], [], [], []
    used = [False] * len(detections)
    for i in range(len(detections)):
        if used[i]:
            continue
        box = xyxy[i]
        group = [i]
        for j in range(i + 1, len(detections)):
            if used[j]:
                continue
            if np.all(np.abs(xyxy[j] - box) < box_eps):
                group.append(j)
                used[j] = True
        used[i] = True
        merged_xyxy.append(box)
        merged_conf.append(conf[group[0]])
        merged_cls.append(cls_id[group[0]])
        merged_labels.append("\n".join(labels[k] for k in group))
    return (
        sv.Detections(
            xyxy=np.array(merged_xyxy),
            confidence=np.array(merged_conf),
            class_id=np.array(merged_cls),
        ),
        merged_labels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SwiftDetr inference on one image.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained checkpoint (.pth).")
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["tiny", "small", "base"],
        default="base",
        help="SwiftDetr tier: tiny | small | base.",
    )
    parser.add_argument(
        "--pretrained-encoder",
        type=str,
        default=DEFAULT_PRETRAINED,
        help="Path to SwiftNet encoder .pth. If unset, looks under swiftnet_pretrained/ or project root (see SWIFTNET_WEIGHTS_BY_SIZE).",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--save", type=str, default=None, help="Optional output image path with drawn boxes.")
    parser.add_argument(
        "--classes-file",
        type=str,
        default=None,
        help="Optional text file with one class name per line (overrides checkpoint).",
    )
    args = parser.parse_args()

    explicit = args.pretrained_encoder or DEFAULT_PRETRAINED
    pretrained = resolve_pretrained_encoder_path(
        project_root,
        args.model_size,
        explicit=explicit if explicit else None,
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
    detections = model.predict(args.image, threshold=args.threshold)
    print(model.class_names)
    if args.classes_file:
        with open(args.classes_file) as f:
            names = [line.strip() for line in f if line.strip()]
        class_names = {i + 1: n for i, n in enumerate(names)}
    else:
        class_names = model.class_names or COCO_CLASSES

    # Debug: show class_names format so mapping issues are visible
    if class_names and class_names is not COCO_CLASSES:
        sample = dict(list(class_names.items())[:3])
        val_types = {type(v).__name__ for v in list(class_names.values())[:5]}
        print(f"[debug] model.class_names sample={sample} value_types={val_types}")

    print(f"Detections: {len(detections)}")
    for i, (xyxy, conf, cls_id) in enumerate(
        zip(detections.xyxy, detections.confidence, detections.class_id),
        start=1,
    ):
        cls_int = int(cls_id)
        cls_name = _get_class_name(cls_int, class_names)
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        print(
            f"{i:03d} | class={cls_name} (id={cls_int}) | conf={float(conf):.4f} "
            f"| box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )

    if args.save is not None:
        image_bgr = cv2.imread(args.image)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read input image: {args.image}")

        labels = [
            f"{_get_class_name(int(cid), class_names)} {float(conf):.2f}"
            for conf, cid in zip(detections.confidence, detections.class_id)
        ]
        draw_detections, draw_labels = _merge_overlapping_detections(detections, labels)

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(
            text_scale=0.35,
            text_thickness=1,
            text_padding=2,
        )
        annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=draw_detections)
        annotated = label_annotator.annotate(scene=annotated, detections=draw_detections, labels=draw_labels)

        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved annotated image to: {output_path}")


if __name__ == "__main__":
    main()