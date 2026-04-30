#!/usr/bin/env python3
"""
Simple video inference with SwiftDetr.
Reads video, runs detection on each frame, writes annotated output.
"""
from pathlib import Path
import argparse
import sys

import cv2
import supervision as sv

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfdetrv2 import SwiftDetrBase, SwiftDetrSmall, SwiftDetrTiny
from rfdetrv2.util.coco_classes import COCO_CLASSES
from rfdetrv2.util.swiftnet_pretrained import (
    SWIFTNET_WEIGHTS_BY_SIZE,
    resolve_pretrained_encoder_path,
)

VIDEO_WEIGHTS_BY_SIZE = {k: SWIFTNET_WEIGHTS_BY_SIZE[k] for k in ("tiny", "small", "base")}


def _get_class_name(cls_id: int, class_names: dict, use_coco_fallback: bool = True) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SwiftDetr inference on video.")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default=None, help="Output video path (default: input_video_det.mp4)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--model-size", choices=["tiny", "small", "base"], default="base")
    parser.add_argument("--show", action="store_true", help="Display output in window (press 'q' to quit)")
    parser.add_argument("--skip-frames", type=int, default=0, help="Process every Nth frame (0 = all)")
    args = parser.parse_args()

    # Load model
    model_table = {"tiny": SwiftDetrTiny, "small": SwiftDetrSmall, "base": SwiftDetrBase}
    ModelClass = model_table[args.model_size]
    pretrained = resolve_pretrained_encoder_path(
        project_root,
        args.model_size,
        explicit=None,
        weights_by_size=VIDEO_WEIGHTS_BY_SIZE,
    )
    model = ModelClass(pretrain_weights=args.weights, pretrained_encoder=pretrained, device=args.device)
    class_names = model.class_names or COCO_CLASSES

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = args.output or str(Path(args.video).parent / (Path(args.video).stem + "_det.mp4"))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    frame_idx = 0
    skip = max(1, args.skip_frames) if args.skip_frames else 1

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % skip != 0:
                writer.write(frame_bgr)
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = model.predict(frame_rgb, threshold=args.threshold)

            labels = [
                f"{_get_class_name(int(cid), class_names)} {float(conf):.2f}"
                for conf, cid in zip(detections.confidence, detections.class_id)
            ]

            annotated = box_annotator.annotate(scene=frame_bgr.copy(), detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

            writer.write(annotated)

            if args.show:
                cv2.imshow("RF-DETR", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Saved output to {out_path}")


if __name__ == "__main__":
    main()
