"""
Evaluate SwiftDetr on COCO val2017.

Usage
-----
python scripts/evaluate.py --weights checkpoints/model.pth --dataset-dir /path/to/COCO2017

Expects COCO2017 layout:
  dataset_dir/val2017/              (or images/val2017/)
  dataset_dir/annotations/instances_val2017.json

Threshold
---------
Repo's internal eval (--eval in main) uses postprocess(outputs) with NO score threshold —
it keeps top-300 predictions and passes them to CocoEvaluator. Our evaluate.py uses
model.predict() which applies a threshold. Default 0.001 matches submit_coco2017 and
COCO benchmark convention: keep low-score boxes so COCOeval can compute AP correctly.
"""
from pathlib import Path
import argparse
import os
import sys
import time

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rfdetrv2 import SwiftDetrBase, SwiftDetrSmall, SwiftDetrTiny
from rfdetrv2.datasets.coco_eval import patched_pycocotools_summarize
from rfdetrv2.util.swiftnet_pretrained import (
    SWIFTNET_WEIGHTS_BY_SIZE,
    resolve_pretrained_encoder_path,
)
DEFAULT_PRETRAINED = os.environ.get("PRETRAINED_ENCODER")


def _resolve_category_id(raw_cls_id: int) -> int:
    """Map model class id to COCO 1-indexed category_id. COCO uses 1–90 (sparse)."""
    return int(raw_cls_id)


def _find_val_images(dataset_dir: Path) -> tuple[Path, list[Path]]:
    """Find val2017 images and annotations. Returns (ann_file, image_paths)."""
    ann_file = dataset_dir / "annotations" / "instances_val2017.json"
    if not ann_file.exists():
        raise FileNotFoundError(
            f"COCO val annotations not found: {ann_file}\n"
            "Expected layout: dataset_dir/annotations/instances_val2017.json"
        )

    for subdir in ["val2017", "images/val2017"]:
        img_dir = dataset_dir / subdir
        if img_dir.exists():
            paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
            if paths:
                return ann_file, paths

    raise FileNotFoundError(
        f"No val2017 images found in {dataset_dir}\n"
        "Tried: val2017/, images/val2017/"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RF-DETRv2 on COCO val2017")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.environ.get("DATASET_DIR", "/lustre/scratch/client/scratch/dms/dms_group/COCO2017"),
        help="COCO root (val2017/ and annotations/ inside)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["tiny", "small", "base"],
        default="base",
    )
    parser.add_argument("--pretrained-encoder", type=str, default=DEFAULT_PRETRAINED)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="Score threshold for model.predict. Default 0.001 matches COCO convention: "
        "keep low-score boxes so COCOeval can compute AP at all recall levels. "
        "Repo's internal --eval uses postprocess with no threshold; we use predict so need low value.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument(
        "--max-dets",
        type=int,
        default=500,
        help="Max detections per image. Uses patched summarize so AP@[.5:.95] respects this value.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    ann_file, image_paths = _find_val_images(dataset_dir)

    # Load COCO GT for image_id mapping
    coco_gt = COCO(str(ann_file))
    filename_to_id = {Path(p["file_name"]).name: p["id"] for p in coco_gt.dataset["images"]}
    image_paths = [p for p in image_paths if p.name in filename_to_id]
    if not image_paths:
        raise ValueError("No val images found in annotations. Check dataset layout.")
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    print(f"Evaluating on {len(image_paths)} images")

    # Resolve pretrained encoder
    explicit = args.pretrained_encoder or DEFAULT_PRETRAINED
    pretrained = resolve_pretrained_encoder_path(
        project_root,
        args.model_size,
        explicit=explicit if explicit else None,
        weights_by_size=SWIFTNET_WEIGHTS_BY_SIZE,
    )

    # Load model
    model_cls = {
        "tiny": SwiftDetrTiny,
        "small": SwiftDetrSmall,
        "base": SwiftDetrBase,
    }[args.model_size]
    print(f"Loading {args.model_size} from {args.weights} ...")
    model = model_cls(
        pretrain_weights=args.weights,
        pretrained_encoder=pretrained,
        device=args.device,
    )

    # Run inference
    coco_results = []
    total = len(image_paths)
    t0 = time.time()

    for idx, img_path in enumerate(image_paths):
        image_id = filename_to_id.get(img_path.name)
        if image_id is None:
            continue

        detections = model.predict(str(img_path), threshold=args.threshold)

        for xyxy, conf, cls_id in zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id,
        ):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            category_id = _resolve_category_id(int(cls_id))
            coco_results.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(conf),
            })

        if (idx + 1) % 500 == 0 or idx + 1 == total:
            elapsed = time.time() - t0
            print(f"[{idx + 1}/{total}] {elapsed:.1f}s  {len(coco_results)} dets")

    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s ({total / elapsed:.1f} img/s)")

    # COCO evaluation
    if len(coco_results) == 0:
        print("No detections — cannot evaluate.")
        return

    coco_dt = coco_gt.loadRes(coco_results)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.maxDets = [1, 10, args.max_dets]
    evaluator.evaluate()
    evaluator.accumulate()
    patched_pycocotools_summarize(evaluator)

    stats = evaluator.stats
    print("\n" + "=" * 60)
    print(f"AP@[.5:.95] = {stats[0]:.3f}")
    print(f"AP@.50       = {stats[1]:.3f}")
    print(f"AP@.75       = {stats[2]:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
