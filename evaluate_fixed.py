"""
evaluate_fixed.py — Ground-truth evaluation harness.
=====================================================
ĐÂY LÀ FILE CỐ ĐỊNH. AGENT KHÔNG ĐƯỢC SỬA FILE NÀY.
In metrics theo định dạng chuẩn để agent grep được.
"""

import argparse
import os
import sys
import time
import torch
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def parse_args():
    p = argparse.ArgumentParser("RF-DETRv2 Fixed Evaluator")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-size", default="base",
                   choices=["tiny", "small", "base"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def get_peak_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def main():
    args = parse_args()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    try:
        from rfdetrv2 import SwiftDetrBase, SwiftDetrSmall, SwiftDetrTiny
        from rfdetrv2.util.swiftnet_pretrained import (
            SWIFTNET_WEIGHTS_BY_SIZE,
            resolve_pretrained_encoder_path,
        )

        pretrained_encoder = resolve_pretrained_encoder_path(
            project_root, args.model_size,
            explicit=os.environ.get("PRETRAINED_ENCODER"),
            weights_by_size=SWIFTNET_WEIGHTS_BY_SIZE,
        )

        model_cls = {
            "tiny": SwiftDetrTiny,
            "small": SwiftDetrSmall,
            "base": SwiftDetrBase,
        }[args.model_size]

        model = model_cls(pretrained_encoder=pretrained_encoder)

        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            # Fallback: last checkpoint
            ckpt_path = Path(args.checkpoint).parent / "checkpoint.pth"

        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("ema", ckpt.get("model", ckpt))
            model.model.load_state_dict(state, strict=False)
            print(f"[eval] Loaded checkpoint: {ckpt_path}")
        else:
            print(f"[eval] WARNING: checkpoint not found at {ckpt_path}")

        # Run COCO-style evaluation
        stats = model.val(
            dataset_dir=args.dataset_dir,
            dataset_file="coco",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # stats: dict or list from pycocotools
        # Standard COCO stats order: [mAP, mAP50, mAP75, mAP_s, mAP_m, mAP_l, ...]
        if isinstance(stats, (list, tuple)):
            val_map   = float(stats[0]) if len(stats) > 0 else 0.0
            val_map50 = float(stats[1]) if len(stats) > 1 else 0.0
            val_map75 = float(stats[2]) if len(stats) > 2 else 0.0
        elif isinstance(stats, dict):
            val_map   = float(stats.get("AP",    stats.get("mAP",   0.0)))
            val_map50 = float(stats.get("AP50",  stats.get("mAP50", 0.0)))
            val_map75 = float(stats.get("AP75",  stats.get("mAP75", 0.0)))
        else:
            val_map = val_map50 = val_map75 = 0.0

        elapsed = time.time() - t0
        peak_vram = get_peak_vram_mb()

        print("---")
        print(f"val_mAP:          {val_map:.6f}")
        print(f"val_mAP50:        {val_map50:.6f}")
        print(f"val_mAP75:        {val_map75:.6f}")
        print(f"eval_seconds:     {elapsed:.1f}")
        print(f"peak_vram_mb:     {peak_vram:.1f}")

    except Exception:
        traceback.print_exc()
        print("---")
        print("val_mAP:          0.000000")
        print("val_mAP50:        0.000000")
        print("val_mAP75:        0.000000")
        print(f"peak_vram_mb:     {get_peak_vram_mb():.1f}")
        sys.exit(1)


if __name__ == "__main__":
    main()