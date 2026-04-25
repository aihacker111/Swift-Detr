#!/usr/bin/env python3
"""Train Swift-DETR with pure PyTorch (no Lightning).

Examples::

    # COCO 2017 layout: DATA_DIR/train2017, DATA_DIR/val2017, DATA_DIR/annotations/*.json
    python train.py --dataset /data/coco --output ./output/my_run

    python train.py --variant base --dataset /data/coco --output ./out \\
        --epochs 12 --batch-size 2 --grad-accum 8 --num-workers 4

    # Custom input size (multiple of 32), e.g. 512 or 768
    python train.py --dataset /data/coco --output ./out --resolution 512

    # Resume a checkpoint
    python train.py --dataset /data/coco --output ./out --resume ./out/last.pth

    # Augmentation backend
    python train.py --dataset /data/coco --output ./out --aug-preset aggressive --augmentation-backend auto
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Type

# Allow ``python train.py`` without ``pip install -e .``
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from swiftdetr.config import (  # noqa: E402
    ModelConfig,
    SwiftDetrBaseConfig,
    SwiftDetrSmallConfig,
    SwiftDetrTinyConfig,
    TrainConfig,
)
from swiftdetr.datasets.aug_config import (  # noqa: E402
    AUG_AERIAL,
    AUG_AGGRESSIVE,
    AUG_CONSERVATIVE,
    AUG_INDUSTRIAL,
)
from swiftdetr.training.module_data import SwiftDetrData  # noqa: E402
from swiftdetr.training.module_model import SwiftDetrWrapper  # noqa: E402
from swiftdetr.training.trainer import fit  # noqa: E402

_VARIANT: dict[str, Type[ModelConfig]] = {
    "tiny": SwiftDetrTinyConfig,
    "small": SwiftDetrSmallConfig,
    "base": SwiftDetrBaseConfig,
}

_AUG_PRESET = {
    "default": None,
    "conservative": AUG_CONSERVATIVE,
    "aggressive": AUG_AGGRESSIVE,
    "aerial": AUG_AERIAL,
    "industrial": AUG_INDUSTRIAL,
    "none": {},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=str, required=True,
                   help="Dataset root (COCO: folder with train2017/, val2017/, annotations/).")
    p.add_argument("--output", type=str, default="./output/swiftdetr_train",
                   help="Checkpoints, logs, and metrics (default: ./output/swiftdetr_train).")
    p.add_argument("--variant", type=str, choices=sorted(_VARIANT), default="small",
                   help="Backbone/decoder preset.")
    p.add_argument("--resolution", type=int, default=None, metavar="PX",
                   help="Square training input size (must be divisible by 32).")
    p.add_argument("--dataset-file", type=str, choices=("coco", "roboflow", "yolo"), default="coco")
    p.add_argument("--num-classes", type=int, default=90,
                   help="Number of object classes (COCO: 80; background added internally).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4, dest="grad_accum")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pretrain-weights", type=str, default=None,
                   help="Optional Swift-DETR .pth to warm-start.")
    p.add_argument("--encoder-imagenet-weights", type=str, default=None,
                   help="Optional SwiftNet ImageNet trunk checkpoint.")
    p.add_argument("--augmentation-backend", type=str, choices=("cpu", "auto", "gpu"), default="cpu")
    p.add_argument("--aug-preset", type=str, choices=tuple(_AUG_PRESET), default="default", metavar="NAME")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume training from this checkpoint path (.pth or .ckpt).")
    p.add_argument("--fast-dev-run", type=int, default=None, metavar="N",
                   help="Run N train/val batches only (sanity check).")
    p.add_argument("--debug-limit-data", type=int, default=None, metavar="N",
                   help="Limit train and val to N batches per epoch (for quick code testing).")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = os.path.realpath(os.path.expanduser(args.dataset))
    output_dir = os.path.realpath(os.path.expanduser(args.output))
    os.makedirs(output_dir, exist_ok=True)

    _mc: dict = {"num_classes": args.num_classes}
    if args.encoder_imagenet_weights:
        _mc["encoder_imagenet_weights"] = os.path.realpath(os.path.expanduser(args.encoder_imagenet_weights))
    if args.pretrain_weights:
        _mc["pretrain_weights"] = os.path.realpath(os.path.expanduser(args.pretrain_weights))
        _mc["load_detection_pretrain"] = True
    if args.resolution is not None:
        if args.resolution <= 0 or args.resolution % 32 != 0:
            sys.exit(f"--resolution must be positive and divisible by 32, got {args.resolution}.")
        _mc["resolution"] = args.resolution
        _mc["positional_encoding_size"] = args.resolution // 16
    model_config = _VARIANT[args.variant](**_mc)

    train_kwargs: dict = {
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "dataset_file": args.dataset_file,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "augmentation_backend": args.augmentation_backend,
    }
    if args.aug_preset != "default":
        train_kwargs["aug_config"] = _AUG_PRESET[args.aug_preset]
    if args.debug_limit_data is not None:
        train_kwargs["debug_limit_data"] = args.debug_limit_data
    train_config = TrainConfig(**train_kwargs)

    wrapper = SwiftDetrWrapper(model_config, train_config)
    data = SwiftDetrData(model_config, train_config)

    resume = args.resume
    if resume:
        resume = os.path.realpath(os.path.expanduser(resume))

    fit(
        wrapper=wrapper,
        data=data,
        output_dir=output_dir,
        resume=resume,
        fast_dev_run=args.fast_dev_run,
    )


if __name__ == "__main__":
    main()
