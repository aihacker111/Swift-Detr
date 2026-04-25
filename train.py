#!/usr/bin/env python3
"""Train Swift-DETR with PyTorch Lightning (programmatic entry point).

Equivalent in spirit to::

    swiftdetr fit --config configs/swiftdetr_small.yaml

but uses the public Python API so you can script experiments without the
jsonargparse CLI. Install the package in editable mode or set ``PYTHONPATH``.

Examples::

    # COCO 2017 layout: DATA_DIR/train2017, DATA_DIR/val2017, DATA_DIR/annotations/*.json
    python train.py --dataset /data/coco --output ./output/my_run

    python train.py --variant base --dataset /data/coco --output ./out \\
        --epochs 12 --batch-size 2 --grad-accum 8 --num-workers 4

    # Custom input size (multiple of 32), e.g. 512 or 768
    python train.py --dataset /data/coco --output ./out --resolution 512

    # Resume a Lightning checkpoint
    python train.py --dataset /data/coco --output ./out --resume ./out/last.ckpt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Type

# Repo root: allow ``python train.py`` without ``pip install -e .``
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
from swiftdetr.training.module_data import SwiftDetrDataModule  # noqa: E402
from swiftdetr.training.module_model import SwiftDetrModule  # noqa: E402
from swiftdetr.training.trainer import build_trainer  # noqa: E402

_VARIANT: dict[str, Type[ModelConfig]] = {
    "tiny": SwiftDetrTinyConfig,
    "small": SwiftDetrSmallConfig,
    "base": SwiftDetrBaseConfig,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset root. For COCO: folder containing train2017/, val2017/, annotations/.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="./output/swiftdetr_train",
        help="Checkpoints, logs, and metrics (default: ./output/swiftdetr_train).",
    )
    p.add_argument(
        "--variant",
        type=str,
        choices=sorted(_VARIANT),
        default="small",
        help="Backbone/decoder preset. Default train resolutions: tiny=512, small/base=640 (override with --resolution).",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "Square training input size (short side / square side depending on dataloader). "
            "Must be divisible by 32 (SwiftNet stride). "
            "Default: variant preset (e.g. 640 for small). "
            "Also sets positional_encoding_size to resolution//16."
        ),
    )
    p.add_argument(
        "--dataset-file",
        type=str,
        choices=("coco", "roboflow", "yolo"),
        default="coco",
        help="Loader format. Use ``coco`` for official COCO 2017 layout.",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=80,
        help="Number of object classes (COCO: 80; background is added internally).",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4, dest="grad_accum")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--pretrain-weights",
        type=str,
        default=None,
        help="Optional full Swift-DETR / Lightning .ckpt or legacy .pth to warm-start.",
    )
    p.add_argument(
        "--encoder-imagenet-weights",
        type=str,
        default=None,
        help="Optional SWIFTNet ImageNet trunk checkpoint (loaded into the backbone).",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from this PyTorch Lightning checkpoint path.",
    )
    p.add_argument(
        "--fast-dev-run",
        type=int,
        default=None,
        metavar="N",
        help="If set, run N train/val batches only (sanity check).",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed (optional).")
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
    if args.resolution is not None:
        if args.resolution <= 0 or args.resolution % 32 != 0:
            sys.exit(
                f"--resolution must be positive and divisible by 32 (SwiftNet), got {args.resolution}."
            )
        _mc["resolution"] = args.resolution
        _mc["positional_encoding_size"] = args.resolution // 16
    model_config = _VARIANT[args.variant](**_mc)

    train_config = TrainConfig(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        dataset_file=args.dataset_file,  # type: ignore[arg-type]
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    if args.seed is not None:
        from pytorch_lightning import seed_everything

        seed_everything(args.seed, workers=True)

    model = SwiftDetrModule(model_config, train_config)
    datamodule = SwiftDetrDataModule(model_config, train_config)

    train_kw: dict = {}
    if args.fast_dev_run is not None:
        train_kw["fast_dev_run"] = args.fast_dev_run

    trainer = build_trainer(train_config, model_config, **train_kw)
    resume = args.resume
    if resume:
        resume = os.path.realpath(os.path.expanduser(resume))
    trainer.fit(model, datamodule, ckpt_path=resume)


if __name__ == "__main__":
    main()
