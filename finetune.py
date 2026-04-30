"""
RF-DETRv2 Fine-tuning — autoresearch target file.
=================================================
ĐÂY LÀ FILE DUY NHẤT AGENT ĐƯỢC PHÉP SỬA.
Mọi hyperparameter đều là fair game.

Metric: val_mAP (higher = better).
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from rfdetrv2 import SwiftDetrBase, SwiftDetrSmall, SwiftDetrTiny
from rfdetrv2.util.swiftnet_pretrained import (
    SWIFTNET_WEIGHTS_BY_SIZE,
    resolve_pretrained_encoder_path,
)
from rfdetrv2.util.rfdetr_pretrained import resolve_rfdetr_coco_checkpoint

# ─── PATHS (không sửa) ────────────────────────────────────────────────────────
DATASET_DIR   = os.environ.get("DATASET_DIR",   "/kaggle/input/datasets/duwipurnamasidik/visdrone-2019-coco-format")     # custom small dataset
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR",    "/kaggle/working/output/finetune")
# COCO RF-DETR checkpoint: optional path. If unset → auto-download from HuggingFace
# (https://huggingface.co/myn0908/rfdetrv2) into rfdetr_pretrained/ for this MODEL_SIZE.
COCO_WEIGHTS  = os.environ.get("COCO_WEIGHTS")  # e.g. /path/to/custom.pth or weights/mine.pth

PRETRAINED_ENCODER = os.environ.get("PRETRAINED_ENCODER")  # optional explicit SwiftNet .pth

# ─── HYPERPARAMETERS (agent edits below) ──────────────────────────────────────

MODEL_SIZE = "tiny"   # "tiny" | "small" | "base"

# Training
EPOCHS               = 10
BATCH_SIZE           = 4
GRAD_ACCUM_STEPS     = 4    # effective_batch = BATCH_SIZE * GRAD_ACCUM_STEPS * n_gpus
NUM_WORKERS          = 4
AMP                  = True   # mixed precision

# Learning rate
LR                   = 2e-4       # decoder LR
LR_ENCODER           = 2.5e-5    # encoder LR (ratio ~1:8 vs decoder)
LR_SCALE_MODE        = "sqrt"    # "sqrt" | "linear"
WARMUP_EPOCHS        = 1

# LR scheduler
LR_SCHEDULER         = "cosine_restart"
LR_RESTART_PERIOD    = 5         # epochs per cosine restart cycle
LR_RESTART_DECAY     = 0.8       # LR peak decay each cycle
LR_MIN_FACTOR        = 0.05      # min LR = LR * LR_MIN_FACTOR

# Loss coefficients
CLS_LOSS_COEF        = 1.0
BBOX_LOSS_COEF       = 5.0
GIOU_LOSS_COEF       = 2.0
USE_VARIFOCAL_LOSS   = False

# Architecture
USE_CONVNEXT_PROJECTOR = True
USE_WINDOWED_ATTN      = False
FREEZE_ENCODER         = False   # True = chỉ train decoder (tốt cho very small data)

# ─── MAIN (không sửa) ─────────────────────────────────────────────────────────

def main():
    pretrained_encoder = resolve_pretrained_encoder_path(
        project_root, MODEL_SIZE,
        explicit=PRETRAINED_ENCODER,
        weights_by_size=SWIFTNET_WEIGHTS_BY_SIZE,
    )

    model_cls = {
        "tiny": SwiftDetrTiny,
        "small": SwiftDetrSmall,
        "base": SwiftDetrBase,
    }[MODEL_SIZE]

    # Full RF-DETR checkpoint (COCO) — ``pretrain_weights`` uses ``Model`` loader (head resize, etc.).
    coco_ckpt = resolve_rfdetr_coco_checkpoint(
        project_root, MODEL_SIZE, explicit=COCO_WEIGHTS
    )
    model_kw = dict(
        pretrained_encoder=pretrained_encoder,
        use_windowed_attn=USE_WINDOWED_ATTN,
        use_rsa=False,
        use_convnext_projector=USE_CONVNEXT_PROJECTOR,
        freeze_encoder=FREEZE_ENCODER,
    )
    if coco_ckpt:
        model_kw["pretrain_weights"] = coco_ckpt
        print(f"[finetune] pretrain_weights (RF-DETR COCO): {coco_ckpt}")
    else:
        print(
            "[finetune] RFDETR_SKIP_COCO_CHECKPOINT set — training detection head from "
            "SwiftNet encoder only (no RF-DETR COCO checkpoint)."
        )

    model = model_cls(**model_kw)

    model.train(
        coco_path=DATASET_DIR,
        dataset_dir=DATASET_DIR,
        dataset_file="coco",           # dataset phải ở COCO format
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        use_ema=True,
        num_workers=NUM_WORKERS,
        run_test=False,
        device="cuda",
        output_dir=OUTPUT_DIR,
        amp=AMP,
        tensorboard=False,
        warmup_epochs=WARMUP_EPOCHS,
        lr=LR,
        lr_encoder=LR_ENCODER,
        lr_scale_mode=LR_SCALE_MODE,
        lr_scheduler=LR_SCHEDULER,
        lr_min_factor=LR_MIN_FACTOR,
        lr_restart_period=LR_RESTART_PERIOD,
        lr_restart_decay=LR_RESTART_DECAY,
        use_varifocal_loss=USE_VARIFOCAL_LOSS,
        cls_loss_coef=CLS_LOSS_COEF,
        bbox_loss_coef=BBOX_LOSS_COEF,
        giou_loss_coef=GIOU_LOSS_COEF,
        use_convnext_projector=USE_CONVNEXT_PROJECTOR,
        freeze_encoder=FREEZE_ENCODER,
    )

    # ── Evaluation (always runs after training) ───────────────────────────────
    import subprocess, re
    result = subprocess.run(
        ["python", "evaluate_fixed.py",
         "--dataset-dir", DATASET_DIR,
         "--checkpoint", str(Path(OUTPUT_DIR) / "checkpoint_best.pth"),
         "--model-size", MODEL_SIZE],
        capture_output=True, text=True
    )
    print(result.stdout)
    print(result.stderr, file=sys.stderr)

    # Re-print metrics in parseable format for grep
    for line in result.stdout.splitlines():
        if re.match(r"^(val_mAP|val_mAP50|val_mAP75|peak_vram_mb|training_epochs):", line):
            print(line)


if __name__ == "__main__":
    main()