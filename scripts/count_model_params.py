#!/usr/bin/env python3
"""
Đo tổng số tham số và kích thước (FP32 ước lượng / torch.save state_dict) cho 3 biến thể:
tiny, small, base (SwiftDetr / rfdetrv2.config).

Chạy từ thư mục gốc repo:
  python scripts/count_model_params.py
  python count_model_params.py   # wrapper tại root, tương đương

Không import rfdetrv2/__init__.py (tránh supervision, pycocotools).

Mặc định đếm model **không** SRA (``use_rsa`` trong config = False). Muốn đếm khi bật SRA:
``python scripts/count_model_params.py --use-rsa``

Đóng băng backbone (giống train): ``--freeze-encoder`` — tổng ``numel`` không đổi, ``trainable`` giảm.

**Vì sao đổi G / số head mà tổng param không đổi?**

1. Chưa bật SRA (thiếu ``--use-rsa``) → không tạo module SRA, đổi G/head không có gì để áp dụng.

2. Chỉ sửa dòng default trong ``populate_args`` cục bộ: giá trị thật đến từ ``ModelConfig`` qua
   ``kwargs = _cfg_kwargs(cfg)`` rồi ``defaults.update(extra_kwargs)`` — **config ghi đè** default.
   Cần sửa ``rfdetrv2/config.py`` hoặc dùng ``--sra-G`` / ``--sra-heads`` (ghi đè khi đếm).

3. SRA chỉ là phần nhỏ so với DINOv3+decoder: đổi G (routing) làm tăng chủ yếu ``W_r`` (``dim×G``);
   đổi ``n_heads`` phải chia hết ``dim``. Tổng model có thể chỉ tăng vài trăm KB–vài MB — cột "M"
   làm tròn có thể trông giống nhau dù ``numel()`` đã đổi.
Cần: PyTorch, pydantic, weight SwiftNet ``*.pth`` trong ``swiftnet_pretrained/`` hoặc gốc repo
(hoặc truyền ``--pretrained-encoder``).

``populate_args`` trong file này cần giữ khớp ``rfdetrv2/main.py`` khi thêm field mới.
"""

from __future__ import annotations

import argparse
import io
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RFDETR_PKG = PROJECT_ROOT / "rfdetrv2"

# --- Tránh chạy rfdetrv2/__init__.py (import detr → supervision) ----------------
if "rfdetrv2" not in sys.modules:
    _pkg = types.ModuleType("rfdetrv2")
    _pkg.__path__ = [str(RFDETR_PKG)]
    sys.modules["rfdetrv2"] = _pkg

if "supervision" not in sys.modules:
    sys.modules["supervision"] = types.ModuleType("supervision")

import torch

# --- populate_args: đồng bộ với rfdetrv2/main.py (chỉ dùng để build Namespace) ---
# fmt: off
def populate_args(**extra_kwargs):
    """Mirror of rfdetrv2.main.populate_args defaults + Namespace build."""
    if extra_kwargs.pop("no_use_convnext_projector", None) is True:
        extra_kwargs["use_convnext_projector"] = False
    defaults = dict(
        num_classes=2, grad_accum_steps=1, amp=False, lr=2e-4, lr_encoder=2.5e-5,
        lr_scale_mode="sqrt", batch_size=2, weight_decay=1e-4, epochs=12, lr_drop=11,
        clip_max_norm=0.1, lr_vit_layer_decay=0.8, lr_component_decay=1.0, do_benchmark=False,
        dropout=0, drop_path=0, drop_mode="standard", drop_schedule="constant", cutoff_epoch=0,
        pretrained_encoder=None, pretrain_weights=None, pretrain_exclude_keys=None,
        pretrain_keys_modify_to_load=None, pretrained_distiller=None,
        encoder="vit_tiny", vit_encoder_num_layers=12, window_block_indexes=None,
        position_embedding="sine", out_feature_indexes=[-1], freeze_encoder=False,
        layer_norm=False, rms_norm=False, backbone_lora=False, force_no_pretrain=False,
        dec_layers=3, dim_feedforward=2048, hidden_dim=256, sa_nheads=8, ca_nheads=8,
        num_queries=300, group_detr=13, two_stage=False, projector_scale=["P3", "P4", "P5"],
        lite_refpoint_refine=False, num_select=100, dec_n_points=8, decoder_norm="LN",
        bbox_reparam=False, freeze_batch_norm=False,
        set_cost_class=2, set_cost_bbox=5, set_cost_giou=2,
        cls_loss_coef=1.0, bbox_loss_coef=5.0, giou_loss_coef=2.0, focal_alpha=0.25,
        aux_loss=True, sum_group_losses=False, use_varifocal_loss=False,
        use_position_supervised_loss=False, ia_bce_loss=False,
        dataset_file="coco", coco_path=None, dataset_dir=None, square_resize_div_64=False,
        output_dir="output", dont_save_weights=False, checkpoint_interval=10, seed=42,
        resume="", start_epoch=0, eval=False, use_ema=False, ema_decay=0.9997, ema_tau=0,
        num_workers=2, device="cuda", world_size=1, dist_url="env://", sync_bn=True,
        fp16_eval=False, encoder_only=False, backbone_only=False, resolution=640,
        use_cls_token=False, multi_scale=False, expanded_scales=False,
        do_random_resize_via_padding=False, warmup_epochs=1, lr_scheduler="step",
        lr_min_factor=0.05, lr_stable_ratio=0.7, lr_restart_period=50, lr_restart_decay=0.8,
        early_stopping=True, early_stopping_patience=10, early_stopping_min_delta=0.001,
        early_stopping_use_ema=False, debug_data_limit=0, gradient_checkpointing=False,
        # use_rsa: mặc định False (khớp ModelConfig). Đếm param *có* SRA → chạy: --use-rsa
        use_windowed_attn=False, use_rsa=False, sra_shared=True, sra_G=64, sra_heads=16,
        use_convnext_projector=True,
        use_prototype_align=True, prototype_loss_coef=0.1, prototype_momentum=0.999,
        prototype_warmup_steps=200, prototype_temperature=0.1, prototype_repulsion_coef=0.1,
        prototype_use_freq_weight=True, prototype_use_quality_weight=True,
        prototype_use_repulsion=True, subcommand=None,
        segmentation_head=False, mask_downsample_ratio=4,
        mask_ce_loss_coef=1.0, mask_dice_loss_coef=1.0,
    )
    defaults.update(extra_kwargs)
    return argparse.Namespace(**defaults)
# fmt: on


def _cfg_kwargs(cfg) -> dict:
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()
    return cfg.dict()


def count_params(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def fp32_bytes_state_dict(sd: dict) -> int:
    return sum(v.numel() * v.element_size() for v in sd.values())


def serialized_state_dict_bytes(module: torch.nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.tell()


def fmt_millions(n: int) -> str:
    return f"{n / 1e6:.2f}M"


def fmt_mb(n_bytes: int) -> str:
    return f"{n_bytes / (1024**2):.2f} MiB"


def _find_swiftnet_weights(variant_name: str) -> str | None:
    """Locate SwiftNet encoder .pth under swiftnet_pretrained/ or repo root."""
    from rfdetrv2.util.swiftnet_pretrained import (
        SWIFTNET_WEIGHTS_BY_SIZE,
        swiftnet_pretrained_dir,
    )

    fname = SWIFTNET_WEIGHTS_BY_SIZE.get(variant_name)
    if not fname:
        return None
    for d in (swiftnet_pretrained_dir(PROJECT_ROOT), PROJECT_ROOT):
        p = d / fname
        if p.is_file():
            return str(p)
    return None


def _variants_from_config():
    """(name, config_instance) theo thứ tự tiny → small → base."""
    from rfdetrv2.config import (
        SwiftDetrBaseConfig,
        SwiftDetrSmallConfig,
        SwiftDetrTinyConfig,
    )

    return [
        ("tiny", SwiftDetrTinyConfig()),
        ("small", SwiftDetrSmallConfig()),
        ("base", SwiftDetrBaseConfig()),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Đếm param RF-DETR (4 biến thể config).")
    parser.add_argument(
        "--model-size",
        nargs="+",
        choices=["tiny", "small", "base"],
        default=None,
        metavar="NAME",
        help=(
            "Chọn một hoặc nhiều biến thể (vd. --model-size tiny base). "
            "Mặc định: đo cả tiny, small, base."
        ),
    )
    parser.add_argument(
        "--pretrained-encoder",
        default=None,
        help="Một file .pth dùng chung cho mọi variant (ghi đè auto). Thường không cần.",
    )
    parser.add_argument(
        "--use-rsa",
        action="store_true",
        help="Ghi đè use_rsa=True khi build (đếm thêm tham số SRA). Mặc định: không SRA.",
    )
    parser.add_argument(
        "--sra-G",
        type=int,
        default=None,
        dest="sra_G",
        metavar="G",
        help="Ghi đè sra_G (centroid). Mặc định: lấy từ ModelConfig.",
    )
    parser.add_argument(
        "--sra-heads",
        type=int,
        default=None,
        dest="sra_heads",
        metavar="H",
        help="Ghi đè sra_heads. Phải chia hết cho hidden dim của SRA (dim encoder).",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        dest="freeze_encoder",
        help="Ghi đè freeze_encoder=True (encoder không trainable; tổng param không đổi, trainable giảm).",
    )
    args_cli = parser.parse_args()

    # Import sau khi stub package (config không kéo detr)
    from rfdetrv2.models.lwdetr_prototype import build_model

    all_variants = _variants_from_config()
    if args_cli.model_size:
        by_name = {n: c for n, c in all_variants}
        variants = [(n, by_name[n]) for n in args_cli.model_size]
    else:
        variants = all_variants

    # Hiển thị G/head hiệu lực (từ config, có thể ghi đè CLI)
    _kw0 = _cfg_kwargs(variants[0][1])
    if args_cli.use_rsa:
        _kw0["use_rsa"] = True
    if args_cli.sra_G is not None:
        _kw0["sra_G"] = args_cli.sra_G
    if args_cli.sra_heads is not None:
        _kw0["sra_heads"] = args_cli.sra_heads
    if args_cli.freeze_encoder:
        _kw0["freeze_encoder"] = True
    rsa_note = " (use_rsa=True)" if args_cli.use_rsa else " (use_rsa=False)"
    if args_cli.use_rsa:
        rsa_note += f"  sra_G={_kw0.get('sra_G')}  sra_heads={_kw0.get('sra_heads')}"
    if args_cli.freeze_encoder:
        rsa_note += "  freeze_encoder=True"
    print(f"RF-DETR — parameters & size (device=cpu, full build_model){rsa_note}\n")
    print(
        f"{'name':<8} {'encoder':<16} {'total':>12} {'trainable':>12} "
        f"{'FP32≈':>12} {'state_dict':>12}"
    )
    print("-" * 76)

    for name, cfg in variants:
        kwargs = _cfg_kwargs(cfg)
        kwargs.pop("device", None)
        kwargs["device"] = "cpu"
        if args_cli.pretrained_encoder:
            kwargs["pretrained_encoder"] = args_cli.pretrained_encoder
        elif not kwargs.get("pretrained_encoder"):
            auto_p = _find_swiftnet_weights(name)
            if auto_p:
                kwargs["pretrained_encoder"] = auto_p

        if args_cli.use_rsa:
            kwargs["use_rsa"] = True
        if args_cli.sra_G is not None:
            kwargs["sra_G"] = args_cli.sra_G
        if args_cli.sra_heads is not None:
            kwargs["sra_heads"] = args_cli.sra_heads
        if args_cli.freeze_encoder:
            kwargs["freeze_encoder"] = True

        args = populate_args(**kwargs)
        try:
            model = build_model(args)
        except FileNotFoundError as e:
            print(
                f"\nLỗi khi build '{name}': {e}\n"
                "Đặt weight SwiftNet encoder (.pth) ở thư mục gốc repo hoặc "
                "swiftnet_pretrained/, hoặc chạy:\n"
                "  python scripts/count_model_params.py --pretrained-encoder /path/to/weights.pth\n",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        except ModuleNotFoundError as e:
            print(
                f"\nThiếu dependency khi build '{name}': {e}\n"
                "Cài thêm dependency (vd. pip install torchmetrics) nếu thiếu.\n",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        tot, trn = count_params(model)
        sd = model.state_dict()
        fp32_b = fp32_bytes_state_dict(sd)
        ser_b = serialized_state_dict_bytes(model)
        enc = cfg.encoder
        print(
            f"{name:<8} {enc:<16} {fmt_millions(tot):>12} {fmt_millions(trn):>12} "
            f"{fmt_mb(fp32_b):>12} {fmt_mb(ser_b):>12}"
        )

    print()
    print("FP32≈: bộ nhớ trọng số float32; state_dict: kích thước torch.save (nén).")
    print("Ghi chú: SwiftDetr-Base dùng SwiftNet-Base tại resolution 640 (xem config).")


if __name__ == "__main__":
    main()
