#!/usr/bin/env python3
"""Print Swift-DETR parameter count, estimated weight size, and (optional) GFLOPs.

Run from the repo root or anywhere with the package on PYTHONPATH, e.g.:

    pip install -e .
    python scripts/model_stats.py --variant small --resolution 640

    # Load SWIFTNet ImageNet-1K (or compatible) trunk weights into the backbone:
    python scripts/model_stats.py tiny --encoder-imagenet-weights /path/to/swiftnet_tiny_imagenet.pth

GFLOP estimation needs ``thop`` (``pip install thop``). Some custom ops (e.g.
deformable attention) may be approximated; treat GFLOPs as a rough guide.
"""

from __future__ import annotations

import argparse
import os
import sys


def _repo_src_path() -> str:
    return os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "src"))


# Allow running without a prior ``pip install -e .``
if __name__ == "__main__":
    src = _repo_src_path()
    if src not in sys.path:
        sys.path.insert(0, src)


import torch
from swiftdetr.config import (  # noqa: E402
    SwiftDetrBaseConfig,
    SwiftDetrSmallConfig,
    SwiftDetrTinyConfig,
    TrainConfig,
)
from swiftdetr.models import build_model_from_config  # noqa: E402
from swiftdetr.util.tensors import nested_tensor_from_tensor_list  # noqa: E402

_VARIANTS = {
    "tiny": SwiftDetrTinyConfig,
    "small": SwiftDetrSmallConfig,
    "base": SwiftDetrBaseConfig,
}


def format_int(n: int) -> str:
    return f"{n:,}".replace(",", "_")


def parameter_stats(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def storage_bytes(model: torch.nn.Module) -> tuple[int, int]:
    """(params_bytes, buffers_bytes) in bytes on disk as stored in state_dict."""
    p_b = sum(p.numel() * p.element_size() for p in model.parameters())
    b_b = sum(t.numel() * t.element_size() for t in model.buffers())
    return p_b, b_b


def estimate_fp32_pth_size_mb(params_bytes: int) -> float:
    """Uncompressed float32 state_dict–style size in MiB (≈ .pth on disk, no zip)."""
    return params_bytes / (1024**2)


def try_gflops(
    model: torch.nn.Module,
    samples: object,
) -> float | None:
    try:
        from thop import profile  # type: ignore[import-not-found]
    except ImportError:
        return None

    model = model.eval()
    with torch.inference_mode():
        try:
            macs, _params = profile(model, inputs=(samples,), verbose=False)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  (thop profile failed: {type(exc).__name__}: {exc}\n"
                "   GFLOPs are approximate; try a different resolution or PyTorch build.)"
            )
            return None
    gflops = 2.0 * float(macs) * 1e-9
    return gflops


def build_and_dummy_input(
    variant: str,
    resolution: int,
    device: str,
    encoder_imagenet_weights: str | None = None,
) -> tuple[torch.nn.Module, object]:
    cfg_cls = _VARIANTS[variant]
    kw: dict = {
        "resolution": resolution,
        "positional_encoding_size": resolution // 16,
    }
    if encoder_imagenet_weights is not None:
        kw["encoder_imagenet_weights"] = os.path.realpath(
            os.path.expanduser(encoder_imagenet_weights)
        )
    mc = cfg_cls(**kw)
    tr = TrainConfig(dataset_dir=".", output_dir=".")
    model = build_model_from_config(mc, tr)
    model = model.to(device)
    # nested_tensor_from_tensor_list expects (C, H, W) per image, not (B, C, H, W).
    x = torch.randn(3, resolution, resolution, device=device)
    samples = nested_tensor_from_tensor_list([x])
    return model, samples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "variant",
        nargs="?",
        default="small",
        choices=tuple(_VARIANTS),
        help="Model variant (default: small).",
    )
    ap.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=None,
        help="Input square size (default: variant default, e.g. 512 for tiny, 640 for small/base).",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda (default: cpu, best for a stable FLOP count).",
    )
    ap.add_argument(
        "--encoder-imagenet-weights",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Optional path to a SWIFTNet ImageNet-1K checkpoint; loaded into the "
            "detection backbone (same as training ``encoder_imagenet_weights``)."
        ),
    )
    args = ap.parse_args()
    if args.resolution is None:
        # Match config defaults; tiny=512, small/base=640
        if args.variant == "tiny":
            args.resolution = 512
        else:
            args.resolution = 640

    enc_path = None
    if args.encoder_imagenet_weights:
        enc_path = os.path.realpath(os.path.expanduser(args.encoder_imagenet_weights))
        if not os.path.isfile(enc_path):
            print(f"error: file not found: {enc_path}", file=sys.stderr)
            sys.exit(1)

    model, samples = build_and_dummy_input(
        args.variant,
        args.resolution,
        args.device,
        encoder_imagenet_weights=enc_path,
    )
    print(model)
    total, trn = parameter_stats(model)
    p_b, b_b = storage_bytes(model)
    all_b = p_b + b_b

    print("Swift-DETR stats")
    print("----------------")
    print(f"  variant:        {args.variant}")
    print(f"  resolution:     {args.resolution} x {args.resolution}")
    if enc_path is not None:
        print(f"  ImageNet trunk:  {enc_path}  (loaded into backbone[0].encoder)")
    print(f"  total params:   {format_int(total)}  ({total / 1e6:.2f} M)")
    print(f"  trainable:      {format_int(trn)}  ({trn / 1e6:.2f} M)")
    print(f"  param storage:  {p_b / (1024**2):.2f} MiB  (as dtype in memory/fp32 weights)")
    print(f"  buffer storage: {b_b / (1024**2):.2f} MiB")
    print(f"  total tensors:  {all_b / (1024**2):.2f} MiB  (params + registered buffers)")
    print(f"  est. .pth fp32: ~{estimate_fp32_pth_size_mb(p_b):.2f} MiB  (uncompressed, params only)")

    gfl = try_gflops(model, samples)
    if gfl is not None:
        print(f"  GFLOPs (approx, thop @ 1 image):  {gfl:.2f}")
    else:
        print("  GFLOPs:         install thop (``pip install thop``) and re-run, or use a profiler.")

    print(
        "\nNote: GFLOP tools often under/over-count custom attention; use for comparison only."
    )


if __name__ == "__main__":
    main()
