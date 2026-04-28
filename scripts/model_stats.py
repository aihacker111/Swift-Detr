#!/usr/bin/env python3
"""Print Swift-DETR parameter count, weight size, and GFLOPs.

Run from the repo root or anywhere with the package on PYTHONPATH, e.g.:

    pip install -e .
    python scripts/model_stats.py --variant small --resolution 640
    python scripts/model_stats.py tiny --no-gflops

GFLOPs are measured with ``torch.utils.flop_counter.FlopCounterMode`` (PyTorch >= 2.1)
with ``thop`` as a fallback (``pip install thop``).
"""

from __future__ import annotations

import argparse
import os
import sys
import types


# ---------------------------------------------------------------------------
# Bootstrap: path + stub swiftdetr/__init__.py + stub torchvision if broken
# Must run before ANY swiftdetr or torchvision import.
# ---------------------------------------------------------------------------

def _src_path() -> str:
    return os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "src"))


def _bootstrap() -> None:
    import torch  # import torch first (it works fine)

    src = _src_path()
    if src not in sys.path:
        sys.path.insert(0, src)

    # --- Stub swiftdetr package to skip __init__.py → torchvision import chain ---
    if "swiftdetr" not in sys.modules:
        pkg = types.ModuleType("swiftdetr")
        pkg.__path__ = [os.path.join(src, "swiftdetr")]  # type: ignore[assignment]
        pkg.__package__ = "swiftdetr"
        sys.modules["swiftdetr"] = pkg

    # --- Stub torchvision if the real import would crash (PIL version mismatch) ---
    # Only stub the specific symbols used by the model build + forward path:
    #   torchvision.__version__, torchvision._is_tracing()   (tensors.py, math.py)
    #   torchvision.ops.boxes.box_area                        (box_ops.py)
    #   torchvision.ops.misc                                  (math.py, modern tvision skips it)
    _need_stub = False
    try:
        import torchvision  # noqa: F401
    except Exception:
        _need_stub = True
        # Clear any broken partial entry left by the failed import
        for key in [k for k in sys.modules if k.startswith("torchvision")]:
            del sys.modules[key]

    if _need_stub:
        def _box_area(boxes: torch.Tensor) -> torch.Tensor:
            return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        tv         = types.ModuleType("torchvision")
        tv_ops     = types.ModuleType("torchvision.ops")
        tv_boxes   = types.ModuleType("torchvision.ops.boxes")
        tv_misc    = types.ModuleType("torchvision.ops.misc")

        tv.__version__    = "0.17.0"   # > 0.7 so math.py skips legacy code path
        tv._is_tracing    = lambda: False
        tv.ops            = tv_ops
        tv_ops.boxes      = tv_boxes
        tv_ops.misc       = tv_misc
        tv_boxes.box_area = _box_area

        sys.modules["torchvision"]            = tv
        sys.modules["torchvision.ops"]        = tv_ops
        sys.modules["torchvision.ops.boxes"]  = tv_boxes
        sys.modules["torchvision.ops.misc"]   = tv_misc


_bootstrap()

# ---------------------------------------------------------------------------
# Real imports — after bootstrap
# ---------------------------------------------------------------------------

import torch  # noqa: E402 (already imported in _bootstrap, re-import for type hints)
from swiftdetr.config import (  # noqa: E402
    SwiftDetrBaseConfig,
    SwiftDetrSmallConfig,
    SwiftDetrTinyConfig,
    TrainConfig,
)
from swiftdetr.models.swiftdetr import build_model_from_config  # noqa: E402
from swiftdetr.util.tensors import nested_tensor_from_tensor_list  # noqa: E402

_VARIANTS = {
    "tiny":  SwiftDetrTinyConfig,
    "small": SwiftDetrSmallConfig,
    "base":  SwiftDetrBaseConfig,
}


# ---------------------------------------------------------------------------
# Parameter / storage helpers
# ---------------------------------------------------------------------------

def _count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _count_trainable(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _storage_bytes(model: torch.nn.Module) -> tuple[int, int]:
    p_b = sum(p.numel() * p.element_size() for p in model.parameters())
    b_b = sum(t.numel() * t.element_size() for t in model.buffers())
    return p_b, b_b


def _fmt(n: int) -> str:
    s = f"{n:,}".replace(",", "_")
    return f"{s}  ({n / 1e6:.2f} M)"


def component_table(model: torch.nn.Module) -> list[tuple[str, int]]:
    """Return (label, param_count) rows for the key sub-components."""
    rows: list[tuple[str, int]] = []

    backbone_wrapper = getattr(model, "backbone", None)
    if backbone_wrapper is not None and len(backbone_wrapper) >= 1:
        net = backbone_wrapper[0]
        encoder   = getattr(net, "encoder",   None)
        projector = getattr(net, "projector", None)
        pe = backbone_wrapper[1] if len(backbone_wrapper) > 1 else None
        if encoder is not None:
            rows.append(("backbone.encoder (SwiftNet)", _count(encoder)))
        if projector is not None:
            rows.append(("backbone.projector (ConvNeXt)", _count(projector)))
        if pe is not None and _count(pe) > 0:
            rows.append(("backbone.pos_encoding", _count(pe)))

    for name in ("transformer", "encoder"):
        m = getattr(model, name, None)
        if m is not None:
            rows.append((name, _count(m)))
            break

    for name in ("class_embed", "bbox_embed"):
        m = getattr(model, name, None)
        if m is not None:
            rows.append((name, _count(m)))

    return rows


# ---------------------------------------------------------------------------
# GFLOPs — native FlopCounterMode (PyTorch >= 2.1) with thop fallback
# ---------------------------------------------------------------------------

def _flops_native(model: torch.nn.Module, samples: object) -> float | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    try:
        with torch.no_grad():
            with FlopCounterMode(display=False) as fc:
                model(samples)
        return float(fc.get_total_flops()) * 1e-9
    except Exception as exc:
        print(f"  (FlopCounterMode error: {type(exc).__name__}: {exc})")
        return None


def _flops_thop(model: torch.nn.Module, samples: object) -> float | None:
    try:
        from thop import profile  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with torch.inference_mode():
            macs, _ = profile(model, inputs=(samples,), verbose=False)
        return 2.0 * float(macs) * 1e-9
    except Exception as exc:
        print(f"  (thop error: {type(exc).__name__}: {exc})")
        return None


def estimate_gflops(model: torch.nn.Module, samples: object) -> tuple[float | None, str]:
    gfl = _flops_native(model, samples)
    if gfl is not None:
        return gfl, "torch.FlopCounterMode"
    gfl = _flops_thop(model, samples)
    if gfl is not None:
        return gfl, "thop"
    return None, "unavailable"


# ---------------------------------------------------------------------------
# Build model + dummy input
# ---------------------------------------------------------------------------

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
    x = torch.randn(3, resolution, resolution, device=device)
    samples = nested_tensor_from_tensor_list([x])
    return model, samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        "-r", "--resolution",
        type=int,
        default=None,
        help="Input square size (default: 512 for tiny, 640 for small/base).",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda (default: cpu — stable FLOP counts).",
    )
    ap.add_argument(
        "--no-gflops",
        action="store_true",
        help="Skip GFLOPs estimation (faster when only param counts are needed).",
    )
    ap.add_argument(
        "--encoder-imagenet-weights",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional SWIFTNet ImageNet-1K checkpoint to load into the backbone.",
    )
    args = ap.parse_args()

    if args.resolution is None:
        args.resolution = 512 if args.variant == "tiny" else 640

    enc_path = None
    if args.encoder_imagenet_weights:
        enc_path = os.path.realpath(os.path.expanduser(args.encoder_imagenet_weights))
        if not os.path.isfile(enc_path):
            print(f"error: file not found: {enc_path}", file=sys.stderr)
            sys.exit(1)

    print(f"Building Swift-DETR-{args.variant}  {args.resolution}x{args.resolution} on {args.device} …")
    model, samples = build_and_dummy_input(
        args.variant, args.resolution, args.device,
        encoder_imagenet_weights=enc_path,
    )
    model.eval()

    total     = _count(model)
    trainable = _count_trainable(model)
    p_b, b_b  = _storage_bytes(model)

    W = 58
    print()
    print("=" * W)
    print(f"  Swift-DETR  variant={args.variant}  res={args.resolution}")
    print("=" * W)
    print(f"  total params  : {_fmt(total)}")
    print(f"  trainable     : {_fmt(trainable)}")
    print()
    print("  Component breakdown:")
    rows = component_table(model)
    for label, cnt in rows:
        pct = 100.0 * cnt / total if total else 0.0
        print(f"    {label:<38}  {cnt / 1e6:6.2f} M  ({pct:4.1f}%)")
    print()
    print(f"  param storage : {p_b / (1024**2):.2f} MiB  (fp32 in memory)")
    print(f"  buffer storage: {b_b / (1024**2):.2f} MiB")
    print(f"  est. .pth size: ~{p_b / (1024**2):.2f} MiB  (params only, uncompressed)")
    if enc_path is not None:
        print(f"  ImageNet trunk: {enc_path}")

    if not args.no_gflops:
        print()
        print("  Estimating GFLOPs (1 image, eval, no_grad) …")
        gfl, source = estimate_gflops(model, samples)
        if gfl is not None:
            print(f"  GFLOPs        : {gfl:.2f}  (via {source})")
            print("  Note: deformable-attention MACs may be approximated.")
        else:
            print("  GFLOPs        : not available")
            print("  Upgrade to PyTorch >= 2.1 or install thop (pip install thop).")
    else:
        print()
        print("  GFLOPs        : skipped (--no-gflops)")

    print("=" * W)


if __name__ == "__main__":
    main()
