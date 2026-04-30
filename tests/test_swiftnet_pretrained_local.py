"""Local-only SwiftNet weight path resolution (no downloads)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SWIFTNET_PRETRAINED_PATH = _ROOT / "rfdetrv2" / "util" / "swiftnet_pretrained.py"


def _load_swiftnet_pretrained():
    name = "rfdetrv2.util.swiftnet_pretrained"
    spec = importlib.util.spec_from_file_location(name, _SWIFTNET_PRETRAINED_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {_SWIFTNET_PRETRAINED_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_snp = _load_swiftnet_pretrained()


def test_explicit_path_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _snp.resolve_pretrained_encoder_path(
            tmp_path,
            "tiny",
            explicit=str(tmp_path / "missing.pth"),
            weights_by_size=_snp.SWIFTNET_WEIGHTS_BY_SIZE,
        )

    p = tmp_path / "ok.pth"
    p.write_bytes(b"x")
    out = _snp.resolve_pretrained_encoder_path(
        tmp_path,
        "tiny",
        explicit=str(p),
        weights_by_size=_snp.SWIFTNET_WEIGHTS_BY_SIZE,
    )
    assert out == str(p.resolve())


def test_fallback_swiftnet_pretrained_dir(tmp_path: Path) -> None:
    hub = _snp.swiftnet_pretrained_dir(tmp_path)
    hub.mkdir(parents=True)
    fname = _snp.SWIFTNET_WEIGHTS_BY_SIZE["tiny"]
    (hub / fname).write_bytes(b"x")
    out = _snp.resolve_pretrained_encoder_path(
        tmp_path,
        "tiny",
        explicit=None,
        weights_by_size=_snp.SWIFTNET_WEIGHTS_BY_SIZE,
    )
    assert out == str((hub / fname).resolve())
