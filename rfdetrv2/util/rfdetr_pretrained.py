"""
Download RF-DETR COCO-trained checkpoints from Hugging Face into ``<project_root>/rfdetr_pretrained``.

Repo: https://huggingface.co/myn0908/rfdetrv2
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from rfdetrv2.util.http_download import download_file, exclusive_download_lock

logger = logging.getLogger(__name__)

HF_BASE_URL = "https://huggingface.co/myn0908/rfdetrv2/resolve/main"

# Official COCO checkpoints on Hub (filenames unchanged for compatibility)
SWIFTDETR_COCO_CHECKPOINT_BY_SIZE: dict[str, str] = {
    "tiny": "rfdetrv2_nano.pth",
    "small": "rfdetrv2_small.pth",
    "base": "rfdetrv2_base.pth",
}

RFDETR_COCO_CHECKPOINT_BY_SIZE = SWIFTDETR_COCO_CHECKPOINT_BY_SIZE


def rfdetr_pretrained_dir(project_root: Path | None = None) -> Path:
    """``<project_root>/rfdetr_pretrained``."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    return project_root / "rfdetr_pretrained"


def _hf_url(filename: str) -> str:
    return f"{HF_BASE_URL.rstrip('/')}/{filename}?download=true"


def download_rfdetr_coco_checkpoint(
    project_root: Path,
    model_size: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Download the COCO checkpoint for *model_size* if missing."""
    if model_size not in RFDETR_COCO_CHECKPOINT_BY_SIZE:
        raise ValueError(
            f"Unknown model_size={model_size!r}; expected one of "
            f"{sorted(RFDETR_COCO_CHECKPOINT_BY_SIZE)}"
        )
    fname = RFDETR_COCO_CHECKPOINT_BY_SIZE[model_size]
    out_dir = rfdetr_pretrained_dir(project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / fname
    if target.exists() and not overwrite:
        logger.debug("Skip existing RF-DETR checkpoint: %s", target)
        return target

    lock_path = out_dir / f".{fname}.download.lock"
    with exclusive_download_lock(lock_path):
        if target.exists() and not overwrite:
            return target
        url = _hf_url(fname)
        logger.info("Downloading RF-DETR COCO weights → %s  url=%s", target.name, url)
        print(f"Downloading {fname} from HuggingFace ...", file=sys.stderr)
        download_file(
            url,
            target,
            timeout=600,
            user_agent="RF-DETR/rfdetr_pretrained",
        )
        print(f"Saved → {target}", file=sys.stderr)
    return target


def resolve_rfdetr_coco_checkpoint(
    project_root: Path,
    model_size: str,
    *,
    explicit: str | None,
) -> str | None:
    """Return path to a COCO-pretrained RF-DETR ``.pth`` for ``Model(pretrain_weights=...)``.

    Parameters
    ----------
    explicit:
        Optional path from env/CLI. If it points to an existing file, that path is used.
        If set but missing, falls back to downloading the official Hub file for *model_size*.
        If ``None`` or empty, downloads (or reuses) under ``rfdetr_pretrained/``.

    Returns
    -------
    str | None
        Absolute path to ``.pth``, or ``None`` if ``RFDETR_SKIP_COCO_CHECKPOINT=1``.
    """
    if os.environ.get("RFDETR_SKIP_COCO_CHECKPOINT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return None

    candidates: list[Path] = []
    if explicit and str(explicit).strip():
        raw = Path(explicit.strip()).expanduser()
        if raw.is_absolute():
            candidates.append(raw.resolve())
        else:
            candidates.append((project_root / raw).resolve())

    for p in candidates:
        if p.is_file():
            logger.info("Using RF-DETR COCO checkpoint: %s", p)
            return str(p)

    if candidates:
        logger.warning(
            "COCO_WEIGHTS path not found (%s); downloading official checkpoint for size=%r.",
            explicit,
            model_size,
        )

    dest = download_rfdetr_coco_checkpoint(project_root, model_size)
    return str(dest.resolve())
