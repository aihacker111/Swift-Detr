"""
Resolve SwiftNet backbone checkpoints from local paths only (no downloads).

Use ``pretrained_encoder`` (CLI/config) for an explicit ``.pth``, or place weights under
``<project_root>/swiftnet_pretrained/`` or the project root using the default filenames below.
"""
from __future__ import annotations

from pathlib import Path

# SwiftDetr tier → default encoder filename when searching local dirs
SWIFTNET_WEIGHTS_BY_SIZE: dict[str, str] = {
    "tiny": "swiftnet_encoder_tiny.pth",
    "small": "swiftnet_encoder_small.pth",
    "base": "swiftnet_encoder_base.pth",
}


def swiftnet_pretrained_dir(project_root: Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    return project_root / "swiftnet_pretrained"


def resolve_pretrained_encoder_path(
    project_root: Path,
    model_size: str,
    *,
    explicit: str | None,
    weights_by_size: dict[str, str],
) -> str:
    """Return path to a SwiftNet encoder ``.pth`` on disk.

    Resolution order
    ----------------
    1. *explicit* — used as-is when non-empty (must exist).
    2. ``swiftnet_pretrained/<filename>`` under *project_root* when that file exists.
    3. ``<project_root>/<filename>`` when that file exists.

    Raises ``FileNotFoundError`` if no file is found. Set ``pretrained_encoder`` to your local
    ``.pth`` or add weights under ``swiftnet_pretrained/``.
    """
    if explicit and str(explicit).strip():
        p = Path(explicit.strip()).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"pretrained_encoder path does not exist: {p}")
        return str(p.resolve())

    fname = weights_by_size[model_size]
    p_hub = swiftnet_pretrained_dir(project_root) / fname
    p_root = project_root / fname

    if p_hub.is_file():
        return str(p_hub.resolve())
    if p_root.is_file():
        return str(p_root.resolve())

    raise FileNotFoundError(
        f"SwiftNet encoder weights for model_size={model_size!r} not found locally "
        f"(tried {p_hub} and {p_root}). "
        f"Set pretrained_encoder to a local .pth or copy weights to swiftnet_pretrained/."
    )
