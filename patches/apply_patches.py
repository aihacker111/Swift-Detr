#!/usr/bin/env python3
"""
Dynamic COCO class names and auto-detect ``build_coco`` live in the main tree:

  - ``rfdetrv2/util/coco_classes.py`` — ``load_classes_from_coco_json``,
    ``infer_classes_from_dataset_dir``, ``coco_classes_for_dataset``
  - ``rfdetrv2/datasets/coco.py`` — ``build_coco`` uses ``_resolve_build_coco_ann_and_images``

Run from repo root (no extra deps):

    python patches/apply_patches.py
"""

from __future__ import annotations

import py_compile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "rfdetrv2/util/coco_classes.py",
        "rfdetrv2/datasets/coco.py",
    ):
        path = root / rel
        assert path.is_file(), path
        py_compile.compile(str(path), doraise=True)
        print(f"OK: syntax {rel}")
    print("All checks passed (patches are integrated in-tree).")


if __name__ == "__main__":
    main()
