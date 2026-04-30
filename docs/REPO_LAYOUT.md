# Repository layout

## Top-level directories

| Path | Role |
|------|------|
| **`rfdetrv2/`** | Python package: models, engine, datasets, training `Model`, utilities. |
| **`dinov3/`** | Vendored DINOv3 torch.hub tree (`hubconf.py`). Required for `torch.hub.load(..., source="local")`; may be auto-populated from a zip if missing. |
| **`configs/`** | YAML presets for model sizes. |
| **`scripts/`** | **Canonical CLI tools**: training, inference, eval, visualization, param counting. See `scripts/README.md`. |
| **`tests/`** | Pytest (`pytest.ini` sets `pythonpath` and `testpaths`). |
| **`docs/`** | Project documentation (this file). |

## Root files (thin wrappers + config)

| Item | Role |
|------|------|
| **`train_supervised.py`**, **`inference.py`**, **`evaluate.py`**, … | One-line **wrappers** that execute `scripts/<same name>.py` so `python inference.py` still works from the repo root. |
| **`run_train_supervised.sh`** | Delegates to `scripts/run_train_supervised.sh`. |
| **`pytest.ini`** | Test discovery. |
| **`README.md`** | Overview and quick start. |

## Runtime artifacts (gitignored)

- `dinov3_pretrained/` — DINOv3 backbone `.pth` (Hugging Face `myn0908/dinov3`).
- `rfdetr_pretrained/` — RF-DETR COCO full checkpoints (Hugging Face `myn0908/rfdetrv2`), used e.g. by `finetune.py`.
- Checkpoints under your chosen `--output-dir`.
- See root `.gitignore` for caches and temp files.

## Conventions

- Set the **current working directory** to the repo root when running scripts.
- Prefer **`python scripts/<tool>.py`** in documentation; wrappers exist for convenience only.
- Do **not** move `dinov3/` without updating `DEFAULT_DINOV3_REPO_DIR` in `rfdetrv2/models/backbone/dinov3.py` and related helpers.
