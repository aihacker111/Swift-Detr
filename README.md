# RF-DETR (rfdetrv2)

Object detection built on RF-DETR with DINOv3 backbones, supervised training, and optional prototype alignment.

## Layout

See **[docs/REPO_LAYOUT.md](docs/REPO_LAYOUT.md)** and **[scripts/README.md](scripts/README.md)** for folders, CLIs, and conventions.

## Quick start (supervised training)

From the repository root:

```bash
# Multi-GPU (edit env vars in the script as needed)
./run_train_supervised.sh
# or
./scripts/run_train_supervised.sh
```

```bash
python scripts/train_supervised.py --help
# equivalent:
python train_supervised.py --help
```

Weights resolve under `dinov3_pretrained/` or download automatically; DINOv3 hub source is ensured under `dinov3/` when needed.

## Other tools (inference, eval, viz)

All live under **`scripts/`**; the same filenames at the repo root forward to them, e.g.:

```bash
python scripts/inference.py --help
python inference.py --help    # same
python scripts/evaluate.py --weights ... --dataset-dir ...
python scripts/count_model_params.py
```

## Tests

```bash
pip install pytest
pytest
```

## Requirements

Install project dependencies for `rfdetrv2` (PyTorch, supervision, etc.) per your environment; `dinov3/requirements.txt` covers the vendored hub tree if you develop inside `dinov3/`.
