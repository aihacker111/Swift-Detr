# Scripts (CLI entry points)

Run from the **repository root** so `import rfdetrv2` resolves.

| Script | Purpose |
|--------|---------|
| `train_supervised.py` | COCO supervised training (`torchrun` via `run_train_supervised.sh`). |
| `run_train_supervised.sh` | Multi-GPU launcher for `train_supervised.py`. |
| `run_finetune.sh` | Multi-GPU launcher for `finetune.py` (`torchrun`, default 2 GPUs). |
| `inference.py` | Image inference. |
| `inference_video.py` | Video inference. |
| `evaluate.py` | COCO val evaluation. |
| `visualize_attention.py` | RoPE2D / decoder attention visualization. |
| `visualize_prototype_rope2d.py` | Prototype vs RoPE2D visualization. |
| `count_model_params.py` | Parameter counts per model size (avoids heavy `rfdetrv2` import path). |

Shallow **wrappers** at repo root (`train_supervised.py`, `inference.py`, …) forward to these files so short commands keep working:

```bash
python inference.py --help
python scripts/inference.py --help   # equivalent
```
