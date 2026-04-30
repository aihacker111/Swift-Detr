#!/usr/bin/env bash
# Fine-tune RF-DETRv2 (finetune.py) on multiple GPUs via torchrun.
#
# Environment (optional):
#   DATASET_DIR   - COCO-format dataset root
#   OUTPUT_DIR    - run output
#   NUM_GPUS      - torchrun --nproc_per_node (default: 2)
#   MASTER_PORT   - distributed rendezvous port (default: 29501)
#   CUDA_VISIBLE_DEVICES - which GPUs to use (default: 0,1)
#
# Per-GPU batch size is BATCH_SIZE in finetune.py; global batch ≈ BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FINETUNE_PY="${SCRIPT_DIR}/../finetune.py"

NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"

echo "Checking GPUs..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
  NUM_AVAILABLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
else
  echo "Warning: nvidia-smi not found."
  NUM_AVAILABLE_GPUS=0
fi

if [[ "${NUM_AVAILABLE_GPUS}" =~ ^[0-9]+$ ]] && (( NUM_AVAILABLE_GPUS < NUM_GPUS )); then
  echo "Warning: requested ${NUM_GPUS} GPU(s), only ${NUM_AVAILABLE_GPUS} visible — using ${NUM_AVAILABLE_GPUS}."
  NUM_GPUS="${NUM_AVAILABLE_GPUS}"
fi
if (( NUM_GPUS < 1 )); then
  echo "Error: need at least one GPU for this script." >&2
  exit 1
fi

echo "Starting finetune: ${NUM_GPUS} process(es)"
echo "  dataset: ${DATASET_DIR:-<finetune.py default>}"
echo "  output:  ${OUTPUT_DIR:-<finetune.py default>}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  "${FINETUNE_PY}"

echo "Done."
