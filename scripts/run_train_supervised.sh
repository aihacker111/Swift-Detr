#!/usr/bin/env bash
# RF-DETR supervised training (multi-GPU via torchrun).
#
# Environment (optional):
#   DATASET_DIR   - COCO root (default: /workspace/coco_2017)
#   OUTPUT_DIR    - run output (default: /workspace/output/rfdetrv2_small_supervised)
#   NUM_GPUS      - torchrun --nproc_per_node (default: 2)
#   MASTER_PORT   - distributed rendezvous port (default: 29500)
#   CUDA_VISIBLE_DEVICES - which GPUs to use (default below: 0,1)

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TRAIN_PY="${SCRIPT_DIR}/train_supervised.py"

DATASET_DIR="${DATASET_DIR:-/workspace/coco_2017}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/rfdetrv2_small_supervised}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
MASTER_PORT="${MASTER_PORT:-29500}"

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

echo "Starting supervised training: ${NUM_GPUS} process(es)"
echo "  dataset: ${DATASET_DIR}"
echo "  output:  ${OUTPUT_DIR}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  "${TRAIN_PY}" \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE_PER_GPU}" \
  --num-workers 4 \
  --epochs 50 \
  --model-size small \
  --use-varifocal-loss \
  --tensorboard

echo "Done."
