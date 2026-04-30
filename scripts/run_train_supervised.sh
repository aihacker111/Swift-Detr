#!/usr/bin/env bash
# SwiftDetr supervised training (multi-GPU via torchrun).
#
# Environment (optional):
#   DATASET_DIR          - COCO root (default: /workspace/coco_2017)
#   OUTPUT_DIR           - run output (default: /workspace/output/swiftdetr_base_supervised)
#   NUM_GPUS             - torchrun --nproc_per_node (default: 2)
#   MASTER_PORT          - distributed rendezvous port (default: 29500)
#   CUDA_VISIBLE_DEVICES - which GPUs to use (default below: 0,1)
#   MODEL_SIZE           - tiny | small | base (default: base)
#   PRETRAINED_ENCODER   - path to SwiftNet encoder .pth (passed as --pretrained-encoder when set)

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TRAIN_PY="${SCRIPT_DIR}/train_supervised.py"

DATASET_DIR="${DATASET_DIR:-/workspace/coco}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/swiftdetr_base_supervised}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
MASTER_PORT="${MASTER_PORT:-29500}"
MODEL_SIZE="${MODEL_SIZE:-base}"
PRETRAINED_ENCODER="${PRETRAINED_ENCODER:-/workspace/Swift-Detr/checkpoints/swift_net_base/2026_04_26_15_28_59/checkpoint_best.pth}"

PRETRAIN_ARGS=()
if [[ -n "${PRETRAINED_ENCODER}" ]]; then
  PRETRAIN_ARGS+=(--pretrained-encoder "${PRETRAINED_ENCODER}")
fi

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
echo "  model:   ${MODEL_SIZE}"
if [[ -n "${PRETRAINED_ENCODER}" ]]; then
  echo "  pretrained_encoder: ${PRETRAINED_ENCODER}"
else
  echo "  pretrained_encoder: (unset — train_supervised resolves swiftnet_pretrained/ or env PRETRAINED_ENCODER)"
fi

torchrun --standalone --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  /Users/tinvo0908/Downloads/swift-detr/Swift-Detr/scripts/train_supervised.py \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE_PER_GPU}" \
  --num-workers 16 \
  --epochs 50 \
  --model-size "${MODEL_SIZE}" \
  "${PRETRAIN_ARGS[@]}" \
  --use-varifocal-loss \
  --tensorboard

echo "Done."
