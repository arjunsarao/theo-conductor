#!/usr/bin/env bash
#SBATCH --job-name=conductor-format-grpo
#SBATCH --partition=train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Train only the conductor's workflow formatting and structural validity.
# Worker model endpoints are described by the registry for prompting and
# model-ID validation, but no worker servers are started or called.

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/local_small_models.yaml}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${REPO_ROOT}/outputs/grpo-${SLURM_JOB_ID:-manual}}"
WANDB_PROJECT="${WANDB_PROJECT:-theo-conductor-format}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
GPU_MEMORY_LOG_INTERVAL="${GPU_MEMORY_LOG_INTERVAL:-5}"

export HF_HOME="${THEO_HF_HOME:-/mnt/data/home/arjun/.cache/huggingface}"
export HF_HUB_CACHE="${THEO_HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${THEO_TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_XET_CACHE="${THEO_HF_XET_CACHE:-${HF_HOME}/xet}"

if [[ -n "${VLLM_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${VLLM_ENV}/bin/activate"
fi

for command in python nvidia-smi; do
  command -v "$command" >/dev/null || { echo "Required command not found: $command" >&2; exit 2; }
done
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }
mkdir -p "$TRAIN_OUTPUT_DIR" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_XET_CACHE"

GPU_MEMORY_MONITOR_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$GPU_MEMORY_MONITOR_PID" ]]; then
    kill "$GPU_MEMORY_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MEMORY_MONITOR_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ "$GPU_MEMORY_LOG_INTERVAL" != "0" ]]; then
  "${REPO_ROOT}/scripts/log_gpu_memory.sh" "$TRAIN_OUTPUT_DIR" "$GPU_MEMORY_LOG_INTERVAL" &
  GPU_MEMORY_MONITOR_PID=$!
fi

echo "Starting format-only GRPO on GPU(s) ${TRAIN_GPUS}; worker execution is disabled"
train_extra=(--use-vllm --wandb-project "$WANDB_PROJECT")
# shellcheck disable=SC2206
train_extra+=( $TRAIN_EXTRA_ARGS )
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m theo_conductor.train \
    --config-path "$CONFIG" \
    --output-dir "$TRAIN_OUTPUT_DIR" \
    "${train_extra[@]}"

echo "Format-only GRPO completed. Output: $TRAIN_OUTPUT_DIR"
