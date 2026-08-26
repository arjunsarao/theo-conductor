#!/usr/bin/env bash
#SBATCH --job-name=hle-plan
#SBATCH --partition=train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/worker_pool_frontier.yaml}"
DATASET="${DATASET:-hle-all}"
PLAN_SHARDS="${PLAN_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
PLAN_RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${PLAN_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-plans-${PLAN_RUN_ID}}"
CONDUCTOR_HOST="${CONDUCTOR_HOST:-127.0.0.1}"
CONDUCTOR_PORT="${CONDUCTOR_PORT:-8007}"
CONDUCTOR_READY_TIMEOUT="${CONDUCTOR_READY_TIMEOUT:-1800}"
CONDUCTOR_GPU_MEMORY_UTILIZATION="${CONDUCTOR_GPU_MEMORY_UTILIZATION:-0.90}"
CONDUCTOR_MAX_MODEL_LEN="${CONDUCTOR_MAX_MODEL_LEN:-16384}"
PLAN_CONCURRENCY="${PLAN_CONCURRENCY:-16}"
PLAN_MAX_TOKENS="${PLAN_MAX_TOKENS:-2048}"
PLAN_ATTEMPTS="${PLAN_ATTEMPTS:-2}"
PLAN_TEMPERATURE="${PLAN_TEMPERATURE:-0.1}"
USER_HF_HOME="${THEO_HF_HOME:-${HOME}/.cache/huggingface}"

export HF_HOME="$USER_HF_HOME"
export HF_HUB_CACHE="${THEO_HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${THEO_TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_XET_CACHE="${THEO_HF_XET_CACHE:-${HF_HOME}/xet}"

mkdir -p \
  "${REPO_ROOT}/slurm_logs" \
  "$OUTPUT_DIR" \
  "$HF_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_XET_CACHE"
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }

if [[ -n "${VLLM_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${VLLM_ENV}/bin/activate"
fi

for command in python curl vllm; do
  command -v "$command" >/dev/null || { echo "Required command not found: $command" >&2; exit 2; }
done

CONFIG_CONDUCTOR_MODEL="$(python - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
model = config.get("conductor_model")
if not isinstance(model, str) or not model.strip():
    raise SystemExit("Config must define conductor_model or set CONDUCTOR_MODEL")
print(model)
PY
)"
CONDUCTOR_SOURCE_MODEL="${CONDUCTOR_SOURCE_MODEL:-$CONFIG_CONDUCTOR_MODEL}"
if [[ -n "${CONDUCTOR_LORA_PATH:-}" ]]; then
  CONDUCTOR_MODEL="${CONDUCTOR_MODEL:-theo-conductor}"
  BASE_SERVED_MODEL="${CONDUCTOR_BASE_SERVED_MODEL:-$CONDUCTOR_SOURCE_MODEL}"
else
  CONDUCTOR_MODEL="${CONDUCTOR_MODEL:-$CONFIG_CONDUCTOR_MODEL}"
  BASE_SERVED_MODEL="${CONDUCTOR_BASE_SERVED_MODEL:-$CONDUCTOR_MODEL}"
fi
SERVER_LOG="${OUTPUT_DIR}/conductor-server-shard-${SHARD_INDEX}.log"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting conductor $CONDUCTOR_SOURCE_MODEL for shard ${SHARD_INDEX}/${PLAN_SHARDS}"
server_args=(
  --host "$CONDUCTOR_HOST"
  --port "$CONDUCTOR_PORT"
  --served-model-name "$BASE_SERVED_MODEL"
  --tensor-parallel-size 1
  --gpu-memory-utilization "$CONDUCTOR_GPU_MEMORY_UTILIZATION"
  --max-model-len "$CONDUCTOR_MAX_MODEL_LEN"
)
if [[ -n "${CONDUCTOR_LORA_PATH:-}" ]]; then
  server_args+=(--enable-lora --lora-modules "${CONDUCTOR_MODEL}=${CONDUCTOR_LORA_PATH}")
fi
CUDA_VISIBLE_DEVICES="${CONDUCTOR_GPU:-0}" vllm serve "$CONDUCTOR_SOURCE_MODEL" \
  "${server_args[@]}" \
  >"$SERVER_LOG" 2>&1 &
server_pid=$!

deadline=$((SECONDS + CONDUCTOR_READY_TIMEOUT))
until curl --fail --silent --connect-timeout 5 --max-time 15 \
  "http://${CONDUCTOR_HOST}:${CONDUCTOR_PORT}/health" >/dev/null; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -n 120 "$SERVER_LOG" >&2
    exit 1
  fi
  (( SECONDS < deadline )) || { tail -n 120 "$SERVER_LOG" >&2; exit 1; }
  sleep 5
done

args=(
  generate
  --dataset "$DATASET"
  --config "$CONFIG"
  --output-dir "$OUTPUT_DIR"
  --conductor-base-url "http://${CONDUCTOR_HOST}:${CONDUCTOR_PORT}/v1"
  --conductor-model "$CONDUCTOR_MODEL"
  --conductor-source "${CONDUCTOR_LORA_PATH:-$CONDUCTOR_SOURCE_MODEL}"
  --shard-index "$SHARD_INDEX"
  --shard-count "$PLAN_SHARDS"
  --concurrency "$PLAN_CONCURRENCY"
  --max-tokens "$PLAN_MAX_TOKENS"
  --temperature "$PLAN_TEMPERATURE"
  --attempts "$PLAN_ATTEMPTS"
  --seed "${PLAN_SEED:-42}"
)
[[ -n "${DATASET_SAMPLES:-}" ]] && args+=(--max-samples "$DATASET_SAMPLES")
[[ "${PLAN_RETRY_INVALID:-0}" == "1" ]] && args+=(--retry-invalid)

PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m theo_conductor.plan "${args[@]}"

echo "Planning shard completed: $OUTPUT_DIR"
