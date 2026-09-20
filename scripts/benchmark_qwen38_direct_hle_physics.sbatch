#!/usr/bin/env bash
#SBATCH --job-name=qwen38-hle-direct
#SBATCH --partition=train
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arjun@firstprinciples.com

# Serve Qwen3.8-27B locally and benchmark it directly—without conductor plans
# or frontier workers—on all 202 text-only HLE Physics questions.

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/qwen38_direct.yaml}"
SOURCE_MODEL="${QWEN_SOURCE_MODEL:-Qwen/Qwen3.8-27B}"
SERVED_MODEL="qwen3.8-27b-direct"
HOST="127.0.0.1"
PORT="8007"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-physics-qwen38-direct-262k}"
SERVER_LOG="${OUTPUT_DIR}/qwen-vllm-${SLURM_JOB_ID:-manual}.log"
READY_TIMEOUT="${QWEN_READY_TIMEOUT:-1800}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.90}"
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

for command in curl uv vllm; do
  command -v "$command" >/dev/null || { echo "Required command not found: $command" >&2; exit 2; }
done

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting local vLLM server for $SOURCE_MODEL on GPU ${QWEN_GPU:-0}"
CUDA_VISIBLE_DEVICES="${QWEN_GPU:-0}" vllm serve "$SOURCE_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  >"$SERVER_LOG" 2>&1 &
server_pid=$!

deadline=$((SECONDS + READY_TIMEOUT))
until curl --fail --silent --connect-timeout 5 --max-time 15 \
  "http://${HOST}:${PORT}/health" >/dev/null; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Qwen vLLM server exited during startup; log follows:" >&2
    tail -n 120 "$SERVER_LOG" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for the Qwen vLLM server; log follows:" >&2
    tail -n 120 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done

args=(
  --config "$CONFIG"
  --output-dir "$OUTPUT_DIR"
  --dataset hle-physics-text
  --full-dataset
  --expect-samples 202
  --use-model-output-limits
  --temperature "${BENCHMARK_TEMPERATURE:-0.0}"
  --concurrency "${BENCHMARK_CONCURRENCY:-8}"
  --bootstrap-samples "${BENCHMARK_BOOTSTRAP_SAMPLES:-10000}"
  --judge-concurrency "${JUDGE_CONCURRENCY:-8}"
  # Each judge request contains one answer. The requests may still run concurrently.
  --judge-batch-size "${JUDGE_BATCH_SIZE:-1}"
)
[[ "${BENCHMARK_JUDGE:-1}" == "1" ]] || args+=(--no-judge)

echo "Benchmarking Qwen3.8-27B directly on 202 HLE Physics questions"
echo "Results will be checkpointed in: $OUTPUT_DIR"

cd "$REPO_ROOT"
uv run theo-benchmark "${args[@]}"
