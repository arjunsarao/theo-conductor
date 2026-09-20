#!/usr/bin/env bash
#SBATCH --job-name=hle-workflow
#SBATCH --partition=train
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arjun@firstprinciples.com

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PLANS="${BENCHMARK_PLANS:-${REPO_ROOT}/outputs/hle-physics}"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/worker_pool_frontier.yaml}"
SHARD_SIZE="${BENCHMARK_SHARD_SIZE:-202}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
BASE_OFFSET="${BENCHMARK_BASE_OFFSET:-0}"
OFFSET="${BENCHMARK_OFFSET:-$((BASE_OFFSET + SHARD_INDEX * SHARD_SIZE))}"
LAST_POSITION=$((OFFSET + SHARD_SIZE - 1))
RUN_NAME="${BENCHMARK_RUN_NAME:-hle-physics-text}"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/outputs/${RUN_NAME}-$(printf '%04d' "$OFFSET")-$(printf '%04d' "$LAST_POSITION")}"

mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_DIR"
[[ -e "$PLANS" ]] || { echo "Plans not found: $PLANS" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }
command -v uv >/dev/null || { echo "Required command not found: uv" >&2; exit 2; }

args=(
  --plans "$PLANS"
  --config "$CONFIG"
  --output-dir "$OUTPUT_DIR"
  --max-samples "$SHARD_SIZE"
  --offset "$OFFSET"
  --concurrency "${BENCHMARK_CONCURRENCY:-4}"
)

[[ "${BENCHMARK_JUDGE:-1}" == "1" ]] || args+=(--no-judge)
[[ "${BENCHMARK_RETRY_FAILURES:-0}" == "1" ]] && args+=(--retry-failures)
[[ "${BENCHMARK_INCLUDE_MULTIMODAL:-0}" == "1" ]] && args+=(--include-multimodal)
[[ "${BENCHMARK_BATCH_WORKERS:-0}" == "1" ]] && args+=(--batch-workers)
[[ -n "${WORKER_BATCH_SIZE:-}" ]] && args+=(--worker-batch-size "$WORKER_BATCH_SIZE")
[[ -n "${MAX_WORKER_TOKENS:-}" ]] && args+=(--max-worker-tokens "$MAX_WORKER_TOKENS")

echo "Running HLE physics text-only workflow shard at offset ${OFFSET}, up to ${SHARD_SIZE} questions"
echo "Output directory: ${OUTPUT_DIR}"

cd "$REPO_ROOT"
uv run theo-workflow-benchmark "${args[@]}"
