#!/usr/bin/env bash
#SBATCH --job-name=hle-frontier-workers
#SBATCH --partition=train
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arjun@firstprinciples.com

# Benchmark every model in worker_pool_frontier.yaml independently on the
# complete 202-question, text-only HLE Physics subset. GPT and Claude use
# OpenRouter's provider Batch API; the remaining models use live requests.

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/worker_pool_frontier.yaml}"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-physics-frontier}"

mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_DIR"
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }
command -v uv >/dev/null || { echo "Required command not found: uv" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || {
  echo "OPENROUTER_API_KEY must be exported when this job is submitted" >&2
  exit 2
}

args=(
  --config "$CONFIG"
  --output-dir "$OUTPUT_DIR"
  --dataset hle-physics-text
  --full-dataset
  --expect-samples 202
  --use-model-output-limits
  --temperature "${BENCHMARK_TEMPERATURE:-0.0}"
  --concurrency "${BENCHMARK_CONCURRENCY:-4}"
  --batch-model gpt-5.5
  --batch-model claude-opus-4.8
  --worker-batch-size "${WORKER_BATCH_SIZE:-202}"
  --bootstrap-samples "${BENCHMARK_BOOTSTRAP_SAMPLES:-10000}"
  --judge-concurrency "${JUDGE_CONCURRENCY:-8}"
  --judge-batch-size "${JUDGE_BATCH_SIZE:-10}"
)

[[ "${BENCHMARK_JUDGE:-1}" == "1" ]] || args+=(--no-judge)

echo "Benchmarking all frontier workers on 202 text-only HLE Physics questions"
echo "Results will be checkpointed in: $OUTPUT_DIR"

cd "$REPO_ROOT"
uv run theo-benchmark "${args[@]}"
