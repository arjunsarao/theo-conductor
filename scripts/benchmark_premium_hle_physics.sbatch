#!/usr/bin/env bash
#SBATCH --job-name=hle-premium
#SBATCH --partition=train
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Benchmark GPT-5.5 and Claude Opus 4.8 independently on the complete
# 202-question text-only HLE Physics set using ordinary concurrent requests.

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/worker_pool_premium.yaml}"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-physics-premium-live}"

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
  --bootstrap-samples "${BENCHMARK_BOOTSTRAP_SAMPLES:-10000}"
  --judge-concurrency "${JUDGE_CONCURRENCY:-8}"
  --judge-batch-size "${JUDGE_BATCH_SIZE:-10}"
)

[[ "${BENCHMARK_JUDGE:-1}" == "1" ]] || args+=(--no-judge)

echo "Benchmarking GPT-5.5 and Claude Opus 4.8 over 202 HLE Physics questions"
echo "Results will be checkpointed in: $OUTPUT_DIR"

cd "$REPO_ROOT"
uv run theo-benchmark "${args[@]}"
