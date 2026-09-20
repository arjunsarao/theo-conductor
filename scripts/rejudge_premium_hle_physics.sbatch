#!/usr/bin/env bash
#SBATCH --job-name=hle-premium-rejudge
#SBATCH --partition=train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Resume semantic judging for the existing GPT-5.5 and Claude Opus 4.8
# HLE Physics results. Completed generations and valid verdicts are reused.

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/worker_pool_premium.yaml}"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-physics-premium-live}"
RESULTS_PATH="${OUTPUT_DIR}/results.jsonl"

mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_DIR"
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }
[[ -f "$RESULTS_PATH" ]] || { echo "Existing results not found: $RESULTS_PATH" >&2; exit 2; }
command -v uv >/dev/null || { echo "Required command not found: uv" >&2; exit 2; }

# Prevent two rejudge jobs from updating the same checkpoint file concurrently.
exec 9>"${OUTPUT_DIR}/.rejudge.lock"
flock -n 9 || { echo "Another premium rejudge is already running" >&2; exit 3; }

echo "Resuming unresolved premium HLE Physics judgments in: $OUTPUT_DIR"
echo "Completed generations and successful judge verdicts will be reused"

cd "$REPO_ROOT"
uv run theo-benchmark \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --dataset hle-physics-text \
  --full-dataset \
  --expect-samples 202 \
  --use-model-output-limits \
  --temperature 0.0 \
  --concurrency 4 \
  --bootstrap-samples "${BENCHMARK_BOOTSTRAP_SAMPLES:-10000}" \
  --judge-concurrency "${JUDGE_CONCURRENCY:-8}" \
  --judge-batch-size 1 \
  --judge-max-tokens "${JUDGE_MAX_TOKENS:-16384}" \
  --judge-attempts "${JUDGE_ATTEMPTS:-3}" \
  --judge-checkpoint-size "${JUDGE_CHECKPOINT_SIZE:-25}"
