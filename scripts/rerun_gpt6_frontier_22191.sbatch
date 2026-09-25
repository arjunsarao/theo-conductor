#!/usr/bin/env bash
#SBATCH --job-name=hle-gpt6-retry
#SBATCH --partition=train
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arjun@firstprinciples.com

# Recover job 22191 without regenerating plans or repeating its 24 completed
# workflows. First repair their judge verdicts, then rerun only the 178 workflow
# failures with the corrected frontier registry and immediate DeepSeek judging.

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_DIR="${SOURCE_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-physics-gpt6-frontier-22191}"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/worker_pool_frontier.yaml}"
RUN_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-physics-gpt6-frontier-22191-retry-${RUN_ID}}"
FAILED_PLANS="${OUTPUT_DIR}/plans.jsonl"
JUDGE_MODEL='~deepseek/deepseek-flash-latest'

mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_DIR"
[[ -f "${SOURCE_DIR}/plans/plans.jsonl" ]] || { echo "Source plans not found" >&2; exit 2; }
[[ -f "${SOURCE_DIR}/results.jsonl" ]] || { echo "Source results not found" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }
command -v uv >/dev/null || { echo "Required command not found: uv" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is required" >&2; exit 2; }

cd "$REPO_ROOT"

echo "Repairing judge verdicts for the 24 workflows completed by job 22191"
uv run python scripts/judge_megascience_results.py \
  "${SOURCE_DIR}/results.jsonl" \
  --summary "${SOURCE_DIR}/summary.json" \
  --base-url https://openrouter.ai/api/v1 \
  --api-key "$OPENROUTER_API_KEY" \
  --model "$JUDGE_MODEL" \
  --fallback-base-url https://openrouter.ai/api/v1 \
  --fallback-api-key "$OPENROUTER_API_KEY" \
  --fallback-model "$JUDGE_MODEL" \
  --concurrency "${JUDGE_CONCURRENCY:-8}" \
  --batch-size 1 \
  --max-tokens "${JUDGE_MAX_TOKENS:-8192}" \
  --fallback-max-tokens "${JUDGE_MAX_TOKENS:-8192}" \
  --attempts "${JUDGE_ATTEMPTS:-3}" \
  --checkpoint-size 1 \
  --force

uv run python scripts/select_failed_workflows.py \
  --results "${SOURCE_DIR}/results.jsonl" \
  --plans "${SOURCE_DIR}/plans/plans.jsonl" \
  --output "$FAILED_PLANS" \
  --expect 178

echo "Rerunning 178 failed workflows with $CONFIG"
uv run theo-workflow-benchmark \
  --plans "$FAILED_PLANS" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --max-samples 178 \
  --concurrency "${BENCHMARK_CONCURRENCY:-4}" \
  --use-model-output-limits \
  --worker-temperature "${WORKER_TEMPERATURE:-0.2}" \
  --judge-base-url https://openrouter.ai/api/v1 \
  --judge-api-key "$OPENROUTER_API_KEY" \
  --judge-model "$JUDGE_MODEL" \
  --judge-immediately \
  --judge-concurrency "${JUDGE_CONCURRENCY:-8}" \
  --judge-max-tokens "${JUDGE_MAX_TOKENS:-8192}" \
  --judge-attempts "${JUDGE_ATTEMPTS:-3}" \
  --bootstrap-samples "${BENCHMARK_BOOTSTRAP_SAMPLES:-10000}"
