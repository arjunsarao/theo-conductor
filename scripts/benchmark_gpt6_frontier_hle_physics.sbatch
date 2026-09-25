#!/usr/bin/env bash
#SBATCH --job-name=hle-gpt6-frontier
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

# Have GPT-6 Astra generate fresh plans for all 202 text-only HLE Physics
# questions, then execute them with the frontier worker pool. DeepSeek Flash's
# moving OpenRouter alias judges each workflow before its result is checkpointed.

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/worker_pool_frontier.yaml}"
RUN_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-physics-gpt6-frontier-${RUN_ID}}"
PLAN_DIR="${PLAN_OUTPUT_DIR:-${OUTPUT_DIR}/plans}"

mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_DIR"
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }
command -v uv >/dev/null || { echo "Required command not found: uv" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || {
  echo "OPENROUTER_API_KEY must be exported when this job is submitted" >&2
  exit 2
}

plan_args=(
  --prompt-file "${CONDUCTOR_PROMPT_FILE:-${REPO_ROOT}/CONDUCTOR_PROMPT.md}"
  --output-dir "$PLAN_DIR"
  --planner-model-id gpt-6-astra
  --sample-count 202
  --expect-samples 202
  --concurrency "${PLAN_CONCURRENCY:-4}"
  --temperature "${PLAN_TEMPERATURE:-0.1}"
  --max-tokens "${PLAN_MAX_TOKENS:-8192}"
  --attempts "${PLAN_ATTEMPTS:-3}"
  --retry-delay-seconds "${PLAN_RETRY_DELAY_SECONDS:-120}"
  --retry-jitter-seconds "${PLAN_RETRY_JITTER_SECONDS:-30}"
  --request-interval-seconds "${PLAN_REQUEST_INTERVAL_SECONDS:-1}"
  --request-jitter-seconds "${PLAN_REQUEST_JITTER_SECONDS:-0.5}"
)

workflow_args=(
  --plans "$PLAN_DIR"
  --config "$CONFIG"
  --output-dir "$OUTPUT_DIR"
  --max-samples 202
  --concurrency "${BENCHMARK_CONCURRENCY:-4}"
  --use-model-output-limits
  --worker-temperature "${WORKER_TEMPERATURE:-0.2}"
  --judge-base-url https://openrouter.ai/api/v1
  --judge-api-key "$OPENROUTER_API_KEY"
  --judge-model '~deepseek/deepseek-flash-latest'
  --judge-immediately
  --judge-concurrency "${JUDGE_CONCURRENCY:-8}"
  --judge-max-tokens "${JUDGE_MAX_TOKENS:-8192}"
  --judge-attempts "${JUDGE_ATTEMPTS:-3}"
  --bootstrap-samples "${BENCHMARK_BOOTSTRAP_SAMPLES:-10000}"
)

[[ "${BENCHMARK_JUDGE:-1}" == "1" ]] || workflow_args+=(--no-judge)
[[ "${BENCHMARK_RETRY_FAILURES:-0}" == "1" ]] && workflow_args+=(--retry-failures)

echo "Generating fresh GPT-6 Astra conductor plans for all 202 text-only HLE Physics questions"
echo "Plan output: $PLAN_DIR"
echo "Workers: $CONFIG"
echo "Judge: ~deepseek/deepseek-flash-latest via OpenRouter, per completed workflow"
echo "Results: $OUTPUT_DIR"

cd "$REPO_ROOT"
uv run python scripts/generate_frontier_plan_sample.py "${plan_args[@]}"
uv run theo-workflow-benchmark "${workflow_args[@]}"
