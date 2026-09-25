#!/usr/bin/env bash
#SBATCH --job-name=hle-gpt6-direct-batch
#SBATCH --partition=train
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arjun@firstprinciples.com

# Ask GPT-6 Astra to solve all 202 text-only HLE Physics questions directly
# in one OpenRouter Batch API job. Once that job completes, grade each answer
# with the same DeepSeek Flash setup used for the GPT-6 conductor benchmark.

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/gpt6_astra_direct_batch.yaml}"
RUN_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-physics-gpt6-direct-batch-${RUN_ID}}"
JUDGE_MODEL='~deepseek/deepseek-flash-latest'

# Slurm normally inherits an exported key. Also support the repository's
# existing dotenv setup when this job is submitted from a fresh shell.
if [[ -z "${OPENROUTER_API_KEY:-}" && -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_DIR"
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }
command -v uv >/dev/null || { echo "Required command not found: uv" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || {
  echo "OPENROUTER_API_KEY must be exported or defined in ${REPO_ROOT}/.env" >&2
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
  --batch-model gpt-6-astra-direct
  --worker-batch-size 202
  --bootstrap-samples "${BENCHMARK_BOOTSTRAP_SAMPLES:-10000}"
  --judge-base-url https://openrouter.ai/api/v1
  --judge-api-key "$OPENROUTER_API_KEY"
  --judge-model "$JUDGE_MODEL"
  --fallback-judge-base-url https://openrouter.ai/api/v1
  --fallback-judge-api-key "$OPENROUTER_API_KEY"
  --fallback-judge-model "$JUDGE_MODEL"
  --judge-concurrency "${JUDGE_CONCURRENCY:-8}"
  --judge-batch-size 1
  --judge-max-tokens "${JUDGE_MAX_TOKENS:-8192}"
  --judge-attempts "${JUDGE_ATTEMPTS:-3}"
  --judge-checkpoint-size 1
)

[[ "${BENCHMARK_JUDGE:-1}" == "1" ]] || args+=(--no-judge)

echo "Submitting all 202 text-only HLE Physics questions in one GPT-6 Astra batch"
echo "Solver: openai/gpt-6-astra through OpenRouter's native Batch API"
echo "Judge: $JUDGE_MODEL via OpenRouter, one answer per judge request"
echo "Results: $OUTPUT_DIR"

cd "$REPO_ROOT"
uv run theo-benchmark "${args[@]}"
