#!/usr/bin/env bash
#SBATCH --job-name=hle-frontier-plans
#SBATCH --partition=train
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=arjun@firstprinciples.com

# Generate plans for the same deterministic 15-question sample from the
# canonical 202-question text-only HLE Physics set with three frontier models.
# This is API inference only and does not request a GPU.

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="${PLAN_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-frontier-plan-sample-${RUN_ID}}"
PROMPT_FILE="${CONDUCTOR_PROMPT_FILE:-${REPO_ROOT}/CONDUCTOR_PROMPT.md}"
PYTHON_BIN="${THEO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
SAMPLE_COUNT="${PLAN_SAMPLE_COUNT:-15}"
[[ "$SAMPLE_COUNT" =~ ^[1-9][0-9]*$ ]] || {
  echo "PLAN_SAMPLE_COUNT must be a positive integer" >&2
  exit 2
}

mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_DIR"
[[ -x "$PYTHON_BIN" ]] || { echo "Python environment not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$PROMPT_FILE" ]] || { echo "Conductor prompt not found: $PROMPT_FILE" >&2; exit 2; }
# The Python generator loads REPO_ROOT/.env and also honors an exported key.

args=(
  --prompt-file "$PROMPT_FILE"
  --output-dir "$OUTPUT_DIR"
  --expect-samples 202
  --sample-count "$SAMPLE_COUNT"
  --seed "${PLAN_SAMPLE_SEED:-42}"
  --concurrency "${PLAN_CONCURRENCY:-6}"
  --temperature "${PLAN_TEMPERATURE:-0.1}"
  --max-tokens "${PLAN_MAX_TOKENS:-8192}"
  --attempts "${PLAN_ATTEMPTS:-3}"
  --retry-delay-seconds "${PLAN_RETRY_DELAY_SECONDS:-120}"
  --retry-jitter-seconds "${PLAN_RETRY_JITTER_SECONDS:-30}"
  --request-interval-seconds "${PLAN_REQUEST_INTERVAL_SECONDS:-1}"
  --request-jitter-seconds "${PLAN_REQUEST_JITTER_SECONDS:-0.5}"
)

echo "Generating $((SAMPLE_COUNT * 3)) matched plans (${SAMPLE_COUNT} questions x 3 models)"
echo "Models: GPT-6 Astra, Claude Fable 5.1, Gemini 3.8 Flash"
echo "Output directory: $OUTPUT_DIR"

cd "$REPO_ROOT"
PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "$PYTHON_BIN" scripts/generate_frontier_plan_sample.py "${args[@]}"
