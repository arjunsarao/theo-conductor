#!/usr/bin/env bash
#SBATCH --job-name=hle-kimi-glm-plans
#SBATCH --partition=train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arjun@firstprinciples.com

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROMPT_FILE="${CONDUCTOR_PROMPT_FILE:-${REPO_ROOT}/CONDUCTOR_PROMPT.md}"
RUN_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="${WORKFLOW_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-kimi-glm-workflows-${RUN_ID}}"

mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_DIR"
[[ -f "$PROMPT_FILE" ]] || { echo "Conductor prompt not found: $PROMPT_FILE" >&2; exit 2; }
PYTHON_BIN="${THEO_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Python environment not found or not executable: $PYTHON_BIN" >&2; exit 2; }

args=(
  --prompt-file "$PROMPT_FILE"
  --output-dir "$OUTPUT_DIR"
  --expect-samples 202
  --concurrency "${WORKFLOW_CONCURRENCY:-8}"
  --temperature "${WORKFLOW_TEMPERATURE:-0.1}"
)

echo "Generating 404 independent workflows (202 questions x Kimi/GLM)"
echo "No request batching and no max-output-token parameter are enabled"
echo "Output directory: $OUTPUT_DIR"

cd "$REPO_ROOT"
PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "$PYTHON_BIN" scripts/generate_endpoint_workflows.py "${args[@]}"
