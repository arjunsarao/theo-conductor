#!/usr/bin/env bash
#SBATCH --job-name=hle-diamond-physics-tags
#SBATCH --partition=train
#SBATCH --time=30:00:00
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
OUTPUT_DIR="${CLASSIFICATION_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-diamond-physics-classifications-gpt-5.6-sol}"
INPUT_JSONL="${CLASSIFICATION_INPUT:-${REPO_ROOT}/outputs/hle-diamond-physics/questions.jsonl}"
PROMPT="${CLASSIFICATION_PROMPT:-${REPO_ROOT}/classification_prompt.txt}"
SCHEMA="${CLASSIFICATION_SCHEMA:-${REPO_ROOT}/configs/hle_physics_classification_schema.json}"

mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_DIR"
[[ -f "$INPUT_JSONL" ]] || { echo "Classification input not found: $INPUT_JSONL" >&2; exit 2; }
[[ -f "$PROMPT" ]] || { echo "Classification prompt not found: $PROMPT" >&2; exit 2; }
[[ -f "$SCHEMA" ]] || { echo "Classification schema not found: $SCHEMA" >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Required command not executable: $PYTHON_BIN" >&2; exit 2; }

if [[ -z "${OPENROUTER_API_KEY:-}" && -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY must be exported or defined in ${REPO_ROOT}/.env" >&2; exit 2; }

exec 9>"${OUTPUT_DIR}/.classification.lock"
flock -n 9 || { echo "Another classification job is already using $OUTPUT_DIR" >&2; exit 3; }

echo "Classifying 61 Physics-tagged HLE-Diamond questions with GPT-5.6 Sol via OpenRouter"
echo "Results: ${OUTPUT_DIR}/classifications.jsonl"

cd "$REPO_ROOT"
"$PYTHON_BIN" scripts/classify_hle_physics.py \
  --input-jsonl "$INPUT_JSONL" \
  --prompt "$PROMPT" \
  --schema "$SCHEMA" \
  --output-dir "$OUTPUT_DIR" \
  --model "${CLASSIFICATION_MODEL:-openai/gpt-5.6-sol}" \
  --base-url https://openrouter.ai/api/v1 \
  --reasoning-effort "${CLASSIFICATION_REASONING_EFFORT:-low}" \
  --concurrency "${CLASSIFICATION_CONCURRENCY:-4}" \
  --attempts "${CLASSIFICATION_ATTEMPTS:-3}" \
  --timeout-seconds "${CLASSIFICATION_TIMEOUT_SECONDS:-1800}" \
  --expect-samples 61
