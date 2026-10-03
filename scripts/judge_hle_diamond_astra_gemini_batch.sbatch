#!/usr/bin/env bash
#SBATCH --job-name=hle-diamond-gemini-judge
#SBATCH --partition=train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arjun@firstprinciples.com

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INPUT="${BENCHMARK_INPUT:-${REPO_ROOT}/outputs/hle-diamond-physics/questions.jsonl}"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-diamond-astra-gemini-batch-22621}"

if [[ -z "${OPENROUTER_API_KEY:-}" && -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

mkdir -p "${REPO_ROOT}/slurm_logs"
[[ -f "$INPUT" ]] || { echo "Input not found: $INPUT" >&2; exit 2; }
[[ -f "${OUTPUT_DIR}/results.jsonl" ]] || { echo "Results not found: ${OUTPUT_DIR}/results.jsonl" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is required" >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 2; }

exec 9>"${OUTPUT_DIR}/.judge.lock"
flock -n 9 || { echo "Another judge job is using $OUTPUT_DIR" >&2; exit 3; }

cd "$REPO_ROOT"
"$PYTHON_BIN" scripts/benchmark_hle_diamond_astra_gemini_batch.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --expect-samples 61 \
  --solver-model "${SOLVER_MODEL:-openai/gpt-6-astra}" \
  --solver-name "${SOLVER_NAME:-GPT-6 Astra}" \
  --judge-only \
  --judge-mode "${JUDGE_MODE:-batch}" \
  --judge-concurrency "${JUDGE_CONCURRENCY:-8}" \
  --judge-max-tokens "${JUDGE_MAX_TOKENS:-8192}" \
  --attempts "${JUDGE_ATTEMPTS:-3}" \
  --poll-seconds "${BATCH_POLL_SECONDS:-30}"
