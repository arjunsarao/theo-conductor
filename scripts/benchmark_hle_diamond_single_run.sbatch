#!/usr/bin/env bash
#SBATCH --partition=train
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/scripts/lib/evaluation_env.sh"
INPUT="${BENCHMARK_INPUT:-${REPO_ROOT}/outputs/hle-diamond-physics/questions.jsonl}"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:?BENCHMARK_OUTPUT_DIR is required}"
SOLVER_MODEL="${SOLVER_MODEL:?SOLVER_MODEL is required}"
SOLVER_NAME="${SOLVER_NAME:?SOLVER_NAME is required}"


mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_DIR"
[[ -f "$INPUT" ]] || { echo "Input not found: $INPUT" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is required" >&2; exit 2; }

exec 9>"${OUTPUT_DIR}/.run.lock"
flock -n 9 || { echo "Another job is using $OUTPUT_DIR" >&2; exit 3; }

cd "$REPO_ROOT"
"$PYTHON_BIN" scripts/benchmark_hle_diamond_astra_gemini_batch.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --expect-samples 61 \
  --solver-model "$SOLVER_MODEL" \
  --solver-name "$SOLVER_NAME" \
  --solver-standard \
  --solver-timeout-seconds "${SOLVER_TIMEOUT_SECONDS:-7200}" \
  --solver-concurrency "${BENCHMARK_CONCURRENCY:-4}" \
  --max-tokens "${SOLVER_MAX_TOKENS:-32768}" \
  --judge-mode sync \
  --judge-concurrency "${JUDGE_CONCURRENCY:-8}" \
  --judge-max-tokens "${JUDGE_MAX_TOKENS:-8192}" \
  --attempts "${BENCHMARK_ATTEMPTS:-3}"
