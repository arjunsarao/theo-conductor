#!/usr/bin/env bash
#SBATCH --job-name=hle-router-direct
#SBATCH --partition=train
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Run one configured remote router as a direct solver over all 202 text-only
# HLE Physics questions. Judging is deliberately disabled for this generation
# stage; results.jsonl checkpoints each completion as it arrives.

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${MODEL_CONFIG:?MODEL_CONFIG must name a single-model YAML config}"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:?BENCHMARK_OUTPUT_DIR must be set}"
PYTHON="${REPO_ROOT}/.venv/bin/python"

if [[ -z "${OPENROUTER_API_KEY:-}" && -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_DIR"
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Python environment not found: $PYTHON" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || {
  echo "OPENROUTER_API_KEY must be exported or defined in ${REPO_ROOT}/.env" >&2
  exit 2
}

echo "Generating completions for all 202 text-only HLE Physics questions"
echo "Model config: $CONFIG"
echo "Judging: disabled"
echo "Results: $OUTPUT_DIR"

cd "$REPO_ROOT"
"$PYTHON" -m theo_conductor.benchmark \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --dataset hle-physics-text \
  --full-dataset \
  --expect-samples 202 \
  --use-model-output-limits \
  --temperature "${BENCHMARK_TEMPERATURE:-0.0}" \
  --concurrency "${BENCHMARK_CONCURRENCY:-4}" \
  --bootstrap-samples "${BENCHMARK_BOOTSTRAP_SAMPLES:-10000}" \
  --no-judge
