#!/usr/bin/env bash
#SBATCH --partition=train
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -Eeuo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?}"
set -a
source "${REPO_ROOT}/.env"
set +a
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:?}"
mkdir -p "$OUTPUT_DIR"
exec 9>"${OUTPUT_DIR}/.run.lock"
flock -n 9 || exit 3
cd "$REPO_ROOT"
"${REPO_ROOT}/.venv/bin/python" scripts/benchmark_hle_diamond_anthropic_batch.py \
  --input "${REPO_ROOT}/outputs/hle-diamond-physics/questions.jsonl" \
  --output-dir "$OUTPUT_DIR" --max-tokens 128000
