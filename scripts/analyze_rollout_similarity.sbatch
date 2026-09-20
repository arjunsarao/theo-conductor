#!/usr/bin/env bash
#SBATCH --job-name=rollout-sim-21510
#SBATCH --partition=train
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arjun@firstprinciples.com

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INPUT_DIR="${SIMILARITY_INPUT_DIR:-${REPO_ROOT}/outputs/hle-plans-21510}"
REPORT_DIR="${SIMILARITY_REPORT_DIR:-${REPO_ROOT}/reports}"
TRIALS="${SIMILARITY_TRIALS:-200}"

mkdir -p "${REPO_ROOT}/slurm_logs" "$REPORT_DIR" "${INPUT_DIR}/similarity-analysis"
cd "$REPO_ROOT"

export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/analyze_rollout_similarity.py "$INPUT_DIR" \
  --report "${REPORT_DIR}/rollout_similarity_21510.md" \
  --per-rollout-csv "${REPORT_DIR}/rollout_similarity_21510_per_rollout.csv" \
  --pairwise-csv "${REPORT_DIR}/rollout_similarity_21510_pairwise.csv" \
  --metrics-json "${REPORT_DIR}/rollout_similarity_21510_metrics.json" \
  --embedding-cache "${INPUT_DIR}/similarity-analysis/minilm_embeddings.npz" \
  --batch-size 256 \
  --trials "$TRIALS" \
  --seed 21510
