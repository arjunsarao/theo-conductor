#!/usr/bin/env bash
#SBATCH --job-name=theo-evaluate
#SBATCH --partition=train
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Usage: sbatch scripts/evaluate.sbatch workflows --input ... --config ... --output-dir ...
# Repeat this exact submission to recover interrupted/failed items.
set -Eeuo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/scripts/lib/evaluation_env.sh"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "$PYTHON_BIN" -m theo_conductor.evaluate "$@"
