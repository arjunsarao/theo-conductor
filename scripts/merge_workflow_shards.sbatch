#!/usr/bin/env bash
#SBATCH --job-name=hle-plan-merge
#SBATCH --partition=train
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PLAN_RUN_ID="${PLAN_JOB_ID:?Set PLAN_JOB_ID to the pregeneration array job ID}"
OUTPUT_DIR="${PLAN_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-plans-${PLAN_RUN_ID}}"
PLAN_SHARDS="${PLAN_SHARDS:-10}"

PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m theo_conductor.plan merge "$OUTPUT_DIR" --expected-shards "$PLAN_SHARDS"

echo "Merged workflows: ${OUTPUT_DIR}/plans.jsonl"
