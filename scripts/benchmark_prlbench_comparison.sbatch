#!/usr/bin/env bash
#SBATCH --job-name=prlbench-fugu-conductor-m4
#SBATCH --partition=train
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Uses OpenRouter hosted shell; the compute node only needs API access.
# Usage: sbatch scripts/benchmark_prlbench_comparison.sbatch PREPARED_DIRECTORY
set -Eeuo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/scripts/lib/evaluation_env.sh"
COMPARISON_ROOT="$(realpath "${1:?Pass the prepared comparison directory}")"
shift
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo 'OPENROUTER_API_KEY is required' >&2; exit 2; }
exec "$PYTHON_BIN" -m theo_conductor.prlbench_comparison run --output-dir "$COMPARISON_ROOT" "$@"
