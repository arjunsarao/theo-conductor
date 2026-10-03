#!/usr/bin/env bash
#SBATCH --job-name=hle-diamond-jev-m4
#SBATCH --partition=train
#SBATCH --time=48:00:00
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
source "${REPO_ROOT}/scripts/lib/evaluation_env.sh"
INPUT="${BENCHMARK_INPUT:-${REPO_ROOT}/outputs/hle-diamond-physics/questions.jsonl}"
RUN_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-${REPO_ROOT}/outputs/hle-diamond-jev-router-mean4-${RUN_ID}}"


mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_ROOT"
[[ -f "$INPUT" ]] || { echo "Input not found: $INPUT" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is required" >&2; exit 2; }

run_direct_repeats "$OUTPUT_ROOT" 1 "${BENCHMARK_SAMPLES:-4}" typesafe/jev-router \
  "$PYTHON_BIN" scripts/benchmark_hle_diamond_astra_gemini_batch.py \
    --input "$INPUT" \
    --expect-samples 61 \
    --solver-model typesafe/jev-router \
    --solver-name JevRouter \
    --solver-standard \
    --solver-timeout-seconds "${SOLVER_TIMEOUT_SECONDS:-7200}" \
    --solver-concurrency "${BENCHMARK_CONCURRENCY:-4}" \
    --max-tokens "${JEV_MAX_TOKENS:-32768}" \
    --judge-max-tokens "${JUDGE_MAX_TOKENS:-8192}" \
    --attempts "${BENCHMARK_ATTEMPTS:-3}" \
    --poll-seconds "${BATCH_POLL_SECONDS:-30}"
