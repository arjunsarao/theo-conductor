#!/usr/bin/env bash
#SBATCH --job-name=hle-flash-capped-rerun
#SBATCH --partition=train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_RESULTS="${SOURCE_RESULTS:-${REPO_ROOT}/outputs/hle-physics-flash-eval-21504/results.jsonl}"
SOURCE_PLANS="${SOURCE_PLANS:-${REPO_ROOT}/outputs/hle-plans-21504/plans.jsonl}"
SOURCE_CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/worker_pool_flash.yaml}"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-physics-flash-capped-rerun-${SLURM_JOB_ID}}"
SELECTED_PLANS="${OUTPUT_DIR}/selected-plans.jsonl"
CONFIG="${OUTPUT_DIR}/worker-pool-double-output.yaml"

mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_DIR"
cd "$REPO_ROOT"

uv run python scripts/select_capped_workflows.py \
  --results "$SOURCE_RESULTS" \
  --plans "$SOURCE_PLANS" \
  --output "$SELECTED_PLANS" \
  --config "$SOURCE_CONFIG" \
  --config-output "$CONFIG"

TARGET_COUNT="$(wc -l < "$SELECTED_PLANS")"
echo "Rerunning ${TARGET_COUNT} unsuccessful capped or worker-error workflows"
echo "Capped models use doubled output-token allowances; workflow prompts are unchanged"
echo "Results will be written separately to: ${OUTPUT_DIR}"

uv run theo-workflow-benchmark \
  --plans "$SELECTED_PLANS" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --max-samples "$TARGET_COUNT" \
  --use-model-output-limits \
  --worker-temperature 0.2 \
  --concurrency "${BENCHMARK_CONCURRENCY:-2}" \
  --judge-concurrency "${JUDGE_CONCURRENCY:-2}" \
  --judge-batch-size 1 \
  --judge-max-tokens "${JUDGE_MAX_TOKENS:-16384}" \
  --judge-attempts "${JUDGE_ATTEMPTS:-3}" \
  --judge-checkpoint-size 1
