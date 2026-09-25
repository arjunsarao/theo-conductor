#!/usr/bin/env bash
#SBATCH --job-name=hle-gpt6-capped-retry
#SBATCH --partition=train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arjun@firstprinciples.com

# Regenerate and immediately rejudge the incorrect GPT-6-conductor workflows
# whose original worker execution stopped at its output-token limit.

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_DIR="${SOURCE_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-physics-gpt6-frontier}"
SOURCE_RESULTS="${SOURCE_DIR}/results.jsonl"
SOURCE_PLANS="${SOURCE_DIR}/plans/plans.jsonl"
SOURCE_CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/worker_pool_frontier.yaml}"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-physics-gpt6-frontier-capped-rerun-${SLURM_JOB_ID}}"
SELECTED_PLANS="${OUTPUT_DIR}/selected-plans.jsonl"
CONFIG="${OUTPUT_DIR}/worker-pool-double-output.yaml"
EXPECTED_COUNT="${EXPECTED_CAPPED_WORKFLOWS:-8}"
JUDGE_MODEL='~deepseek/deepseek-flash-latest'

mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_DIR"
[[ -f "$SOURCE_RESULTS" ]] || { echo "Source results not found: $SOURCE_RESULTS" >&2; exit 2; }
[[ -f "$SOURCE_PLANS" ]] || { echo "Source plans not found: $SOURCE_PLANS" >&2; exit 2; }
[[ -f "$SOURCE_CONFIG" ]] || { echo "Model config not found: $SOURCE_CONFIG" >&2; exit 2; }
command -v uv >/dev/null || { echo "Required command not found: uv" >&2; exit 2; }

cd "$REPO_ROOT"

uv run python scripts/select_capped_workflows.py \
  --results "$SOURCE_RESULTS" \
  --plans "$SOURCE_PLANS" \
  --output "$SELECTED_PLANS" \
  --config "$SOURCE_CONFIG" \
  --config-output "$CONFIG" \
  --capped-only

TARGET_COUNT="$(wc -l < "$SELECTED_PLANS")"
if [[ "$TARGET_COUNT" -ne "$EXPECTED_COUNT" ]]; then
  echo "Expected ${EXPECTED_COUNT} incorrect capped workflows, selected ${TARGET_COUNT}" >&2
  exit 2
fi

echo "Regenerating and rejudging ${TARGET_COUNT} incorrect capped workflows"
echo "DeepSeek V4 Pro max output tokens: 40960 -> 81920"
echo "Claude Fable 5.1 max output tokens: 65536 -> 131072"
echo "Results: ${OUTPUT_DIR}"

uv run theo-workflow-benchmark \
  --plans "$SELECTED_PLANS" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --max-samples "$TARGET_COUNT" \
  --use-model-output-limits \
  --worker-temperature "${WORKER_TEMPERATURE:-0.2}" \
  --concurrency "${BENCHMARK_CONCURRENCY:-2}" \
  --judge-base-url https://openrouter.ai/api/v1 \
  --judge-model "$JUDGE_MODEL" \
  --judge-immediately \
  --judge-concurrency "${JUDGE_CONCURRENCY:-2}" \
  --judge-batch-size 1 \
  --judge-max-tokens "${JUDGE_MAX_TOKENS:-32768}" \
  --judge-attempts "${JUDGE_ATTEMPTS:-3}" \
  --judge-checkpoint-size 1 \
  --bootstrap-samples "${BENCHMARK_BOOTSTRAP_SAMPLES:-10000}"
