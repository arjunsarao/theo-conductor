#!/usr/bin/env bash
#SBATCH --job-name=hle-diamond-gpt61-frontier-m4
#SBATCH --partition=train
#SBATCH --time=72:00:00
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
source "${REPO_ROOT}/scripts/lib/evaluation_env.sh"
INPUT="${BENCHMARK_INPUT:-${REPO_ROOT}/outputs/hle-diamond-physics/questions.jsonl}"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/worker_pool_frontier.yaml}"
PROMPT="${CONDUCTOR_PROMPT_FILE:-${REPO_ROOT}/conductor-prompt-refined.md}"
RUN_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-${REPO_ROOT}/outputs/hle-diamond-gpt61-frontier-mean4-${RUN_ID}}"


mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_ROOT"
[[ -f "$INPUT" ]] || { echo "Input not found: $INPUT" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }
[[ -f "$PROMPT" ]] || { echo "Conductor prompt not found: $PROMPT" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is required" >&2; exit 2; }

cd "$REPO_ROOT"
results_args=()
for run_index in 1 2 3 4; do
  run_dir="${OUTPUT_ROOT}/run-${run_index}"
  plan_dir="${run_dir}/plans"
  seed=$((41 + run_index))
  echo "Starting independent GPT-6.1 Sol conductor rollout ${run_index}; output=${run_dir}"
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "$PYTHON_BIN" scripts/generate_frontier_plan_sample.py \
      --input-jsonl "$INPUT" \
      --prompt-file "$PROMPT" \
      --output-dir "$plan_dir" \
      --planner-model-id gpt-6.1-sol \
      --sample-count 61 \
      --expect-samples 61 \
      --seed "$seed" \
      --concurrency "${PLAN_CONCURRENCY:-4}" \
      --temperature "${PLAN_TEMPERATURE:-0.1}" \
      --max-tokens "${PLAN_MAX_TOKENS:-8192}" \
      --attempts "${PLAN_ATTEMPTS:-3}" \
      --retry-delay-seconds "${PLAN_RETRY_DELAY_SECONDS:-120}" \
      --retry-jitter-seconds "${PLAN_RETRY_JITTER_SECONDS:-30}" \
      --request-interval-seconds "${PLAN_REQUEST_INTERVAL_SECONDS:-1}" \
      --request-jitter-seconds "${PLAN_REQUEST_JITTER_SECONDS:-0.5}"

  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "$PYTHON_BIN" -m theo_conductor.workflow_benchmark \
      --plans "$plan_dir" \
      --planner-model-id gpt-6.1-sol \
      --config "$CONFIG" \
      --output-dir "$run_dir" \
      --include-multimodal \
      --max-samples 61 \
      --concurrency "${BENCHMARK_CONCURRENCY:-4}" \
      --use-model-output-limits \
      --worker-temperature "${WORKER_TEMPERATURE:-0.2}" \
      --judge-base-url https://openrouter.ai/api/v1 \
      --judge-api-key "$OPENROUTER_API_KEY" \
      --judge-model google/gemini-3.8-flash \
      --judge-immediately \
      --judge-concurrency "${JUDGE_CONCURRENCY:-8}" \
      --judge-max-tokens "${JUDGE_MAX_TOKENS:-8192}" \
      --judge-attempts "${JUDGE_ATTEMPTS:-3}" \
      --bootstrap-samples "${BENCHMARK_BOOTSTRAP_SAMPLES:-10000}"
  results_args+=(--results "${run_dir}/results.jsonl")
done

"$PYTHON_BIN" scripts/summarize_hle_diamond_at_k.py \
  "${results_args[@]}" \
  --output "${OUTPUT_ROOT}/aggregate-summary.json" \
  --expect-samples 61 \
  --model theo-conductor:gpt-6.1-sol+frontier

echo "Aggregate mean@4/pass@4: ${OUTPUT_ROOT}/aggregate-summary.json"
