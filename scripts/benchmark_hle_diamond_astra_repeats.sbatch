#!/usr/bin/env bash
#SBATCH --job-name=hle-diamond-astra-repeats
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
BASELINE_RESULTS="${BASELINE_RESULTS:-${REPO_ROOT}/outputs/hle-diamond-astra-gemini-batch-22621/results.jsonl}"
RUN_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-${REPO_ROOT}/outputs/hle-diamond-astra-repeats-${RUN_ID}}"
NUM_ADDITIONAL_RUNS="${NUM_ADDITIONAL_RUNS:-3}"
START_RUN_INDEX="${START_RUN_INDEX:-2}"


mkdir -p "${REPO_ROOT}/slurm_logs" "$OUTPUT_ROOT"
[[ -f "$INPUT" ]] || { echo "Input not found: $INPUT" >&2; exit 2; }
[[ -f "$BASELINE_RESULTS" ]] || { echo "Baseline results not found: $BASELINE_RESULTS" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is required" >&2; exit 2; }
[[ "$NUM_ADDITIONAL_RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_ADDITIONAL_RUNS must be positive" >&2; exit 2; }
[[ "$START_RUN_INDEX" =~ ^[1-9][0-9]*$ ]] || { echo "START_RUN_INDEX must be positive" >&2; exit 2; }

exec 9>"${OUTPUT_ROOT}/.runs.lock"
flock -n 9 || { echo "Another repeat job is using $OUTPUT_ROOT" >&2; exit 3; }

results_args=(--results "$BASELINE_RESULTS")
last_run=$((START_RUN_INDEX + NUM_ADDITIONAL_RUNS - 1))
for run_index in $(seq "$START_RUN_INDEX" "$last_run"); do
  run_dir="${OUTPUT_ROOT}/run-${run_index}"
  echo "Starting independent Astra run ${run_index}; output=${run_dir}"
  set +e
  "$PYTHON_BIN" scripts/benchmark_hle_diamond_astra_gemini_batch.py \
    --input "$INPUT" \
    --output-dir "$run_dir" \
    --expect-samples 61 \
    --max-tokens "${ASTRA_MAX_TOKENS:-32768}" \
    --judge-max-tokens "${JUDGE_MAX_TOKENS:-8192}" \
    --multimodal-concurrency "${MULTIMODAL_CONCURRENCY:-4}" \
    --attempts "${BENCHMARK_ATTEMPTS:-3}" \
    --poll-seconds "${BATCH_POLL_SECONDS:-30}"
  run_status=$?
  set -e
  [[ -f "${run_dir}/summary.json" ]] || { echo "Run ${run_index} did not produce a summary (exit ${run_status})" >&2; exit "$run_status"; }
  results_args+=(--results "${run_dir}/results.jsonl")
  echo "Finished independent Astra run ${run_index} (exit ${run_status})"
done

"$PYTHON_BIN" scripts/summarize_hle_diamond_at_k.py \
  "${results_args[@]}" \
  --output "${OUTPUT_ROOT}/aggregate-summary.json" \
  --expect-samples 61

echo "Aggregate results: ${OUTPUT_ROOT}/aggregate-summary.json"
