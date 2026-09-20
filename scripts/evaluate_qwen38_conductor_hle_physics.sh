#!/usr/bin/env bash

# Submit an end-to-end evaluation of the Qwen3.8-27B conductor on the canonical
# 202 text-only HLE Physics questions. The GPU planning job is followed by a
# dependent CPU job that executes and judges the generated workflows.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/worker_pool_frontier.yaml}"
CONDUCTOR_SOURCE_MODEL="${CONDUCTOR_SOURCE_MODEL:-Qwen/Qwen3.8-27B}"

command -v sbatch >/dev/null || { echo "Required command not found: sbatch" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || {
  echo "OPENROUTER_API_KEY must be exported before submitting the evaluation" >&2
  exit 2
}

mkdir -p "${REPO_ROOT}/slurm_logs"
cd "$REPO_ROOT"

plan_export="ALL,DATASET=hle-physics-text,DATASET_SAMPLES=202,MODEL_CONFIG=${CONFIG},CONDUCTOR_SOURCE_MODEL=${CONDUCTOR_SOURCE_MODEL}"
plan_job_id="$(sbatch --parsable \
  --job-name=qwen38-hle-plan \
  --export="$plan_export" \
  "${REPO_ROOT}/scripts/pregenerate_workflows.sbatch")"
plan_job_id="${plan_job_id%%;*}"

plan_dir="${REPO_ROOT}/outputs/hle-plans-${plan_job_id}"
result_dir="${BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/outputs/hle-physics-qwen38-conductor-${plan_job_id}}"
benchmark_export="ALL,MODEL_CONFIG=${CONFIG},BENCHMARK_PLANS=${plan_dir},BENCHMARK_OUTPUT_DIR=${result_dir},BENCHMARK_SHARD_SIZE=202,BENCHMARK_BATCH_WORKERS=1,WORKER_BATCH_SIZE=202"

benchmark_job_id="$(sbatch --parsable \
  --job-name=qwen38-hle-eval \
  --dependency="afterok:${plan_job_id}" \
  --export="$benchmark_export" \
  "${REPO_ROOT}/scripts/workflow_benchmark.sbatch")"
benchmark_job_id="${benchmark_job_id%%;*}"

echo "Submitted Qwen3.8-27B planning job: ${plan_job_id}"
echo "Submitted dependent workflow evaluation job: ${benchmark_job_id}"
echo "Plans: ${plan_dir}"
echo "Results: ${result_dir}"
