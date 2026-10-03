#!/usr/bin/env bash
# Shared runtime setup; SLURM resource directives remain in each entry point.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 2; }
if [[ -z "${OPENROUTER_API_KEY:-}" && -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi
mkdir -p "${REPO_ROOT}/slurm_logs"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Run independent legacy direct benchmarks and aggregate their verdicts.
# Arguments: output root, first index, count, model label, then benchmark command.
run_direct_repeats() {
  local output_root="$1" first_index="$2" count="$3" model_label="$4"
  shift 4
  [[ "$first_index" =~ ^[1-9][0-9]*$ && "$count" =~ ^[1-9][0-9]*$ ]] || {
    echo "Repeat indices/count must be positive integers" >&2; return 2;
  }
  mkdir -p "$output_root"
  local lock_fd
  exec {lock_fd}>"${output_root}/.runs.lock"
  flock -n "$lock_fd" || { echo "Another repeat job is using $output_root" >&2; return 3; }
  local run_index run_dir
  local results_args=()
  for ((run_index=first_index; run_index<first_index+count; run_index++)); do
    run_dir="${output_root}/run-${run_index}"
    "$@" --output-dir "$run_dir"
    results_args+=(--results "${run_dir}/results.jsonl")
  done
  "$PYTHON_BIN" scripts/summarize_hle_diamond_at_k.py "${results_args[@]}" \
    --output "${output_root}/aggregate-summary.json" --expect-samples 61 --model "$model_label"
  exec {lock_fd}>&-
}
