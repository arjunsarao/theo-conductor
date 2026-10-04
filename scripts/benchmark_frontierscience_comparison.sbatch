#!/usr/bin/env bash
#SBATCH --job-name=frontierscience-physics-m4
#SBATCH --partition=train
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Usage: sbatch scripts/benchmark_frontierscience_comparison.sbatch ABSOLUTE_COMPARISON_ROOT
set -Eeuo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/scripts/lib/evaluation_env.sh"
COMPARISON_ROOT="$(realpath "${1:?Pass the prepared comparison directory}")"
[[ -f "$COMPARISON_ROOT/comparison-manifest.json" ]] || exit 2
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo 'OPENROUTER_API_KEY is required' >&2; exit 2; }
export PYTHONPATH="$COMPARISON_ROOT/src"
cd "$COMPARISON_ROOT"

"$PYTHON_BIN" -m theo_conductor.evaluate workflows \
  --input "$COMPARISON_ROOT/questions.jsonl" --config "$COMPARISON_ROOT/conductor.yaml" \
  --prompt-file "$COMPARISON_ROOT/conductor-prompt-refined.md" --planner-model gpt-6.1-sol \
  --samples 4 --concurrency 4 --attempts 3 --plan-temperature 0.1 --temperature 0.2 \
  --plan-max-tokens 8192 --judge-model gemini-3.7-flash \
  --output-dir "$COMPARISON_ROOT/conductor" > "$COMPARISON_ROOT/conductor.log" 2>&1 &
conductor_pid=$!
"$PYTHON_BIN" -m theo_conductor.evaluate model \
  --input "$COMPARISON_ROOT/questions.jsonl" --config "$COMPARISON_ROOT/fugu.yaml" \
  --model fugu-ultra-v2 --samples 4 --concurrency 4 --attempts 3 --temperature 0.2 \
  --judge-model gemini-3.7-flash \
  --output-dir "$COMPARISON_ROOT/fugu" > "$COMPARISON_ROOT/fugu.log" 2>&1 &
fugu_pid=$!
trap 'kill "$conductor_pid" "$fugu_pid" 2>/dev/null || true' TERM INT
exit_status=0
wait "$conductor_pid" || exit_status=1
wait "$fugu_pid" || exit_status=1
"$PYTHON_BIN" "$COMPARISON_ROOT/comparison.py" summarize --output-root "$COMPARISON_ROOT"
exit "$exit_status"
