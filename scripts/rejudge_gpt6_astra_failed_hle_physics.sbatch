#!/usr/bin/env bash
#SBATCH --job-name=hle-gpt6-astra-rejudge
#SBATCH --partition=train
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Rejudge only unresolved verdicts in the existing direct GPT-6 Astra run.
# Successful verdicts are preserved; failed verdicts are checkpointed one at a time.

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_PATH="${RESULTS_PATH:-${REPO_ROOT}/outputs/gpt-6-benchmark/results.jsonl}"
SUMMARY_PATH="${SUMMARY_PATH:-${REPO_ROOT}/outputs/gpt-6-benchmark/summary.json}"
JUDGE_MODEL="${JUDGE_MODEL:-~deepseek/deepseek-flash-latest}"

if [[ -z "${OPENROUTER_API_KEY:-}" && -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

mkdir -p "${REPO_ROOT}/slurm_logs"
[[ -f "$RESULTS_PATH" ]] || { echo "Results not found: $RESULTS_PATH" >&2; exit 2; }
command -v uv >/dev/null || { echo "Required command not found: uv" >&2; exit 2; }
[[ -n "${OPENROUTER_API_KEY:-}" ]] || {
  echo "OPENROUTER_API_KEY must be exported or defined in ${REPO_ROOT}/.env" >&2
  exit 2
}

exec 9>"${RESULTS_PATH}.rejudge.lock"
flock -n 9 || { echo "Another rejudge is already updating $RESULTS_PATH" >&2; exit 3; }

echo "Rejudging unresolved GPT-6 Astra verdicts in $RESULTS_PATH"
echo "Judge: $JUDGE_MODEL"

cd "$REPO_ROOT"
uv run python scripts/judge_megascience_results.py \
  "$RESULTS_PATH" \
  --summary "$SUMMARY_PATH" \
  --base-url https://openrouter.ai/api/v1 \
  --api-key "$OPENROUTER_API_KEY" \
  --model "$JUDGE_MODEL" \
  --fallback-base-url https://openrouter.ai/api/v1 \
  --fallback-api-key "$OPENROUTER_API_KEY" \
  --fallback-model "$JUDGE_MODEL" \
  --concurrency "${JUDGE_CONCURRENCY:-4}" \
  --batch-size 1 \
  --max-tokens "${JUDGE_MAX_TOKENS:-16384}" \
  --fallback-max-tokens "${JUDGE_MAX_TOKENS:-16384}" \
  --attempts "${JUDGE_ATTEMPTS:-5}" \
  --checkpoint-size 1
