#!/usr/bin/env bash
# Populate a persistent, vLLM-ready model directory for the local GRPO job.
#
# Run this once from an interactive srun allocation (no GPU is required):
#   VLLM_ENV=.venv scripts/download_grpo_models.sh
#
# The GRPO sbatch script uses the same THEO_MODEL_ROOT by default, so later
# jobs reuse these files instead of staging them into per-job local scratch.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/worker_pool_small.yaml}"
MODEL_ROOT="${THEO_MODEL_ROOT:-/mnt/data/home/arjun/.cache/theo-conductor/models}"
INCLUDE_CONDUCTOR="${INCLUDE_CONDUCTOR:-1}"

if [[ -n "${VLLM_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${VLLM_ENV}/bin/activate"
fi

command -v python >/dev/null || { echo "Required command not found: python" >&2; exit 2; }
command -v hf >/dev/null || { echo "Required command not found: hf (activate the environment that provides huggingface_hub)" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }
mkdir -p "$MODEL_ROOT"

# Keep Hugging Face metadata/cache in a writable persistent location too.
export HF_HOME="${THEO_HF_HOME:-/mnt/data/home/arjun/.cache/huggingface}"
export HF_HUB_CACHE="${THEO_HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${THEO_TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_XET_CACHE="${THEO_HF_XET_CACHE:-${HF_HOME}/xet}"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_XET_CACHE"

mapfile -t MODELS < <(python - "$CONFIG" <<'PY'
import sys
import yaml

config = sys.argv[1]
with open(config, encoding="utf-8") as handle:
    data = yaml.safe_load(handle)
models = [
    item["deployment"].get("source_model", item["client"]["model"])
    for item in data["models"]
    if item.get("deployment", {}).get("mode") == "local"
]
print("\n".join(dict.fromkeys(models)))
PY
)

CONDUCTOR_MODEL="${CONDUCTOR_MODEL:-$(python - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}
print(data.get("conductor_model", ""))
PY
)}"

download() {
  local repo_id=$1
  local destination="${MODEL_ROOT}/$(tr '/' '_' <<< "$repo_id")"
  if [[ -f "$destination/.download-complete" ]]; then
    echo "Already available: $repo_id -> $destination"
    return
  fi
  mkdir -p "$destination"
  echo "Downloading $repo_id -> $destination"
  # hf download is resumable; if interrupted, re-running this script completes it.
  hf download "$repo_id" --local-dir "$destination"
  touch "$destination/.download-complete"
  echo "Ready: $repo_id"
}

for model in "${MODELS[@]}"; do
  download "$model"
done

if [[ "$INCLUDE_CONDUCTOR" == "1" ]]; then
  [[ -n "$CONDUCTOR_MODEL" ]] || {
    echo "Config has no conductor_model; set CONDUCTOR_MODEL or use INCLUDE_CONDUCTOR=0" >&2
    exit 2
  }
  # The trainer loads this model by repository ID rather than LOCAL_MODEL_ROOT,
  # so warm the shared Hugging Face cache instead of making a second full copy.
  echo "Downloading conductor cache: $CONDUCTOR_MODEL"
  hf download "$CONDUCTOR_MODEL"
fi

echo "Persistent model root ready: $MODEL_ROOT"
echo "Use this same path explicitly if needed: THEO_MODEL_ROOT=$MODEL_ROOT sbatch scripts/worker_pool.sbatch"
