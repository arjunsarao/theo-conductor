#!/usr/bin/env bash
#SBATCH --job-name=theo-small-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

# Download configs/local_small_models.yaml to node-local storage, launch one vLLM
# endpoint per entry, and exercise the same ModelRegistry client used by the app.
#
# Submit from the repository root, for example:
#   QWEN_MEGASCIENCE_ADAPTER=org/qwen3-8b-megascience \
#     sbatch scripts/slurm_local_small_models.sbatch
#
# Site-specific examples:
#   VLLM_ENV=/path/to/venv sbatch scripts/slurm_local_small_models.sbatch
#   sbatch --gres=gpu:3 --export=ALL,MODEL_GPU_SETS=0\;1\;2,...

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/local_small_models.yaml}"
LOCAL_ROOT="${LOCAL_MODEL_ROOT:-${SLURM_TMPDIR:-/local_scratch/${USER}/${SLURM_JOB_ID:-manual}}/theo-models}"
LOG_ROOT="${LOCAL_ROOT}/logs"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# Semicolon-separated GPU sets, one per YAML entry. The 30B BF16 default uses 2 GPUs.
MODEL_GPU_SETS="${MODEL_GPU_SETS:-0;1;2,3}"
MODEL_TP_SIZES="${MODEL_TP_SIZES:-1;1;2}"

# The YAML uses a served alias for this fine-tune, so its downloadable adapter ID
# must be supplied by the submitter. Override the base too if the adapter differs.
QWEN_MEGASCIENCE_BASE="${QWEN_MEGASCIENCE_BASE:-Qwen/Qwen3-8B}"
QWEN_MEGASCIENCE_ADAPTER="${QWEN_MEGASCIENCE_ADAPTER:-}"
NEMOTRON_SOURCE="${NEMOTRON_SOURCE:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16}"

if [[ -n "${VLLM_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${VLLM_ENV}/bin/activate"
fi

for command in python hf vllm curl; do
  command -v "$command" >/dev/null || { echo "Required command not found: $command" >&2; exit 2; }
done
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }
mkdir -p "$LOCAL_ROOT" "$LOG_ROOT"

IFS=';' read -r -a GPU_SETS <<< "$MODEL_GPU_SETS"
IFS=';' read -r -a TP_SIZES <<< "$MODEL_TP_SIZES"

# Fields: port, served name, source ID from the client config.
mapfile -t MODEL_ROWS < <(python - "$CONFIG" <<'PY'
import sys
from urllib.parse import urlparse
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    models = yaml.safe_load(handle)["models"]
for model in models:
    client = model["client"]
    port = urlparse(client["base_url"]).port
    print(f'{port}\t{client["model"]}\t{client["model"]}')
PY
)

if (( ${#MODEL_ROWS[@]} != ${#GPU_SETS[@]} || ${#MODEL_ROWS[@]} != ${#TP_SIZES[@]} )); then
  echo "MODEL_GPU_SETS and MODEL_TP_SIZES must each have ${#MODEL_ROWS[@]} semicolon-separated entries" >&2
  exit 2
fi

PIDS=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if (( ${#PIDS[@]} )); then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

download() {
  local repo_id=$1 destination=$2
  if [[ ! -f "$destination/.download-complete" ]]; then
    mkdir -p "$destination"
    echo "Downloading $repo_id -> $destination"
    hf download "$repo_id" --local-dir "$destination"
    touch "$destination/.download-complete"
  fi
}

declare -a PORTS SERVED_NAMES MODEL_PATHS ADAPTER_PATHS
for i in "${!MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r port served source <<< "${MODEL_ROWS[$i]}"
  adapter=""
  case "$served" in
    qwen-megascience)
      if [[ -z "$QWEN_MEGASCIENCE_ADAPTER" ]]; then
        echo "qwen-megascience is only a served alias. Set QWEN_MEGASCIENCE_ADAPTER to its Hugging Face repo ID." >&2
        exit 2
      fi
      source="$QWEN_MEGASCIENCE_BASE"
      adapter="$QWEN_MEGASCIENCE_ADAPTER"
      ;;
    nvidia-nemotron) source="$NEMOTRON_SOURCE" ;;
    */*) ;;
    *)
      echo "Cannot download alias '$served'; add a source mapping in this script or use a repo ID in the YAML." >&2
      exit 2
      ;;
  esac

  model_dir="${LOCAL_ROOT}/$(tr '/' '_' <<< "$source")"
  download "$source" "$model_dir"
  if [[ -n "$adapter" ]]; then
    adapter_dir="${LOCAL_ROOT}/$(tr '/' '_' <<< "$adapter")"
    download "$adapter" "$adapter_dir"
  else
    adapter_dir=""
  fi
  PORTS+=("$port"); SERVED_NAMES+=("$served"); MODEL_PATHS+=("$model_dir"); ADAPTER_PATHS+=("$adapter_dir")
done

for i in "${!MODEL_ROWS[@]}"; do
  base_served_name="${SERVED_NAMES[$i]}"
  [[ -n "${ADAPTER_PATHS[$i]}" ]] && base_served_name="${SERVED_NAMES[$i]}-base"
  args=(serve "${MODEL_PATHS[$i]}" --host 127.0.0.1 --port "${PORTS[$i]}"
        --served-model-name "$base_served_name" --tensor-parallel-size "${TP_SIZES[$i]}")
  if [[ -n "${ADAPTER_PATHS[$i]}" ]]; then
    args+=(--enable-lora --lora-modules "${SERVED_NAMES[$i]}=${ADAPTER_PATHS[$i]}")
  fi
  # Intentional word splitting permits a site to append multiple vLLM flags.
  # shellcheck disable=SC2206
  extra=( $VLLM_EXTRA_ARGS )
  echo "Starting ${SERVED_NAMES[$i]} on GPU(s) ${GPU_SETS[$i]}, port ${PORTS[$i]}"
  CUDA_VISIBLE_DEVICES="${GPU_SETS[$i]}" vllm "${args[@]}" "${extra[@]}" \
    >"${LOG_ROOT}/${SERVED_NAMES[$i]}.log" 2>&1 &
  PIDS+=("$!")
done

deadline=$((SECONDS + READY_TIMEOUT))
for i in "${!MODEL_ROWS[@]}"; do
  url="http://127.0.0.1:${PORTS[$i]}/v1/models"
  until models_json=$(curl --fail --silent "$url"); do
    kill -0 "${PIDS[$i]}" 2>/dev/null || {
      echo "${SERVED_NAMES[$i]} exited during startup; log follows:" >&2
      tail -n 100 "${LOG_ROOT}/${SERVED_NAMES[$i]}.log" >&2
      exit 1
    }
    (( SECONDS < deadline )) || { echo "Timed out waiting for $url" >&2; exit 1; }
    sleep 5
  done
  python - "${SERVED_NAMES[$i]}" "$models_json" <<'PY'
import json
import sys

served_name, payload = sys.argv[1:]
available = {item["id"] for item in json.loads(payload)["data"]}
if served_name not in available:
    raise SystemExit(f"Expected served model {served_name!r}; endpoint advertised {sorted(available)!r}")
PY
  echo "Ready: ${SERVED_NAMES[$i]}"
done

echo "Running end-to-end ModelRegistry smoke test"
PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python - "$CONFIG" <<'PY'
import asyncio
import sys
from theo_conductor.models.registry import ModelRegistry

async def main():
    registry = ModelRegistry.from_yaml_file(sys.argv[1])
    for key, spec in registry._models.items():
        response = await spec.client.generate(
            instruction="Reply with exactly: SMOKE_TEST_OK",
            question="Verify this inference endpoint.",
            context={},
            max_tokens=32,
            temperature=0.0,
        )
        if not response.text.strip():
            raise RuntimeError(f"{key}: empty completion")
        print(f"PASS {key}: {response.text[:120]!r}")

asyncio.run(main())
PY

echo "All configured models and the theo-conductor client pipeline passed. Logs: $LOG_ROOT"
