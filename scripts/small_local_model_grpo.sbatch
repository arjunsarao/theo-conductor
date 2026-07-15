#!/usr/bin/env bash
#SBATCH --job-name=conductor-grpo
#SBATCH --partition=train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Run the three worker vLLM servers and the GRPO conductor training process on
# one 8xH200 node. Workers use GPUs 0-5; GRPO uses GPUs 6-7.
#
# Submit from the repository root:
#   sbatch scripts/slurm_grpo_single_node.sbatch
#
# Site-specific examples:
#   VLLM_ENV=/path/to/venv sbatch scripts/slurm_grpo_single_node.sbatch
#   TRAIN_EXTRA_ARGS='--max-steps 10 --wandb-run-name smoke' \
#     sbatch scripts/slurm_grpo_single_node.sbatch

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/local_small_models.yaml}"
# Defaults to persistent storage so models prepared with download_grpo_models.sh
# survive between batch allocations. Set LOCAL_MODEL_ROOT to opt into node-local
# staging for a particular job.
LOCAL_ROOT="${LOCAL_MODEL_ROOT:-${THEO_MODEL_ROOT:-/mnt/data/home/arjun/.cache/theo-conductor/models}}"
LOG_ROOT="${LOCAL_ROOT}/logs"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
KILL_EXISTING_PORTS="${KILL_EXISTING_PORTS:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-6,7}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${REPO_ROOT}/outputs/grpo-${SLURM_JOB_ID:-manual}}"
WANDB_PROJECT="${WANDB_PROJECT:-theo-conductor}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

export HF_HOME="${THEO_HF_HOME:-/mnt/data/home/arjun/.cache/huggingface}"
export HF_HUB_CACHE="${THEO_HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${THEO_TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_XET_CACHE="${THEO_HF_XET_CACHE:-${HF_HOME}/xet}"

# Two GPUs per 32B worker. The conductor training process uses the remaining
# two H200s and runs its own vLLM generation backend there.
MODEL_GPU_SETS="${MODEL_GPU_SETS:-0,1;2,3;4,5}"
MODEL_TP_SIZES="${MODEL_TP_SIZES:-2;2;2}"

if [[ -n "${VLLM_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${VLLM_ENV}/bin/activate"
fi

for command in python hf vllm curl; do
  command -v "$command" >/dev/null || { echo "Required command not found: $command" >&2; exit 2; }
done
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }
mkdir -p "$LOCAL_ROOT" "$LOG_ROOT" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_XET_CACHE"

IFS=';' read -r -a GPU_SETS <<< "$MODEL_GPU_SETS"
IFS=';' read -r -a TP_SIZES <<< "$MODEL_TP_SIZES"

mapfile -t MODEL_ROWS < <(python - "$CONFIG" <<'PY'
import sys
from urllib.parse import urlparse
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    models = yaml.safe_load(handle)["models"]
for model in models:
    client = model["client"]
    port = urlparse(client["base_url"]).port
    print(f'{port}\t{client["model"]}\t{model["context_length"]}')
PY
)

if (( ${#MODEL_ROWS[@]} != ${#GPU_SETS[@]} || ${#MODEL_ROWS[@]} != ${#TP_SIZES[@]} )); then
  echo "MODEL_GPU_SETS and MODEL_TP_SIZES must each have ${#MODEL_ROWS[@]} entries" >&2
  exit 2
fi

port_is_busy() {
  python - "$1" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(0)
else:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

for row in "${MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r port _ _ <<< "$row"
  if port_is_busy "$port"; then
    if [[ "$KILL_EXISTING_PORTS" == "1" ]] && command -v fuser >/dev/null; then
      fuser -k -TERM "${port}/tcp" || true
      sleep 2
    else
      echo "Port $port is already in use; refusing to start." >&2
      exit 2
    fi
  fi
done

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

declare -a PORTS SERVED_NAMES MODEL_PATHS CONTEXT_LENGTHS
for row in "${MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r port served context_length <<< "$row"
  model_dir="${LOCAL_ROOT}/$(tr '/' '_' <<< "$served")"
  download "$served" "$model_dir"
  PORTS+=("$port")
  SERVED_NAMES+=("$served")
  MODEL_PATHS+=("$model_dir")
  CONTEXT_LENGTHS+=("$context_length")
done

for i in "${!MODEL_ROWS[@]}"; do
  args=(serve "${MODEL_PATHS[$i]}" --host 127.0.0.1 --port "${PORTS[$i]}"
        --served-model-name "${SERVED_NAMES[$i]}"
        --tensor-parallel-size "${TP_SIZES[$i]}"
        --max-model-len "${CONTEXT_LENGTHS[$i]}"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
  [[ -n "$KV_CACHE_DTYPE" ]] && args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  # shellcheck disable=SC2206
  extra=( $VLLM_EXTRA_ARGS )
  log_name="${SERVED_NAMES[$i]//\//_}"
  echo "Starting ${SERVED_NAMES[$i]} on GPU(s) ${GPU_SETS[$i]}, port ${PORTS[$i]}"
  CUDA_VISIBLE_DEVICES="${GPU_SETS[$i]}" vllm "${args[@]}" "${extra[@]}" \
    >"${LOG_ROOT}/${log_name}.log" 2>&1 &
  PIDS+=("$!")
done

deadline=$((SECONDS + READY_TIMEOUT))
for i in "${!MODEL_ROWS[@]}"; do
  url="http://127.0.0.1:${PORTS[$i]}/v1/models"
  until models_json=$(curl --fail --silent "$url"); do
    kill -0 "${PIDS[$i]}" 2>/dev/null || {
      echo "${SERVED_NAMES[$i]} exited during startup; log follows:" >&2
      log_name="${SERVED_NAMES[$i]//\//_}"
      tail -n 100 "${LOG_ROOT}/${log_name}.log" >&2
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

echo "Running worker ModelRegistry smoke test"
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

echo "Starting GRPO training on GPU(s) ${TRAIN_GPUS}"
train_extra=(--use-vllm --wandb-project "$WANDB_PROJECT")
# shellcheck disable=SC2206
train_extra+=( $TRAIN_EXTRA_ARGS )
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m theo_conductor.train \
    --config-path "$CONFIG" \
    --output-dir "$TRAIN_OUTPUT_DIR" \
    "${train_extra[@]}"

echo "GRPO training completed. Output: $TRAIN_OUTPUT_DIR"
