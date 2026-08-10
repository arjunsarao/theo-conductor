#!/usr/bin/env bash
#SBATCH --job-name=theo-conductor
#SBATCH --partition=train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arjun@firstprinciples.com

# Launch and verify a worker pool described by MODEL_CONFIG. Each model must
# have deployment.mode set to "local" or "remote". Local entries are
# downloaded and served by this job; remote entries are readiness-checked only.
#
# Examples:
#   RUN_MODE=smoke sbatch scripts/worker_pool.sbatch
#   RUN_MODE=benchmark sbatch scripts/worker_pool.sbatch
#   MODEL_CONFIG=configs/worker_pool_large.yaml RUN_MODE=train \
#     sbatch scripts/worker_pool.sbatch

set -Eeuo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/worker_pool_large.yaml}"
RUN_MODE="${RUN_MODE:-smoke}"
DATASET="${DATASET:-megascience}"
DATASET_SAMPLES="${DATASET_SAMPLES:-}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-200}"
LOCAL_ROOT="${LOCAL_MODEL_ROOT:-${THEO_MODEL_ROOT:-/mnt/data/home/arjun/.cache/theo-conductor/models}}"
LOG_ROOT="${LOCAL_ROOT}/logs"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
WORKER_API_KEY="${WORKER_API_KEY:-change-this}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
KILL_EXISTING_PORTS="${KILL_EXISTING_PORTS:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${REPO_ROOT}/outputs/grpo-${SLURM_JOB_ID:-manual}}"
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-${REPO_ROOT}/outputs/${DATASET}-worker-pool}"
WANDB_PROJECT="${WANDB_PROJECT:-theo-conductor}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
GENERATION_GPU="${GENERATION_GPU:-1}"
GENERATION_VLLM_HOST="${GENERATION_VLLM_HOST:-127.0.0.1}"
GENERATION_VLLM_PORT="${GENERATION_VLLM_PORT:-8007}"
GENERATION_VLLM_GROUP_PORT="${GENERATION_VLLM_GROUP_PORT:-51216}"
GENERATION_VLLM_GPU_MEMORY_UTILIZATION="${GENERATION_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
GENERATION_VLLM_MAX_MODEL_LEN="${GENERATION_VLLM_MAX_MODEL_LEN:-8192}"
GENERATION_VLLM_READY_TIMEOUT="${GENERATION_VLLM_READY_TIMEOUT:-900}"
GPU_MEMORY_LOG_INTERVAL="${GPU_MEMORY_LOG_INTERVAL:-5}"

export HF_HOME="${THEO_HF_HOME:-/mnt/data/home/arjun/.cache/huggingface}"
export HF_HUB_CACHE="${THEO_HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${THEO_TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_XET_CACHE="${THEO_HF_XET_CACHE:-${HF_HOME}/xet}"

if [[ -n "${VLLM_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${VLLM_ENV}/bin/activate"
fi

case "$RUN_MODE" in
  smoke|benchmark|train) ;;
  *)
    echo "RUN_MODE must be one of: smoke, benchmark, train (got: $RUN_MODE)" >&2
    exit 2
    ;;
esac

for command in python curl; do
  command -v "$command" >/dev/null || { echo "Required command not found: $command" >&2; exit 2; }
done
[[ -f "$CONFIG" ]] || { echo "Model config not found: $CONFIG" >&2; exit 2; }

if [[ -z "$TRAIN_GPUS" ]]; then
  TRAIN_GPUS=$(python - "$CONFIG" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}
print(data.get("train_gpu_set", "6,7"))
PY
)
fi

mkdir -p "$LOCAL_ROOT" "$LOG_ROOT" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_XET_CACHE"

# Fields: index, ID, mode, base URL, served name, source model, GPU set,
# tensor parallel size, served context, KV-cache dtype, and per-model memory
# utilization.
mapfile -t MODEL_ROWS < <(python - "$CONFIG" <<'PY'
import sys
from urllib.parse import urlparse

import yaml

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}

models = data.get("models")
if not isinstance(models, list) or not models:
    raise SystemExit(f"{path} must contain a non-empty 'models' list")

claimed_gpus = {}
for index, model in enumerate(models):
    if not isinstance(model, dict):
        raise SystemExit(f"{path}: model {index} must be a mapping")
    client = model.get("client")
    deployment = model.get("deployment")
    if not isinstance(client, dict):
        raise SystemExit(f"{path}: model {index} is missing client configuration")
    if not isinstance(deployment, dict):
        raise SystemExit(f"{path}: model {index} is missing deployment configuration")

    mode = deployment.get("mode")
    if mode not in {"local", "remote"}:
        raise SystemExit(f"{path}: model {index} deployment.mode must be local or remote")

    base_url = str(client.get("base_url", "")).rstrip("/")
    served_name = str(client.get("model", ""))
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not served_name:
        raise SystemExit(f"{path}: model {index} has an invalid client endpoint")

    model_id = model.get("model_idx") or model.get("name") or model.get("display_name") or served_name
    source_model = str(deployment.get("source_model", served_name))
    gpu_set = str(deployment.get("gpu_set", ""))
    tensor_parallel = deployment.get("tensor_parallel_size", "")
    max_model_len = deployment.get("max_model_len", model.get("context_length", ""))
    kv_cache_dtype = deployment.get("kv_cache_dtype", "")
    memory_utilization = deployment.get("gpu_memory_utilization", "")

    if mode == "local":
        if parsed.hostname not in {"localhost", "127.0.0.1"} or parsed.port is None:
            raise SystemExit(
                f"{path}: local model {model_id!r} must use a localhost URL with an explicit port"
            )
        if not gpu_set or not isinstance(tensor_parallel, int) or tensor_parallel <= 0:
            raise SystemExit(
                f"{path}: local model {model_id!r} requires gpu_set and positive tensor_parallel_size"
            )
        if len(gpu_set.split(",")) != tensor_parallel:
            raise SystemExit(
                f"{path}: local model {model_id!r} GPU count must match tensor_parallel_size"
            )
        for gpu in gpu_set.split(","):
            if not gpu.isdigit():
                raise SystemExit(f"{path}: local model {model_id!r} has invalid GPU {gpu!r}")
            if gpu in claimed_gpus:
                raise SystemExit(
                    f"{path}: GPU {gpu} is assigned to both {claimed_gpus[gpu]!r} and {model_id!r}"
                )
            claimed_gpus[gpu] = model_id
        if not isinstance(max_model_len, int) or max_model_len <= 0:
            raise SystemExit(f"{path}: local model {model_id!r} requires positive max_model_len")

    fields = [
        index,
        model_id,
        mode,
        base_url,
        served_name,
        source_model,
        gpu_set,
        tensor_parallel,
        max_model_len,
        kv_cache_dtype,
        memory_utilization,
    ]
    if any("\t" in str(value) or "\n" in str(value) for value in fields):
        raise SystemExit(f"{path}: model {model_id!r} contains unsupported tab/newline characters")
    print("\t".join(str(value) for value in fields))
PY
)
(( ${#MODEL_ROWS[@]} > 0 )) || { echo "No valid models were read from $CONFIG" >&2; exit 2; }

LOCAL_COUNT=0
for row in "${MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r _ _ mode _ _ _ _ _ _ _ _ <<< "$row"
  [[ "$mode" == "local" ]] && ((LOCAL_COUNT += 1))
done

if (( LOCAL_COUNT > 0 )); then
  for command in hf vllm; do
    command -v "$command" >/dev/null || { echo "Local workers require command: $command" >&2; exit 2; }
  done
fi
if [[ "$RUN_MODE" == "train" ]]; then
  for command in nvidia-smi trl; do
    command -v "$command" >/dev/null || { echo "Training requires command: $command" >&2; exit 2; }
  done
  IFS=',' read -r -a train_gpus <<< "$TRAIN_GPUS"
  for train_gpu in "${train_gpus[@]}"; do
    [[ "$train_gpu" != "$GENERATION_GPU" ]] || {
      echo "GPU $GENERATION_GPU cannot be assigned to both training and the vLLM generation server" >&2
      exit 2
    }
  done
  for row in "${MODEL_ROWS[@]}"; do
    IFS=$'\t' read -r _ model_id mode _ _ _ gpu_set _ _ _ _ <<< "$row"
    [[ "$mode" == "local" ]] || continue
    IFS=',' read -r -a worker_gpus <<< "$gpu_set"
    for worker_gpu in "${worker_gpus[@]}"; do
      for train_gpu in "${train_gpus[@]}"; do
        [[ "$worker_gpu" != "$train_gpu" ]] || {
          echo "GPU $worker_gpu is assigned to local worker $model_id and TRAIN_GPUS" >&2
          exit 2
        }
      done
    done
  done
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

# Refuse to interfere with listeners unless the operator explicitly identified
# them as stale. Remote endpoints never enter this cleanup path.
for row in "${MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r _ model_id mode base_url _ _ _ _ _ _ _ <<< "$row"
  [[ "$mode" == "local" ]] || continue
  port=$(python - "$base_url" <<'PY'
import sys
from urllib.parse import urlparse
print(urlparse(sys.argv[1]).port)
PY
)
  if port_is_busy "$port"; then
    if [[ "$KILL_EXISTING_PORTS" == "1" ]] && command -v fuser >/dev/null; then
      echo "Terminating stale listener for $model_id on port $port"
      fuser -k -TERM "${port}/tcp" || true
      sleep 2
      port_is_busy "$port" && { echo "Port $port remains busy" >&2; exit 2; }
    else
      echo "Local port $port for $model_id is already in use; refusing to start." >&2
      exit 2
    fi
  fi
done

PIDS=()
GPU_MEMORY_MONITOR_PID=""
declare -A PID_BY_INDEX LOG_BY_INDEX
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$GPU_MEMORY_MONITOR_PID" ]]; then
    kill "$GPU_MEMORY_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MEMORY_MONITOR_PID" 2>/dev/null || true
  fi
  if (( ${#PIDS[@]} )); then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

download() {
  local repo_id=$1 destination=$2
  if [[ -f "$destination/.download-complete" ]]; then
    echo "Already available: $repo_id"
    return
  fi
  mkdir -p "$destination"
  echo "Downloading $repo_id -> $destination"
  hf download "$repo_id" --local-dir "$destination"
  touch "$destination/.download-complete"
}

for row in "${MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r index model_id mode base_url served source gpu_set tp max_len model_kv_dtype model_memory <<< "$row"
  if [[ "$mode" == "remote" ]]; then
    echo "Using remote worker: $model_id ($served at $base_url)"
    continue
  fi

  model_dir="${LOCAL_ROOT}/$(tr '/' '_' <<< "$source")"
  download "$source" "$model_dir"
  port=$(python - "$base_url" <<'PY'
import sys
from urllib.parse import urlparse
print(urlparse(sys.argv[1]).port)
PY
)
  memory="${model_memory:-$GPU_MEMORY_UTILIZATION}"
  kv_dtype="${model_kv_dtype:-$KV_CACHE_DTYPE}"
  args=(serve "$model_dir" --host 127.0.0.1 --port "$port"
        --served-model-name "$served"
        --tensor-parallel-size "$tp"
        --max-model-len "$max_len"
        --gpu-memory-utilization "$memory")
  [[ -n "$kv_dtype" ]] && args+=(--kv-cache-dtype "$kv_dtype")
  # Intentional splitting allows operators to append multiple vLLM flags.
  # shellcheck disable=SC2206
  extra=( $VLLM_EXTRA_ARGS )
  log_name="${served//\//_}"
  log_path="${LOG_ROOT}/${log_name}-${SLURM_JOB_ID:-manual}.log"
  echo "Starting local worker: $model_id on GPU(s) $gpu_set, port $port"
  CUDA_VISIBLE_DEVICES="$gpu_set" vllm "${args[@]}" "${extra[@]}" >"$log_path" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  PID_BY_INDEX["$index"]="$pid"
  LOG_BY_INDEX["$index"]="$log_path"
done

deadline=$((SECONDS + READY_TIMEOUT))
for row in "${MODEL_ROWS[@]}"; do
  IFS=$'\t' read -r index model_id mode base_url served _ _ _ _ _ _ <<< "$row"
  models_url="${base_url%/}/models"
  until models_json=$(curl --fail --silent --show-error \
      --connect-timeout 15 --max-time 30 \
      -H "Authorization: Bearer ${WORKER_API_KEY}" "$models_url"); do
    if [[ "$mode" == "local" ]] && ! kill -0 "${PID_BY_INDEX[$index]}" 2>/dev/null; then
      echo "$model_id exited during startup; log follows:" >&2
      tail -n 100 "${LOG_BY_INDEX[$index]}" >&2
      exit 1
    fi
    (( SECONDS < deadline )) || { echo "Timed out waiting for $models_url" >&2; exit 1; }
    sleep 5
  done
  python - "$served" "$models_json" <<'PY'
import json
import sys

served_name, payload = sys.argv[1:]
available = {item["id"] for item in json.loads(payload)["data"]}
if served_name not in available:
    raise SystemExit(f"Expected {served_name!r}; endpoint advertised {sorted(available)!r}")
PY
  echo "Ready: $model_id ($served)"
done

echo "Running config-driven model access and generation test"
python "${REPO_ROOT}/test_model_access.py" \
  --config "$CONFIG" \
  --max-tokens "${SMOKE_MAX_TOKENS:-1024}" \
  --timeout "${SMOKE_TIMEOUT:-300}"

if [[ "$RUN_MODE" == "smoke" ]]; then
  echo "Worker-pool smoke test passed."
  exit 0
fi

if [[ "$RUN_MODE" == "benchmark" ]]; then
  mkdir -p "$BENCHMARK_OUTPUT_DIR"
  benchmark_args=(
    --config "$CONFIG"
    --output-dir "$BENCHMARK_OUTPUT_DIR"
    --dataset "$DATASET"
    --seed "${BENCHMARK_SEED:-42}"
    --validation-samples "${BENCHMARK_VALIDATION_SAMPLES:-$VALIDATION_SAMPLES}"
    --max-tokens "${BENCHMARK_MAX_TOKENS:-4096}"
    --temperature "${BENCHMARK_TEMPERATURE:-0.0}"
    --concurrency "${BENCHMARK_CONCURRENCY:-4}"
  )
  benchmark_total_samples="${BENCHMARK_TOTAL_SAMPLES:-$DATASET_SAMPLES}"
  [[ -n "$benchmark_total_samples" ]] && benchmark_args+=(--total-samples "$benchmark_total_samples")
  [[ -n "${BENCHMARK_MAX_SAMPLES:-}" ]] && benchmark_args+=(--max-samples "$BENCHMARK_MAX_SAMPLES")
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m theo_conductor.benchmark "${benchmark_args[@]}"
  echo "Benchmark completed. Output: $BENCHMARK_OUTPUT_DIR"
  exit 0
fi

CONDUCTOR_MODEL=$(python - "$CONFIG" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
model = config.get("conductor_model")
if not isinstance(model, str) or not model.strip():
    raise SystemExit(f"{sys.argv[1]} must define a non-empty conductor_model")
print(model)
PY
)

for server_port in "$GENERATION_VLLM_PORT" "$GENERATION_VLLM_GROUP_PORT"; do
  if port_is_busy "$server_port"; then
    echo "Required TRL vLLM server port $server_port is already in use" >&2
    exit 2
  fi
done

generation_log="${LOG_ROOT}/conductor-vllm-${SLURM_JOB_ID:-manual}.log"
echo "Starting dedicated conductor vLLM server on GPU $GENERATION_GPU: $CONDUCTOR_MODEL"
CUDA_VISIBLE_DEVICES="$GENERATION_GPU" trl vllm-serve \
  --model "$CONDUCTOR_MODEL" \
  --host "$GENERATION_VLLM_HOST" \
  --port "$GENERATION_VLLM_PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GENERATION_VLLM_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$GENERATION_VLLM_MAX_MODEL_LEN" \
  >"$generation_log" 2>&1 &
generation_pid=$!
PIDS+=("$generation_pid")

generation_deadline=$((SECONDS + GENERATION_VLLM_READY_TIMEOUT))
generation_health_url="http://${GENERATION_VLLM_HOST}:${GENERATION_VLLM_PORT}/health/"
until curl --fail --silent --show-error --connect-timeout 5 --max-time 15 "$generation_health_url" >/dev/null; do
  if ! kill -0 "$generation_pid" 2>/dev/null; then
    echo "Dedicated conductor vLLM server exited during startup; log follows:" >&2
    tail -n 120 "$generation_log" >&2
    exit 1
  fi
  (( SECONDS < generation_deadline )) || {
    echo "Timed out waiting for $generation_health_url; server log follows:" >&2
    tail -n 120 "$generation_log" >&2
    exit 1
  }
  sleep 5
done
echo "Ready: dedicated conductor vLLM server at $generation_health_url"

mkdir -p "$TRAIN_OUTPUT_DIR"
if [[ "$GPU_MEMORY_LOG_INTERVAL" != "0" ]]; then
  "${REPO_ROOT}/scripts/log_gpu_memory.sh" "$TRAIN_OUTPUT_DIR" "$GPU_MEMORY_LOG_INTERVAL" &
  GPU_MEMORY_MONITOR_PID=$!
fi

echo "Starting executed-workflow GRPO on GPU(s) $TRAIN_GPUS"
train_extra=(
  --use-vllm
  --vllm-mode server
  --vllm-server-base-url "http://${GENERATION_VLLM_HOST}:${GENERATION_VLLM_PORT}"
  --vllm-server-timeout "$GENERATION_VLLM_READY_TIMEOUT"
  --vllm-group-port "$GENERATION_VLLM_GROUP_PORT"
  --execute-workflows
  --wandb-project "$WANDB_PROJECT"
)
train_extra+=(--dataset "$DATASET" --validation-samples "$VALIDATION_SAMPLES")
[[ -n "$DATASET_SAMPLES" ]] && train_extra+=(--dataset-samples "$DATASET_SAMPLES")
# shellcheck disable=SC2206
train_extra+=( $TRAIN_EXTRA_ARGS )
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m theo_conductor.train \
    --config-path "$CONFIG" \
    --output-dir "$TRAIN_OUTPUT_DIR" \
    "${train_extra[@]}"

echo "GRPO training completed. Output: $TRAIN_OUTPUT_DIR"
