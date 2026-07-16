#!/usr/bin/env bash
# Periodically record whole-GPU and per-process memory usage until terminated.

set -Eeuo pipefail

OUTPUT_DIR=${1:?usage: log_gpu_memory.sh OUTPUT_DIR [INTERVAL_SECONDS]}
INTERVAL=${2:-5}
mkdir -p "$OUTPUT_DIR"

GPU_LOG="${OUTPUT_DIR}/gpu-memory.csv"
PROCESS_LOG="${OUTPUT_DIR}/gpu-process-memory.csv"

echo "timestamp,gpu_index,gpu_uuid,memory_used_mib,memory_free_mib,memory_total_mib,utilization_gpu_percent" >"$GPU_LOG"
echo "timestamp,gpu_uuid,pid,process_name,used_memory_mib" >"$PROCESS_LOG"

while true; do
  timestamp=$(date --utc +%Y-%m-%dT%H:%M:%SZ)

  while IFS= read -r row; do
    [[ -n "$row" ]] && echo "$timestamp,$row" >>"$GPU_LOG"
  done < <(
    nvidia-smi \
      --query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
  )

  while IFS= read -r row; do
    [[ -n "$row" ]] && echo "$timestamp,$row" >>"$PROCESS_LOG"
  done < <(
    nvidia-smi \
      --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true
  )

  sleep "$INTERVAL"
done
