#!/usr/bin/env bash
set -euo pipefail

# Memory allocator tuning to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}

MODEL=${MODEL:-openai/gpt-oss-20b}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-8192}
GPU_MEM=${GPU_MEM:-0.90}
SWAP_GB=${SWAP_GB:-64}
DTYPE=${DTYPE:-}
EXTRA=${EXTRA:-}

# Clamp swap to available host RAM minus 4 GiB to satisfy vLLM validation
mem_kb=$(grep -E '^MemTotal:' /proc/meminfo | awk '{print $2}')
mem_gb=$(( (mem_kb + 1024*1024 - 1) / (1024*1024) ))
max_swap=$(( mem_gb - 4 ))
if [ "$max_swap" -lt 8 ]; then max_swap=8; fi
if [ "$SWAP_GB" -gt "$max_swap" ]; then
  echo "Requested SWAP_GB=$SWAP_GB exceeds host RAM ($mem_gb GiB). Clamping to $max_swap GiB." >&2
  SWAP_GB=$max_swap
fi

# Auto-select dtype for MXFP4 models (requires bfloat16). Respect explicit DTYPE if provided.
if [ -z "${DTYPE}" ]; then
  if [[ "$MODEL" == *"gpt-oss"* ]] || [[ "${QUANT:-}" == "mxfp4" ]]; then
    DTYPE=bfloat16
  else
    DTYPE=float16
  fi
fi

echo "Starting vLLM for $MODEL on :$PORT (max-len=$MAX_LEN, util=$GPU_MEM, swap=${SWAP_GB}GB, dtype=$DTYPE)"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --dtype "$DTYPE" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" \
  --swap-space "$SWAP_GB" \
  --port "$PORT" \
  --enforce-eager \
  $EXTRA
