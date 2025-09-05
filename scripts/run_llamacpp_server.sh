#!/usr/bin/env bash
set -euo pipefail

# Start llama.cpp HTTP server for a GGUF model.
# Requirements: built llama.cpp server binary (llama-server) on PATH or at ./llama.cpp/server
# Example:
#   GGUF=~/models/gpt-oss-20b-Q4_K_M.gguf ./scripts/run_llamacpp_server.sh
#   GGUF=~/models/gpt-oss-20b-Q5_K_M.gguf PORT=8080 NGL=99 CTX=8192 ./scripts/run_llamacpp_server.sh

GGUF=${GGUF:-}
if [[ -z "${GGUF}" ]]; then
  echo "Set GGUF=/path/to/gpt-oss-20b-*.gguf" >&2
  exit 2
fi

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
CTX=${CTX:-8192}
NGL=${NGL:-99}     # GPU layers to offload; reduce to offload more to CPU
MAIN_GPU=${MAIN_GPU:-0}
THREADS=${THREADS:-$(nproc)}
BATCH=${BATCH:-512}
UBATCH=${UBATCH:-}
FLASH_ATTN=${FLASH_ATTN:-}

BIN=${LLAMA_SERVER_BIN:-}
if [[ -z "${BIN}" ]]; then
  if command -v llama-server >/dev/null 2>&1; then
    BIN=$(command -v llama-server)
  elif [[ -x ./llama.cpp/server ]]; then
    BIN=./llama.cpp/server
  else
    echo "Could not find llama-server. Put it on PATH or set LLAMA_SERVER_BIN." >&2
    exit 2
  fi
fi

ARGS=( -m "$GGUF" -ngl "$NGL" --main-gpu "$MAIN_GPU" -c "$CTX" -t "$THREADS" -b "$BATCH" --host "$HOST" --port "$PORT" )
if [[ -n "$UBATCH" ]]; then ARGS+=( -ub "$UBATCH" ); fi
# llama.cpp now expects a value for --flash-attn ('on'|'off'|'auto')
if [[ -n "${FLASH_ATTN}" ]]; then
  # normalize common truthy values to 'on'
  case "${FLASH_ATTN,,}" in
    on|true|1)  ARGS+=( --flash-attn on );;
    off|false|0) ARGS+=( --flash-attn off );;
    auto) ARGS+=( --flash-attn auto );;
    *) ARGS+=( --flash-attn "${FLASH_ATTN}" );;
  esac
fi

echo "Starting llama.cpp server: $BIN ${ARGS[*]}"
exec "$BIN" "${ARGS[@]}"
