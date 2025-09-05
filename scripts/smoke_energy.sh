#!/usr/bin/env bash
set -euo pipefail

# Simple, OOM-safe energy smoke probe.
# Usage examples:
#   BACKEND=hf ./scripts/smoke_energy.sh
#   BACKEND=ollama OLLAMA_MODEL=gpt-oss-20b:Q4_K_M ./scripts/smoke_energy.sh
#   BACKEND=hf HF_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 LOAD_IN_4BIT=1 ./scripts/smoke_energy.sh

BACKEND=${BACKEND:-hf}         # hf | ollama | vllm | jan | llamacpp
OUTDIR=out
mkdir -p "$OUTDIR" pins

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}

case "$BACKEND" in
  hf)
    # Safe defaults for HF path: small model, 4-bit, reduced CTX/NEW
    export HF_MODEL=${HF_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
    export LOAD_IN_4BIT=${LOAD_IN_4BIT:-1}
    export GPU_MAX_GB=${GPU_MAX_GB:-22}
    export CPU_MAX_GB=${CPU_MAX_GB:-64}
    export ATTN_IMPL=${ATTN_IMPL:-eager}
    export EMPTY_CACHE=${EMPTY_CACHE:-1}
    export N=${N:-3}
    export NEW=${NEW:-8}

    # Calibrate and produce 2 probe outputs (we keep the filenames the spec expects)
    CTX=${CTX:-2048} python files/probes/energy_calibrate.py
    CTX=2048  python files/probes/latency_energy_probe.py > "$OUTDIR/out_ctx8k.json"
    CTX=4096  python files/probes/latency_energy_probe.py > "$OUTDIR/out_ctx16k.json"
    ;;
  ollama)
    # Use your running Ollama model; adjust OLLAMA_MODEL as needed.
    export OLLAMA_MODEL=${OLLAMA_MODEL:-gpt-oss-20b}
    export N=${N:-3}
    export NEW=${NEW:-8}
    CTX=${CTX:-2048} python files/probes/energy_calibrate_ollama.py
    CTX=2048  python files/probes/latency_energy_probe_ollama.py > "$OUTDIR/out_ctx8k.json"
    CTX=4096  python files/probes/latency_energy_probe_ollama.py > "$OUTDIR/out_ctx16k.json"
    ;;
  vllm)
    # Assumes vLLM server is already running at $SERVER and serving $MODEL
    export N=${N:-3}
    export NEW=${NEW:-8}
    CTX=${CTX:-2048} python files/probes/energy_calibrate_vllm.py || true
    CTX=2048  python files/probes/latency_energy_probe_vllm.py > "$OUTDIR/out_ctx8k.json"
    CTX=4096  python files/probes/latency_energy_probe_vllm.py > "$OUTDIR/out_ctx16k.json"
    ;;
  jan)
    # Jan desktop app exposes OpenAI API at 127.0.0.1:1337/v1
    export SERVER=${SERVER:-http://127.0.0.1:1337/v1}
    export MODEL=${MODEL:-janhq/Jan-v1-4B-GGUF:Q4_K_M}
    export N=${N:-3}
    export NEW=${NEW:-8}
    CTX=${CTX:-2048} python files/probes/energy_calibrate_vllm.py || true
    CTX=2048  python files/probes/latency_energy_probe_vllm.py > "$OUTDIR/out_ctx8k.json"
    CTX=4096  python files/probes/latency_energy_probe_vllm.py > "$OUTDIR/out_ctx16k.json"
    ;;
  llamacpp)
    # llama.cpp server (default http://127.0.0.1:8080). Use scripts/run_llamacpp_server.sh to start.
    export SERVER=${SERVER:-http://127.0.0.1:8080}
    export N=${N:-3}
    export NEW=${NEW:-8}
    CTX=${CTX:-2048} python files/probes/energy_calibrate_llamacpp.py
    # Use conservative prompt sizes to avoid HTTP 400 if server n_ctx differs
    CTX=2048  python files/probes/latency_energy_probe_llamacpp.py > "$OUTDIR/out_ctx8k.json"
    CTX=3072  python files/probes/latency_energy_probe_llamacpp.py > "$OUTDIR/out_ctx16k.json"
    ;;
  *)
    echo "Unknown BACKEND=$BACKEND (use hf|ollama|vllm)" >&2
    exit 2
    ;;
esac

python files/tools/energy_mode_assert.py pins/energy.cal.json "$OUTDIR/out_ctx8k.json" "$OUTDIR/out_ctx16k.json" || true
echo "Energy smoke finished (backend=$BACKEND). Outputs in $OUTDIR/"
