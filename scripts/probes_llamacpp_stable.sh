#!/usr/bin/env bash
set -euo pipefail

# Stable llama.cpp probe runner: warmup + two probe runs with fixed params.
# Env vars you can set:
#   SERVER=http://127.0.0.1:8090
#   CTX8=6144
#   CTX16=8192
#   N=200
#   NEW=24
#   WARMUP=20

SERVER=${SERVER:-http://127.0.0.1:8090}
CTX8=${CTX8:-6144}
CTX16=${CTX16:-8192}
N=${N:-200}
NEW=${NEW:-24}
WARMUP=${WARMUP:-20}

echo "[stable-probe] warming up llama.cpp at $SERVER ($WARMUP requests)"
for i in $(seq 1 "$WARMUP"); do
  curl -s "$SERVER/completion" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Hello","n_predict":8,"temperature":0.0}' >/dev/null || true
done

mkdir -p out pins

# Prefer project venv Python if present
if [[ -x .ucbxtot/.venv/bin/python ]]; then
  PYV=.ucbxtot/.venv/bin/python
else
  PYV=python
fi

echo "[stable-probe] energy calibrate (CTX=$CTX8)"
CTX="$CTX8" SERVER="$SERVER" "$PYV" files/probes/energy_calibrate_llamacpp.py || true

echo "[stable-probe] measure ctx8k surrogate (CTX=$CTX8, N=$N, NEW=$NEW)"
N="$N" NEW="$NEW" CTX="$CTX8" SERVER="$SERVER" DUMP_SAMPLES=1 \
  "$PYV" files/probes/latency_energy_probe_llamacpp.py > out/out_ctx8k.json

echo "[stable-probe] measure ctx16k surrogate (CTX=$CTX16, N=$N, NEW=$NEW)"
N="$N" NEW="$NEW" CTX="$CTX16" SERVER="$SERVER" DUMP_SAMPLES=1 \
  "$PYV" files/probes/latency_energy_probe_llamacpp.py > out/out_ctx16k.json

echo "[stable-probe] energy mode check"
python files/tools/energy_mode_assert.py pins/energy.cal.json out/out_ctx8k.json out/out_ctx16k.json || true

echo "[stable-probe] done. Outputs: out/out_ctx8k.json out/out_ctx16k.json"
