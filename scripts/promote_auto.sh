#!/usr/bin/env bash
set -euo pipefail

# Auto-promotion: use current probe outputs + optional pvals to create a
# candidate, run Gate v2 + Council, and append to transparency log.

# Env vars (optional):
#   SPEC=UCBxTOT-gold.json
#   PVALS_IN=files/configs/pvals.input.json
#   CAND_OUT=files/configs/gatev2_candidates.auto.json
#   SERVER=http://127.0.0.1:8090

SPEC=${SPEC:-UCBxTOT-gold.json}
PVALS_IN=${PVALS_IN:-}
CAND_OUT=${CAND_OUT:-files/configs/gatev2_candidates.auto.json}

mkdir -p out logs

# If probes missing, try stable probes first
if [[ ! -s out/out_ctx8k.json || ! -s out/out_ctx16k.json ]]; then
  echo "[promote-auto] probes missing; running stable llama.cpp probes"
  SERVER=${SERVER:-http://127.0.0.1:8090} bash scripts/probes_llamacpp_stable.sh || true
fi

ARGS=( --spec "$SPEC" --probe8 out/out_ctx8k.json --probe16 out/out_ctx16k.json --out "$CAND_OUT" )
if [[ -n "${PVALS_IN:-}" && -s "$PVALS_IN" ]]; then
  ARGS+=( --pvals-in "$PVALS_IN" )
fi
python3 files/tools/candidates_from_probes.py "${ARGS[@]}"

python3 files/tools/gate_v2.py --spec "$SPEC" --candidates "$CAND_OUT" --pvals-out out/pvals.json --out out/gate_v2.json

# Council input and vote
python3 files/tools/collect_council_input.py --spec "$SPEC" --gate out/gate_v2.json --cv out/cv_tost.json --policy out/policy_scan.json --energy out/energy_mode_check.json --out files/configs/council_input.auto.json || true
python3 files/tools/council_vote.py --spec "$SPEC" --input files/configs/council_input.auto.json --out out/council.json

# Append to transparency log
python3 files/tools/hash_payload.py out/gate_v2.json out/council.json > out/promotion.hash
python3 files/tools/rekor_append.py logs/rekor-local.jsonl out/promotion.hash || true

echo "[promote-auto] done. Outputs: out/gate_v2.json out/council.json (cand: $CAND_OUT)"

