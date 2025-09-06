#!/usr/bin/env bash
set -euo pipefail

# Run Gate v2 selection + Council vote and append to transparency log.
# Env vars (optional):
#   SPEC=UCBxTOT-gold.json
#   CANDS=files/configs/gatev2_candidates.json
#   COUNCIL_IN=files/configs/council_input.json

SPEC=${SPEC:-UCBxTOT-gold.json}
CANDS=${CANDS:-files/configs/gatev2_candidates.json}
COUNCIL_IN=${COUNCIL_IN:-files/configs/council_input.json}

mkdir -p out logs

python3 files/tools/gate_v2.py \
  --spec "$SPEC" \
  --candidates "$CANDS" \
  --pvals-out out/pvals.json \
  --out out/gate_v2.json

python3 files/tools/council_vote.py \
  --spec "$SPEC" \
  --input "$COUNCIL_IN" \
  --out out/council.json

# payload for promotion proof
python3 files/tools/hash_payload.py out/gate_v2.json out/council.json > out/promotion.hash
python3 files/tools/rekor_append.py logs/rekor-local.jsonl out/promotion.hash || true

echo "Promotion stage results: out/gate_v2.json, out/council.json"

