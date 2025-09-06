#!/usr/bin/env bash
set -euo pipefail
mkdir -p out

OUT=out/ucbxtot-proof.tar.zst
MANDATORY=(
  out/out_ctx8k.json
  out/out_ctx16k.json
  out/cv_tost.json
  out/pvals.json
  out/novelty.json
  out/policy_scan.json
  out/ledger_check.json
  out/energy_mode_check.json
  out/payload.hash
  files/configs/env.hash
  logs/rekor-local.jsonl
  logs/atf.daily.jsonl
  reports/b1_b5_report.json
)
OPTIONAL=( out/gate_v2.json out/council.json out/promotion.hash )

ARGS=()
for f in "${MANDATORY[@]}"; do
  [[ -e "$f" ]] && ARGS+=("$f") || echo "[warn] missing $f" >&2
done
for f in pins/*.sha256 pins/*.commit pins/energy.cal.json; do
  [[ -e "$f" ]] && ARGS+=("$f") || true
done
for f in "${OPTIONAL[@]}"; do
  [[ -e "$f" ]] && ARGS+=("$f") || true
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "[error] nothing to bundle" >&2; exit 1
fi

tar -I 'zstd -19 -T0' -cf "$OUT" "${ARGS[@]}"
echo "bundle at $OUT"
