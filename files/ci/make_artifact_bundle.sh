#!/usr/bin/env bash
set -euo pipefail
mkdir -p out
tar -I 'zstd -19 -T0' -cf out/ucbxtot-proof.tar.zst \
  out/out_ctx8k.json out/out_ctx16k.json out/cv_tost.json out/pvals.json \
  out/novelty.json out/policy_scan.json out/ledger_check.json out/energy_mode_check.json \
  out/payload.hash pins/*.sha256 pins/*.commit pins/energy.cal.json \
  files/configs/env.hash logs/rekor-local.jsonl logs/atf.daily.jsonl \
  reports/b1_b5_report.json || echo "Note: some inputs missing; bundle may be partial"
echo "bundle at out/ucbxtot-proof.tar.zst"

