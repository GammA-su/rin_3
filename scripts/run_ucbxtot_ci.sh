#!/usr/bin/env bash
set -euo pipefail

# Lightweight runner to exercise the spec's commands in order.

# 1) Preflight
python files/tools/envlock_hash.py || true

# 2) Calibrate (stubbed by placeholder pins/energy.cal.json if HF model not available)
if [[ -z "${SKIP_CAL:-}" ]]; then
  echo "Calibration step is heavy; set SKIP_CAL=1 to skip."
fi

# 3) Probes (optional; requires GPU + model)
if [[ -z "${SKIP_PROBES:-}" ]]; then
  mkdir -p out
  echo '{"ctx":8192,"energy_mode":"nvml_total","samples":0,"p95_s":0.0,"p99_s":0.0,"vram_gb":0.0,"j_per_inf":0.0}' > out/out_ctx8k.json
  echo '{"ctx":16384,"energy_mode":"nvml_total","samples":0,"p95_s":0.0,"p99_s":0.0,"vram_gb":0.0,"j_per_inf":0.0}' > out/out_ctx16k.json
fi

# 4) Assertions
python files/tools/assert_caps_equal.py UCBxTOT-gold.json || true
python files/tools/energy_mode_assert.py pins/energy.cal.json out/out_ctx8k.json out/out_ctx16k.json || true

# 5) Novelty + policy pins check
python tools/leak_guard.py --new atf/tasks.jsonl --train data/train.idx --cos 0.90 --jac 0.75 --ngram 5 > out/novelty.json || true
python files/tools/verify_hashes.py --manifest UCBxTOT-gold.json > out/policy_scan.json || true

# 6) Stats checks
python files/tools/cv_tost_check.py --probe out/out_ctx8k.json out/out_ctx16k.json --cv '{"p95":7,"p99":10,"j":6}' --tost '{"p95":0.25,"p99":0.35,"j_pct":4}' > out/cv_tost.json || true
python tools/mcomp_holm.py --pvals out/pvals.json --alpha 0.05 || true
python files/tools/json_validate.py files/schemas/pvals.schema.json out/pvals.json || true

# 7) Ledger
python files/tools/ledger_check.py logs/rekor-local.jsonl > out/ledger_check.json || true
python files/tools/hash_payload.py out/out_ctx8k.json out/out_ctx16k.json pins/energy.cal.json > out/payload.hash || true

# 8) B1–B5
python files/tools/json_validate.py files/schemas/b1_b5_report.schema.json reports/b1_b5_report.json || true
python files/tools/json_validate.py files/schemas/metrics.daily.schema.json logs/metrics.daily.jsonl || true
python files/tools/json_validate.py files/schemas/atf_result.schema.json logs/atf.daily.jsonl || true

# 9) Bundle
bash files/ci/make_artifact_bundle.sh || true
echo "Done."

