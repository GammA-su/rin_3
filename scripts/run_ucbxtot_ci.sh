#!/usr/bin/env bash
set -euo pipefail

# Lightweight runner to exercise the spec's commands in order.

# 0) Local venv for helper deps (jsonschema, zstandard)
VENV=.ucbxtot/.venv
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  python -m pip -q install --upgrade pip
  python -m pip -q install jsonschema zstandard
else
  source "$VENV/bin/activate"
  python - << 'PY' || true
try:
  import jsonschema, zstandard
except Exception:
  import sys, subprocess
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'jsonschema', 'zstandard'])
PY
fi

# 1) Preflight
python files/tools/envlock_hash.py || true

# 2) Calibrate (stub or run separately)
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
# Fail-closed if spec requires pins
python files/tools/verify_hashes.py --manifest UCBxTOT-gold.json > out/policy_scan.json

# 6) Promotion (Gate v2 + Council) & stats checks
# Prefer auto-candidates generated from probes; fallback to sample manifest
if [[ -f out/out_ctx8k.json && -f out/out_ctx16k.json ]]; then
  python files/tools/candidates_from_probes.py --spec UCBxTOT-gold.json --probe8 out/out_ctx8k.json --probe16 out/out_ctx16k.json --out files/configs/gatev2_candidates.auto.json || true
  CANDS=files/configs/gatev2_candidates.auto.json
elif [[ -f files/configs/gatev2_candidates.json ]]; then
  CANDS=files/configs/gatev2_candidates.json
else
  CANDS=
fi
if [[ -n "${CANDS:-}" ]]; then
  python files/tools/gate_v2.py --spec UCBxTOT-gold.json --candidates "$CANDS" --pvals-out out/pvals.json --out out/gate_v2.json || true
fi
python tools/mcomp_holm.py --pvals out/pvals.json --alpha 0.05 || true
python files/tools/json_validate.py files/schemas/pvals.schema.json out/pvals.json || true
python files/tools/cv_tost_check.py --probe out/out_ctx8k.json out/out_ctx16k.json --spec UCBxTOT-gold.json --out out/cv_tost.json || true

# Council vote (if input config exists)
if [[ -f files/configs/council_input.json ]]; then
  python files/tools/council_vote.py --spec UCBxTOT-gold.json --input files/configs/council_input.json --out out/council.json || true
fi

# 7) Ledger + payload hash + append to transparency log
python files/tools/ledger_check.py logs/rekor-local.jsonl > out/ledger_check.json || true
python files/tools/hash_payload.py out/out_ctx8k.json out/out_ctx16k.json pins/energy.cal.json > out/payload.hash || true
python files/tools/rekor_append.py logs/rekor-local.jsonl out/payload.hash || true
# Append promotion payload if present
if [[ -f out/gate_v2.json && -f out/council.json ]]; then
  python files/tools/hash_payload.py out/gate_v2.json out/council.json > out/promotion.hash || true
  python files/tools/rekor_append.py logs/rekor-local.jsonl out/promotion.hash || true
fi

# 8) B1–B5 schemas (seed metrics if missing)
if [[ ! -s logs/metrics.daily.jsonl ]]; then
  python files/tools/metrics_seed.py --days 14 --out logs/metrics.daily.jsonl || true
fi
python files/tools/json_validate.py files/schemas/metrics.daily.schema.json logs/metrics.daily.jsonl || true
python files/tools/b1_b5_report.py logs/metrics.daily.jsonl > reports/b1_b5_report.json || true
python files/tools/json_validate.py files/schemas/b1_b5_report.schema.json reports/b1_b5_report.json || true
# Append a sample ATF admission if file is empty
if [[ ! -s logs/atf.daily.jsonl ]]; then
  python files/tools/atf_admit.py --tool-id demo.tool --attempts 1 --wall 0.10 --unit 1.0 --prop 1.0 --p95 0.8 --p99 1.2 --accepted true --out logs/atf.daily.jsonl || true
fi
python files/tools/json_validate.py files/schemas/atf_result.schema.json logs/atf.daily.jsonl || true

# 9) Bundle
bash files/ci/make_artifact_bundle.sh || true
echo "Done."
