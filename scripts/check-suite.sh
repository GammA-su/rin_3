#!/usr/bin/env bash
set -u  # (no -e so jq warnings don't kill the script)
mkdir -p artifacts

# Run the suite and capture its exit code
python3 guardian_agi_min.py --mock-llm --suite full --strict > artifacts/suite_full.json
ec=$?

# Optionally print a quick metric if jq exists, but never change ec
if command -v jq >/dev/null 2>&1; then
  if ! jq -er '.E2E.kpis.ece' artifacts/suite_full.json >/dev/null; then
    echo "[warn] jq failed to read .E2E.kpis.ece (missing key or invalid JSON)" >&2
  else
    echo "ECE=$(jq -r '.E2E.kpis.ece' artifacts/suite_full.json)"
  fi
else
  echo "[info] jq not found; skipping ECE print" >&2
fi

exit "$ec"
