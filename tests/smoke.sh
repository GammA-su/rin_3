#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[*] Running smoke suite (offline, mock LLM)…"
python3 guardian_agi_min.py --mock-llm --smoke
echo "[*] Running demos (pagerank, compare) with memory…"
python3 guardian_agi_min.py --mock-llm --task pagerank --memdir .guardian_mem
python3 guardian_agi_min.py --mock-llm --task compare  --memdir .guardian_mem
echo "[*] Memory summary:"
python3 guardian_agi_min.py --mock-llm --showmem --memdir .guardian_mem
echo "[✓] Smoke completed."
