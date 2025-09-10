#!/usr/bin/env bash
set -euo pipefail

# Sweep a list of concepts (one per line) through ACL novelty gate.
# Usage: scripts/acl_sweep.sh <topics.txt> [MEMDIR=.guardian_mem] [NOVEL_THETA=0.7]

LIST_FILE=${1:-files/configs/topics.txt}
MEMDIR=${2:-.guardian_mem}
NOVEL_THETA=${3:-0.7}

if [ ! -f "$LIST_FILE" ]; then
  echo "topics file not found: $LIST_FILE" 1>&2
  exit 1
fi

mkdir -p "$MEMDIR"

while IFS= read -r topic; do
  topic_trimmed=$(echo "$topic" | sed -E 's/^\s+|\s+$//g')
  [ -z "$topic_trimmed" ] && continue
  echo "[ACL] concept: $topic_trimmed"
  python3 guardian_agi_min.py --task acl --memdir "$MEMDIR" --concept "$topic_trimmed" --novel-theta "$NOVEL_THETA" || true
done < "$LIST_FILE"

echo "[ACL] sweep done. Memory: $MEMDIR"

