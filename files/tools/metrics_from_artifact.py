#!/usr/bin/env python3
"""
Append a daily B1–B5 metrics line to logs/metrics.daily.jsonl using values
extracted from an eval artifact.

Inputs:
  --spec UCBxTOT-gold.json               (to read domain list)
  --artifact artifacts/suite_full.json   (JSON/JSONL with metric fields)
  --map files/configs/domain_map.json    (optional: {DOMAIN: dot.path, ...})
  --day 12                               (optional: day index; default last+1)
  --out logs/metrics.daily.jsonl

Notes:
  - If no --map is provided, falls back to a simple heuristic:
      domain value = E2E.kpis.pass_at_1 (clamped 0..1), if present; else 0.0.
  - For JSONL input, uses the first non-empty JSON object.
  - Creates the output file if missing.
"""

from __future__ import annotations
import argparse, json, os, pathlib, sys
from typing import Any, Dict


def jload_one(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        raise SystemExit(f"artifact not found: {path}")
    text = p.read_text(encoding='utf-8').strip()
    if not text:
        return {}
    # try JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj[0] if obj else {}
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # JSONL: take first object line
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            continue
    return {}


def jget(d: Dict[str, Any], path: str, default: float = 0.0) -> float:
    cur: Any = d
    try:
        for tok in path.split('.'):
            cur = cur[tok]
        return float(cur)
    except Exception:
        return float(default)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def next_day(path: str) -> int:
    if not os.path.exists(path):
        return 1
    last = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                last = int(d.get('day', last))
            except Exception:
                continue
    return last + 1 if last >= 1 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default='UCBxTOT-gold.json')
    ap.add_argument('--artifact', required=True)
    ap.add_argument('--map', default='')
    ap.add_argument('--day', type=int, default=0)
    ap.add_argument('--out', default='logs/metrics.daily.jsonl')
    args = ap.parse_args()

    spec = json.load(open(args.spec, 'r', encoding='utf-8'))
    domains = spec.get('bars', {}).get('B1', {}).get('domains', [
        "MATH","CODE","LANG","VISION","PLAN","TOOL","RETRIEVAL","LOGIC"
    ])
    art = jload_one(args.artifact)

    mapping: Dict[str, str] = {}
    # Prefer explicit --map; else try default map path
    default_map_path = 'files/configs/domain_map.json'
    map_path = args.map or (default_map_path if os.path.exists(default_map_path) else '')
    if map_path:
        try:
            mapping = json.load(open(map_path, 'r', encoding='utf-8'))
        except Exception:
            mapping = {}

    # heuristic fallback path
    default_path = 'E2E.kpis.pass_at_1'

    row = {'day': int(args.day or next_day(args.out)), 'domain_acc': {}}
    for dom in domains:
        path = mapping.get(dom, default_path)
        val = clamp01(jget(art, path, 0.0))
        row['domain_acc'][dom] = val

    os.makedirs(os.path.dirname(args.out) or 'logs', exist_ok=True)
    with open(args.out, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
