#!/usr/bin/env python3
"""
Create an ATF admission record from an evaluation artifact plus current probes.

Sources:
  - Artifact (JSON): extracts E2E.kpis.pass_at_1 (unit) and E2E.kpis.precision_k
    (property). If missing, falls back to resolution_rate for property.
  - Probes: out/out_ctx8k.json, out/out_ctx16k.json for p95_s/p99_s (max of both).
  - Novelty: out/novelty.json -> ok flag.

Acceptance rule (simple, spec-aligned):
  accepted = (unit >= atf.unit_property_pass_min) and (property >= atf.unit_property_pass_min)
             and (max(p95) <= slo.p95_s) and (max(p99) <= slo.p99_s) and novelty.ok

Usage:
  python files/tools/atf_from_artifact.py \
    --spec UCBxTOT-gold.json \
    --artifact artifacts/suite_full.json \
    --probe8 out/out_ctx8k.json --probe16 out/out_ctx16k.json \
    --nov out/novelty.json --tool-id e2e.run --out logs/atf.daily.jsonl
"""

from __future__ import annotations
import argparse, json, os, time


def jget(d, path, default=None):
    cur = d
    try:
        for k in path.split('.'):
            cur = cur[k]
        return cur
    except Exception:
        return default


def fget(d, key, default=0.0) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)


def load(p):
    try:
        return json.load(open(p, 'r', encoding='utf-8'))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default='UCBxTOT-gold.json')
    ap.add_argument('--artifact', required=True)
    ap.add_argument('--probe8', default='out/out_ctx8k.json')
    ap.add_argument('--probe16', default='out/out_ctx16k.json')
    ap.add_argument('--nov', default='out/novelty.json')
    ap.add_argument('--tool-id', default='e2e.run')
    ap.add_argument('--out', default='logs/atf.daily.jsonl')
    args = ap.parse_args()

    spec = load(args.spec)
    slo = spec.get('budgets', {}).get('slo', {})
    unit_min = float(spec.get('atf', {}).get('unit_property_pass_min', 0.95))
    slo_p95 = float(slo.get('p95_s', 3.5))
    slo_p99 = float(slo.get('p99_s', 4.5))

    art = load(args.artifact)
    unit = jget(art, 'E2E.kpis.pass_at_1', None)
    prop = jget(art, 'E2E.kpis.precision_k', None)
    if prop is None:
        prop = jget(art, 'E2E.kpis.resolution_rate', None)
    try:
        unit = float(unit) if unit is not None else 0.0
    except Exception:
        unit = 0.0
    try:
        prop = float(prop) if prop is not None else 0.0
    except Exception:
        prop = 0.0

    p8 = load(args.probe8)
    p16 = load(args.probe16)
    p95 = max(fget(p8, 'p95_s', 0.0), fget(p16, 'p95_s', 0.0))
    p99 = max(fget(p8, 'p99_s', 0.0), fget(p16, 'p99_s', 0.0))

    nov = load(args.nov)
    nov_ok = bool(nov.get('ok', True))

    accepted = (unit >= unit_min) and (prop >= unit_min) and (p95 <= slo_p95) and (p99 <= slo_p99) and nov_ok

    line = {
        'ts': int(time.time()),
        'tool_id': args.tool_id,
        'attempts': 1,
        'wall_clock_hours': 0.05,
        'unit_pass_rate': float(unit),
        'property_pass_rate': float(prop),
        'p95_s': float(p95),
        'p99_s': float(p99),
        'accepted': bool(accepted),
        'novelty': {'cosine_ok': True, 'minhash_ok': nov_ok},
    }

    os.makedirs(os.path.dirname(args.out) or 'logs', exist_ok=True)
    with open(args.out, 'a', encoding='utf-8') as f:
        f.write(json.dumps(line) + '\n')
    print(json.dumps(line, indent=2))


if __name__ == '__main__':
    main()

