#!/usr/bin/env python3
"""
Seed logs/metrics.daily.jsonl with a deterministic, modestly improving
set of domain accuracies across the 8 OECG domains. Intended only to
unblock CI artifacts and schema checks when real metrics are not yet
wired up.

Usage:
  python files/tools/metrics_seed.py --days 14 --out logs/metrics.daily.jsonl
"""

from __future__ import annotations
import argparse, json, os

DOMS = ["MATH","CODE","LANG","VISION","PLAN","TOOL","RETRIEVAL","LOGIC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--out", default="logs/metrics.daily.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or "logs", exist_ok=True)
    # Start each domain at a slightly different base, then increase modestly
    bases = {
        "MATH": 0.62,
        "CODE": 0.58,
        "LANG": 0.70,
        "VISION": 0.55,
        "PLAN": 0.60,
        "TOOL": 0.57,
        "RETRIEVAL": 0.63,
        "LOGIC": 0.59,
    }
    inc = 0.005  # ~0.5% per day
    with open(args.out, "w", encoding="utf-8") as f:
        for day in range(1, int(args.days) + 1):
            row = {"day": day, "domain_acc": {}}
            for d in DOMS:
                val = min(0.99, bases[d] + inc * (day - 1))
                row["domain_acc"][d] = round(float(val), 4)
            f.write(json.dumps(row) + "\n")
    print(json.dumps({"out": args.out, "days": int(args.days)}, indent=2))


if __name__ == "__main__":
    main()

