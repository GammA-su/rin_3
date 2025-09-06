#!/usr/bin/env python3
"""
ATF admission record writer (schema-compliant).

Creates one JSONL line in logs/atf.daily.jsonl reflecting a tool admission
attempt under caps and tails. This is a lightweight generator to make CI
artifacts concrete; integrate your real ATF manager later.

Usage:
  python files/tools/atf_admit.py \
    --tool-id demo.tool \
    --attempts 1 \
    --wall 0.25 \
    --p95 0.8 --p99 1.2 \
    --unit 1.0 --prop 1.0 \
    --cosine-ok  true --minhash-ok true \
    --accepted true
"""

from __future__ import annotations
import argparse, json, os, time


def btoi(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool-id", default="demo.tool")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--wall", type=float, default=0.25, help="wall_clock_hours")
    ap.add_argument("--unit", type=float, default=1.0, help="unit pass rate [0..1]")
    ap.add_argument("--prop", type=float, default=1.0, help="property pass rate [0..1]")
    ap.add_argument("--p95", type=float, default=0.8)
    ap.add_argument("--p99", type=float, default=1.2)
    ap.add_argument("--cosine-ok", default="true")
    ap.add_argument("--minhash-ok", default="true")
    ap.add_argument("--accepted", default="true")
    ap.add_argument("--reason", default="")
    ap.add_argument("--out", default="logs/atf.daily.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or "logs", exist_ok=True)
    line = {
        "ts": int(time.time()),
        "tool_id": args.tool_id,
        "attempts": int(args.attempts),
        "wall_clock_hours": float(args.wall),
        "unit_pass_rate": float(max(0.0, min(1.0, args.unit))),
        "property_pass_rate": float(max(0.0, min(1.0, args.prop))),
        "p95_s": float(args.p95),
        "p99_s": float(args.p99),
        "accepted": btoi(args.accepted),
        "novelty": {
            "cosine_ok": btoi(args.cosine_ok),
            "minhash_ok": btoi(args.minhash_ok)
        }
    }
    if args.reason:
        line["reason"] = args.reason
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    print(json.dumps(line, indent=2))


if __name__ == "__main__":
    main()

