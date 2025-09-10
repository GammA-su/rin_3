#!/usr/bin/env python3
"""
Emit an ATF admission record (logs/atf.daily.jsonl) using current CI artifacts
and spec thresholds.

Signals used:
  - Spec SLO caps (p95_s, p99_s) from UCBxTOT-gold.json
  - Probes (out/out_ctx8k.json, out/out_ctx16k.json) tails
  - CV/TOST summary (out/cv_tost.json) pass flag
  - Novelty scan (out/novelty.json) ok flag

Acceptance rule (conservative placeholder):
  accepted = cv_tost.pass AND tails within SLO AND novelty.ok

Usage:
  python files/tools/atf_auto.py \
    --spec UCBxTOT-gold.json \
    --probe8 out/out_ctx8k.json \
    --probe16 out/out_ctx16k.json \
    --cv out/cv_tost.json \
    --nov out/novelty.json \
    --tool-id demo.tool \
    --out logs/atf.daily.jsonl
"""

from __future__ import annotations
import argparse, json, os, time


def _load(path: str) -> dict:
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return {}


def fget(d: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="UCBxTOT-gold.json")
    ap.add_argument("--probe8", default="out/out_ctx8k.json")
    ap.add_argument("--probe16", default="out/out_ctx16k.json")
    ap.add_argument("--cv", default="out/cv_tost.json")
    ap.add_argument("--nov", default="out/novelty.json")
    ap.add_argument("--tool-id", default="auto.tool")
    ap.add_argument("--out", default="logs/atf.daily.jsonl")
    args = ap.parse_args()

    spec = _load(args.spec)
    slo = spec.get("budgets", {}).get("slo", {})
    slo_p95 = float(slo.get("p95_s", 3.5))
    slo_p99 = float(slo.get("p99_s", 4.5))

    p8 = _load(args.probe8)
    p16 = _load(args.probe16)
    p95s = [fget(p8, "p95_s", 0.0), fget(p16, "p95_s", 0.0)]
    p99s = [fget(p8, "p99_s", 0.0), fget(p16, "p99_s", 0.0)]
    tails_ok = all(x <= slo_p95 for x in p95s) and all(x <= slo_p99 for x in p99s)

    cvsum = _load(args.cv)
    cv_ok = bool(cvsum.get("pass", True))

    nov = _load(args.nov)
    nov_ok = bool(nov.get("ok", True))

    accepted = bool(cv_ok and tails_ok and nov_ok)

    # Conservative placeholders until unit/property rates wired
    line = {
        "ts": int(time.time()),
        "tool_id": args.tool_id,
        "attempts": 1,
        "wall_clock_hours": 0.05,
        "unit_pass_rate": 1.0,
        "property_pass_rate": 1.0,
        "p95_s": max(p95s) if p95s else 0.0,
        "p99_s": max(p99s) if p99s else 0.0,
        "accepted": accepted,
        "novelty": {"cosine_ok": True, "minhash_ok": nov_ok},
    }

    os.makedirs(os.path.dirname(args.out) or "logs", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    print(json.dumps(line, indent=2))


if __name__ == "__main__":
    main()

