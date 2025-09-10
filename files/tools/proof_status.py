#!/usr/bin/env python3
"""
Summarize current proof status across novelty, probes/tails, CV/TOST, pins, council,
ledger, ATF, and B1–B5.

Outputs a single JSON summary to stdout and writes out/proof_status.json.
"""

from __future__ import annotations
import json, pathlib, sys
from typing import Any, Dict


def load(path: str) -> dict:
    p = pathlib.Path(path)
    try:
        if p.suffix == ".jsonl":
            # return last non-empty JSON line
            last = {}
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except Exception:
                    pass
            return last
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_spec(path: str = "UCBxTOT-gold.json") -> dict:
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def main():
    spec = get_spec()
    slo = spec.get("budgets", {}).get("slo", {})
    p95_max = float(slo.get("p95_s", 3.5))
    p99_max = float(slo.get("p99_s", 4.5))

    # Novelty
    nov = load("out/novelty.json")
    novelty_ok = bool(nov.get("ok", False))
    embed_checked = bool(nov.get("embed_checked", False))
    minhash_checked = bool(nov.get("minhash_checked", False))
    require_both = bool(spec.get("novelty", {}).get("require_both", False))
    novelty_checks_ok = novelty_ok and (not require_both or (embed_checked and minhash_checked))

    # Probes & tails
    p8 = load("out/out_ctx8k.json")
    p16 = load("out/out_ctx16k.json")
    tails_ok = (
        float(p8.get("p95_s", 0)) <= p95_max
        and float(p16.get("p95_s", 0)) <= p95_max
        and float(p8.get("p99_s", 0)) <= p99_max
        and float(p16.get("p99_s", 0)) <= p99_max
    )

    # CV/TOST
    cvt = load("out/cv_tost.json")
    cvt_ok = bool(cvt.get("pass", False))

    # Pins/policy
    pins = load("out/policy_scan.json")
    pins_ok = bool(pins.get("ok", False))

    # Council
    council = load("out/council.json")
    council_ok = bool(council.get("ok", False))

    # Ledger
    ledger = load("out/ledger_check.json")
    ledger_ok = bool(ledger.get("ok", True))

    # ATF
    atf = load("logs/atf.daily.jsonl")
    atf_ok = bool(atf.get("accepted", False))

    # B1–B5
    b = load("reports/b1_b5_report.json")
    b_gain = float(b.get("oecg_gain_abs_pct", 0.0))
    b_drop = float(b.get("drops_abs_pct_max", 99.9))
    bars_ok = (b_gain >= 10.0) and (b_drop <= 2.0)

    summary: Dict[str, Any] = {
        "novelty": {
            "ok": novelty_ok,
            "embed_checked": embed_checked,
            "minhash_checked": minhash_checked,
            "require_both": require_both,
            "checks_ok": novelty_checks_ok,
        },
        "tails": {
            "p95_s": {"ctx8k": p8.get("p95_s"), "ctx16k": p16.get("p95_s"), "limit": p95_max},
            "p99_s": {"ctx8k": p8.get("p99_s"), "ctx16k": p16.get("p99_s"), "limit": p99_max},
            "ok": bool(tails_ok),
        },
        "cv_tost": cvt,
        "pins_ok": pins_ok,
        "council_ok": council_ok,
        "ledger_ok": ledger_ok,
        "atf_last": atf,
        "b1_b5": {"oecg_gain_abs_pct": b_gain, "drops_abs_pct_max": b_drop, "ok": bars_ok},
    }
    summary["overall_ok"] = bool(
        novelty_checks_ok and tails_ok and cvt_ok and pins_ok and council_ok and ledger_ok and atf_ok
    )

    outp = pathlib.Path("out/proof_status.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

