#!/usr/bin/env python3
"""
Execute CI assertions from the spec (UCBxTOT-gold.json).

Supports the following assertion forms in spec.ci_assertions:
  - require_file <paths...>
  - eq <json.path.a> <json.path.b>
  - energy_mode_consistent <cal.json> <probe.json> [<probe.json> ...]
  - ledger_append_only <rekor.jsonl>

Exit non-zero on first failure.
"""
from __future__ import annotations
import argparse, json, subprocess, sys


def run(cmd: list[str]) -> None:
    print("[assert]", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="UCBxTOT-gold.json")
    args = ap.parse_args()

    spec = json.load(open(args.spec, "r", encoding="utf-8"))
    items = spec.get("ci_assertions", [])
    for raw in items:
        if not isinstance(raw, str):
            continue
        toks = raw.strip().split()
        if not toks:
            continue
        kind, rest = toks[0], toks[1:]
        if kind == "require_file":
            run([sys.executable, "files/tools/require_file.py", *rest])
        elif kind == "eq" and len(rest) == 2:
            run([sys.executable, "files/tools/json_eq.py", args.spec, rest[0], rest[1]])
        elif kind == "energy_mode_consistent" and len(rest) >= 3:
            run([sys.executable, "files/tools/energy_mode_assert.py", *rest])
        elif kind == "ledger_append_only" and len(rest) == 1:
            run([sys.executable, "files/tools/ledger_check.py", rest[0]])
        elif kind == "novelty_ok":
            # novelty_ok <novelty.json> [<spec.json>]
            if len(rest) == 0:
                print("[warn] novelty_ok missing args; expected novelty.json [spec.json]")
                continue
            cmd = [sys.executable, "files/tools/novelty_assert.py", *rest]
            run(cmd)
        elif kind == "atf_ok" and len(rest) == 1:
            run([sys.executable, "files/tools/atf_assert.py", rest[0]])
        elif kind == "b1_b5_ok":
            # b1_b5_ok <report.json> [<spec.json>]
            if len(rest) == 0:
                print("[warn] b1_b5_ok missing args; expected report.json [spec.json]")
                continue
            run([sys.executable, "files/tools/b1_b5_assert.py", *rest])
        else:
            print(f"[warn] unsupported assertion: {raw}")


if __name__ == "__main__":
    main()
