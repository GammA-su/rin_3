#!/usr/bin/env python3
"""
Scan an eval artifact for scalar metric paths and optionally write a domain map
for B1–B5 reporting.

Usage:
  # List candidate metric paths
  python files/tools/domain_map_wizard.py --artifact artifacts/suite_full.json

  # Set one path for all domains
  python files/tools/domain_map_wizard.py --artifact artifacts/suite_full.json \
    --set-all E2E.kpis.pass_at_1 --out files/configs/domain_map.json

  # Set per-domain via CLI (comma-separated DOMAIN=path)
  python files/tools/domain_map_wizard.py --artifact artifacts/suite_full.json \
    --set MATH=E2E.kpis.pass_at_1,CODE=E2E.kpis.pass_at_1
"""

from __future__ import annotations
import argparse, json, pathlib, sys
from typing import Any, Dict, List, Tuple


def flatten(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key, sep))
        else:
            out[key] = v
    return out


def load_one(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj:
            return obj[0]
    except Exception:
        pass
    # JSONL fallback: take first object
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            continue
    return {}


def parse_set_arg(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not s:
        return out
    for tok in s.split(","):
        if not tok.strip():
            continue
        if "=" not in tok:
            raise SystemExit(f"bad mapping token '{tok}', expected DOMAIN=path")
        dom, path = tok.split("=", 1)
        out[dom.strip()] = path.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", default="files/configs/domain_map.json")
    ap.add_argument("--set-all", default="", help="Set the same metric path for all domains")
    ap.add_argument("--set", default="", help="Comma-separated DOMAIN=path pairs to set per-domain")
    ap.add_argument("--domains", default="MATH,CODE,LANG,VISION,PLAN,TOOL,RETRIEVAL,LOGIC")
    args = ap.parse_args()

    obj = load_one(args.artifact)
    flat = flatten(obj)
    # Print all scalar float-like paths with examples
    candidates: List[Tuple[str, str]] = []
    for k, v in flat.items():
        try:
            float(v)
            candidates.append((k, str(v)))
        except Exception:
            continue

    print("# Candidate numeric metric paths:")
    for k, v in sorted(candidates):
        print(f"{k}\t{v}")

    # Build mapping if requested
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    mapping: Dict[str, str] = {}
    if args.set_all:
        for d in domains:
            mapping[d] = args.set_all
    if args.set:
        mapping.update(parse_set_arg(args.set))

    if mapping:
        outp = pathlib.Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(mapping, indent=2))
        print(f"[domain-map] wrote {outp}")


if __name__ == "__main__":
    main()

