#!/usr/bin/env python3
"""
Build pvals.input.json from an evaluation sources manifest.

Manifest format (JSON):
{
  "csvs": ["evals_seed5.csv", ...],
  "pvals": ["pvals_a.json", ...],
  "artifact_to_csv": [
     {"artifact":"artifacts/suite_full.json","suite":"my-suite","seed":7,
      "base_col":"E2E.kpis.pass_at_1","cand_col":"E2E.kpis.pass_at_1","out":"evals_artifact_seed7.csv"}
  ]
}

Writes files/configs/pvals.input.json by default.
"""

from __future__ import annotations
import argparse, json, pathlib, csv
from typing import Any, Dict, List


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
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            continue
    return {}


def write_tidy_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    outp = pathlib.Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["suite","seed","base","cand"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="files/configs/eval_sources.json")
    ap.add_argument("--out", default="files/configs/pvals.input.json")
    args = ap.parse_args()

    m = json.load(open(args.manifest, "r", encoding="utf-8"))

    # Convert any artifact->csv entries
    for entry in m.get("artifact_to_csv", []):
        art = str(entry.get("artifact"))
        suite = str(entry.get("suite"))
        seed = int(entry.get("seed"))
        base_col = str(entry.get("base_col"))
        cand_col = str(entry.get("cand_col"))
        out_csv = str(entry.get("out") or f"evals_{suite}_seed{seed}.csv")
        obj = load_one(art)
        flat = flatten(obj)
        # This builds a single-row tidy CSV unless you point base/cand to arrays
        try:
            base_v = float(flat.get(base_col, 0.0))
            cand_v = float(flat.get(cand_col, base_v))
        except Exception:
            base_v = 0.0
            cand_v = 0.0
        write_tidy_csv([{"suite": suite, "seed": seed, "base": base_v, "cand": cand_v}], out_csv)

    # Aggregate
    csvs = m.get("csvs", []) + [str(entry.get("out")) for entry in m.get("artifact_to_csv", []) if entry.get("out")]
    pvals = m.get("pvals", [])

    # Reuse aggregator logic by loading modules here
    from pvals_aggregate import load_csv_pvals, load_pvals_doc  # type: ignore
    merged: Dict[str, Dict[int, float]] = {}

    def merge(doc: Dict[str, Dict[int, float]], suite: str, seed: int, p: float) -> None:
        if suite not in doc:
            doc[suite] = {}
        prev = doc[suite].get(seed)
        doc[suite][seed] = min(prev, p) if prev is not None else p

    for c in csvs:
        per = load_csv_pvals(c)
        for (suite, seed), p in per.items():
            merge(merged, suite, seed, p)
    for pj in pvals:
        per = load_pvals_doc(pj)
        for suite, seeds in per.items():
            for seed, p in seeds.items():
                merge(merged, suite, seed, p)

    suites = []
    for suite, seeds in merged.items():
        per_seed = [{"seed": int(k), "p": float(v)} for k, v in sorted(seeds.items())]
        suites.append({"name": suite, "per_seed": per_seed})

    out_doc = {"method": "paired_t", "alpha": 0.05, "suites": suites}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(json.dumps(out_doc, indent=2))
    print(json.dumps({"out": args.out, "suites": len(suites)}, indent=2))


if __name__ == "__main__":
    main()

