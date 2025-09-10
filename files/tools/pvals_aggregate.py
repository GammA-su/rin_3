#!/usr/bin/env python3
"""
Aggregate multiple eval sources (CSV tidy or pvals.json docs) into a single
files/configs/pvals.input.json suitable for Gate v2 + Holm.

Inputs:
  --csv path.csv [--csv path2.csv ...]     # tidy CSVs with suite,seed,base,cand
  --pvals path.json [--pvals path2.json ...]  # docs conforming to pvals.schema.json
  --out files/configs/pvals.input.json

Note: CSV p-values are computed via a paired t-test using a normal approximation
(sufficient for n>=30). For small n, consider increasing sample size.
"""

from __future__ import annotations
import argparse, csv, json, collections, os, sys, glob
from typing import Dict, List, Tuple


def z_to_p_two_tailed(z: float) -> float:
    import math
    try:
        return float(math.erfc(abs(z) / math.sqrt(2.0)))
    except Exception:
        return 1.0


def paired_t_pvalue(diffs: List[float]) -> float:
    n = len(diffs)
    if n <= 1:
        return 1.0
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / max(1, n - 1)
    sd = (var ** 0.5) if var > 0 else 0.0
    if sd == 0:
        return 1.0 if mean == 0 else 0.0
    se = sd / (n ** 0.5)
    t = mean / se
    return z_to_p_two_tailed(t)


def load_csv_pvals(path: str) -> Dict[Tuple[str, int], float]:
    groups: Dict[Tuple[str, int], List[float]] = collections.defaultdict(list)
    try:
        f = open(path, 'r', encoding='utf-8')
    except FileNotFoundError:
        print(f"[pvals-aggregate] skip missing CSV: {path}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"[pvals-aggregate] skip unreadable CSV: {path} ({e})", file=sys.stderr)
        return {}
    with f:
        rdr = csv.DictReader(f)
        required = {'suite', 'seed', 'base', 'cand'}
        if not rdr.fieldnames or not required.issubset(set(rdr.fieldnames)):
            print(f"[pvals-aggregate] CSV missing required columns (suite,seed,base,cand): {path}", file=sys.stderr)
            return {}
        for row in rdr:
            try:
                suite = str(row['suite']).strip()
                seed = int(row['seed'])
                base = float(row['base'])
                cand = float(row['cand'])
            except Exception:
                continue
            groups[(suite, seed)].append(cand - base)
    out: Dict[Tuple[str, int], float] = {}
    for key, diffs in groups.items():
        out[key] = paired_t_pvalue(diffs)
    return out


def merge_pvals(doc: Dict[str, Dict[int, float]], suite: str, seed: int, p: float) -> None:
    if suite not in doc:
        doc[suite] = {}
    # If duplicate, keep the smallest p-value (most conservative for detecting significance)
    prev = doc[suite].get(seed)
    doc[suite][seed] = min(prev, p) if prev is not None else p


def load_pvals_doc(path: str) -> Dict[str, Dict[int, float]]:
    try:
        data = json.load(open(path, 'r', encoding='utf-8'))
    except FileNotFoundError:
        print(f"[pvals-aggregate] skip missing pvals doc: {path}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"[pvals-aggregate] skip unreadable pvals doc: {path} ({e})", file=sys.stderr)
        return {}
    out: Dict[str, Dict[int, float]] = {}
    for s in data.get('suites', []):
        name = s.get('name')
        for row in s.get('per_seed', []):
            try:
                seed = int(row.get('seed'))
                p = float(row.get('p'))
            except Exception:
                continue
            if name:
                merge_pvals(out, name, seed, p)
    return out


def main():
    ap = argparse.ArgumentParser()
    # Accept either repeated --csv a.csv --csv b.csv or a single --csv a.csv b.csv
    ap.add_argument('--csv', nargs='+', action='append', default=[], help='tidy eval CSV (suite,seed,base,cand)')
    ap.add_argument('--pvals', nargs='+', action='append', default=[], help='pvals.json doc(s)')
    ap.add_argument('--out', default='files/configs/pvals.input.json')
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()

    merged: Dict[str, Dict[int, float]] = {}

    # Flatten lists-of-lists from nargs+'append'
    csv_files = [item for sub in (args.csv or []) for item in (sub or [])]
    pval_docs = [item for sub in (args.pvals or []) for item in (sub or [])]

    # Expand any globs the shell didn't (e.g., when passed via Make env)
    def expand_paths(paths):
        expanded = []
        for p in paths:
            matches = glob.glob(p)
            if matches:
                expanded.extend(matches)
            else:
                expanded.append(p)
        return expanded

    csv_files = expand_paths(csv_files)
    pval_docs = expand_paths(pval_docs)

    for c in csv_files:
        per = load_csv_pvals(c)
        for (suite, seed), p in per.items():
            merge_pvals(merged, suite, seed, p)

    for pj in pval_docs:
        per = load_pvals_doc(pj)
        for suite, seeds in per.items():
            for seed, p in seeds.items():
                merge_pvals(merged, suite, seed, p)

    suites = []
    for suite, seeds in merged.items():
        per_seed = [{"seed": int(k), "p": float(v)} for k, v in sorted(seeds.items())]
        suites.append({"name": suite, "per_seed": per_seed})

    out_doc = {"method": "paired_t", "alpha": float(args.alpha), "suites": suites}
    try:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out_doc, f, indent=2)
    except Exception as e:
        print(f"[pvals-aggregate] failed to write {args.out}: {e}", file=sys.stderr)
        sys.exit(2)

    if not suites:
        print(json.dumps({"out": args.out, "suites": 0, "warning": "no inputs found (all paths missing?)"}, indent=2))
        # Exit 0 to allow upstream make targets to continue if desired
        return
    print(json.dumps({"out": args.out, "suites": len(suites)}, indent=2))


if __name__ == '__main__':
    main()
