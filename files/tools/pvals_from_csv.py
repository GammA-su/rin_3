#!/usr/bin/env python3
"""
Build a pvals.input.json (files/schemas/pvals.schema.json) from a tidy CSV of
paired scores: columns suite, seed, base, cand. Groups by (suite, seed) and
performs a paired t-test using a normal approximation (sufficient for n>=30).

Usage:
  python files/tools/pvals_from_csv.py --csv evals.csv --out files/configs/pvals.input.json

CSV example:
  suite,seed,base,cand
  demo,5,0,1
  demo,5,1,1
  demo,5,0,1
  demo,7,1,1
  demo,7,0,1
"""

from __future__ import annotations
import argparse, csv, json, math, collections, sys, pathlib


def z_to_p_two_tailed(z: float) -> float:
    # normal approximation; p = 2 * (1 - Phi(|z|)) = erfc(|z|/sqrt(2))
    try:
        import math
        return float(math.erfc(abs(z) / math.sqrt(2.0)))
    except Exception:
        return 1.0


def paired_t_pvalue(diffs: list[float]) -> float:
    n = len(diffs)
    if n <= 1:
        return 1.0
    mean = sum(diffs) / n
    # sample std dev
    var = sum((x - mean) ** 2 for x in diffs) / max(1, n - 1)
    sd = var ** 0.5
    if sd == 0:
        # If all diffs equal, zero variance: if mean==0 => no effect; else very small p
        return 1.0 if mean == 0 else 0.0
    se = sd / (n ** 0.5)
    t = mean / se
    # normal approximation for tail probability
    return z_to_p_two_tailed(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='files/configs/pvals.input.json')
    args = ap.parse_args()

    groups: dict[tuple[str, int], list[float]] = collections.defaultdict(list)
    with open(args.csv, 'r', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        required = {'suite', 'seed', 'base', 'cand'}
        missing = required - set(rdr.fieldnames or [])
        if missing:
            print(f"missing CSV columns: {', '.join(sorted(missing))}", file=sys.stderr)
            sys.exit(2)
        for row in rdr:
            try:
                suite = str(row['suite']).strip()
                seed = int(row['seed'])
                base = float(row['base'])
                cand = float(row['cand'])
            except Exception:
                continue
            groups[(suite, seed)].append(cand - base)

    suites_out = []
    by_suite: dict[str, list[dict]] = collections.defaultdict(list)
    for (suite, seed), diffs in groups.items():
        p = paired_t_pvalue(diffs)
        by_suite[suite].append({'seed': int(seed), 'p': float(max(0.0, min(1.0, p)))})
    for suite, items in by_suite.items():
        # sort by seed for readability
        items.sort(key=lambda r: r['seed'])
        suites_out.append({'name': suite, 'per_seed': items})

    doc = {'method': 'paired_t', 'alpha': 0.05, 'suites': suites_out}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, 'w', encoding='utf-8').write(json.dumps(doc, indent=2))
    print(json.dumps({'out': args.out, 'suites': len(suites_out)}, indent=2))


if __name__ == '__main__':
    main()

