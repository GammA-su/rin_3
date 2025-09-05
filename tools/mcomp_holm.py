#!/usr/bin/env python3
import argparse, json, sys

def holm(pvals, alpha):
    # Returns list of booleans indicating whether each hypothesis is rejected (True=reject)
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    rejects = [False]*m
    for k, i in enumerate(order):
        thresh = alpha/(m - k)
        if pvals[i] <= thresh:
            rejects[i] = True
        else:
            # Once a non-reject encountered, all larger pvals are not rejected
            break
    return rejects

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pvals', required=True, help='JSON file with per-suite, per-seed p')
    ap.add_argument('--alpha', type=float, required=True)
    args = ap.parse_args()

    data = json.load(open(args.pvals))
    ps = []
    names = []
    for s in data.get('suites', []):
        for row in s.get('per_seed', []):
            ps.append(float(row.get('p', 1.0)))
            names.append(f"{s.get('name')}#seed{row.get('seed')}")

    if not ps:
        print(json.dumps({"ok": True, "note": "no p-values provided"}))
        sys.exit(0)

    rejects = holm(ps, args.alpha)
    survivors = [names[i] for i, r in enumerate(rejects) if not r]
    ok = len(survivors) > 0
    out = {"alpha": args.alpha, "survivors": survivors, "rejected": [names[i] for i,r in enumerate(rejects) if r], "ok": ok}
    print(json.dumps(out))
    if not ok:
        sys.exit(1)

if __name__ == '__main__':
    main()

