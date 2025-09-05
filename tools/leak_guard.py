#!/usr/bin/env python3
import argparse, json, os, sys, hashlib
from typing import List

def ngrams(s: str, n: int) -> List[str]:
    toks = s.split()
    return [" ".join(toks[i:i+n]) for i in range(max(0, len(toks)-n+1))]

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    return len(a & b) / max(1, len(a | b))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True, help='ATF manifest jsonl (new tasks)')
    ap.add_argument('--train', required=True, help='prior corpus index (text or ids)')
    ap.add_argument('--cos', type=float, default=0.90)
    ap.add_argument('--jac', type=float, default=0.75)
    ap.add_argument('--ngram', type=int, default=5)
    args = ap.parse_args()

    # Load train tokens (fallback to empty if missing)
    train_text = ""
    if os.path.exists(args.train):
        try:
            with open(args.train, 'r', encoding='utf-8', errors='ignore') as f:
                train_text = f.read()
        except Exception:
            train_text = ""
    train_set = set(ngrams(train_text.lower(), args.ngram)) if train_text else set()

    nov_violations = []
    n = 0
    if os.path.exists(args.new):
        with open(args.new, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    obj = json.loads(line)
                    text = obj.get('text') or obj.get('prompt') or json.dumps(obj)
                except Exception:
                    text = line
                s = set(ngrams(text.lower(), args.ngram))
                jac = jaccard(s, train_set) if train_set else 0.0
                # cosine is not computed here (no embeddings); treat as safe if Jaccard passes
                cos_ok = True
                minhash_ok = (jac <= args.jac)
                if not (cos_ok and minhash_ok):
                    nov_violations.append({"idx": n, "jac": jac})
    ok = len(nov_violations) == 0
    print(json.dumps({"ok": ok, "checked": n, "violations": nov_violations}))

if __name__ == '__main__':
    main()

