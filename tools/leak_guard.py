#!/usr/bin/env python3
import argparse, json, os, sys
from typing import List, Tuple


def ngrams(s: str, n: int) -> List[str]:
    toks = s.split()
    return [" ".join(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def load_lines(path: str) -> List[str]:
    out: List[str] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(line)
    return out


def try_datasketch_minhash(strings: List[str], n: int, nperm: int) -> Tuple[object, callable]:
    try:
        from datasketch import MinHash  # type: ignore
    except Exception:
        return None, lambda s: 0.0

    def to_mh(text: str) -> "MinHash":
        mh = MinHash(num_perm=nperm)
        for g in set(ngrams(text.lower(), n)):
            mh.update(g.encode("utf-8"))
        return mh

    # Precompute train MinHash by unioning all lines
    if strings:
        train_mh = to_mh("\n".join(strings))
    else:
        train_mh = MinHash(num_perm=nperm)

    def jaccard_estimate(text: str) -> float:
        q = to_mh(text)
        return float(q.jaccard(train_mh))

    return train_mh, jaccard_estimate


def try_sentence_transformers(model_name: str):
    if not model_name:
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None

    class Encoder:
        def __init__(self, m: str):
            self._m = SentenceTransformer(m)

        def cosine(self, a: str, b: str) -> float:
            embs = self._m.encode([a, b], normalize_embeddings=True, show_progress_bar=False)
            return float(embs[0] @ embs[1])

    return Encoder(model_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="ATF manifest jsonl (new tasks)")
    ap.add_argument("--train", required=True, help="prior corpus index (text or ids)")
    ap.add_argument("--cos", type=float, default=0.90, help="cosine similarity max (embed)")
    ap.add_argument("--jac", type=float, default=0.75, help="Jaccard max (MinHash/ngram)")
    ap.add_argument("--ngram", type=int, default=5)
    ap.add_argument("--nperm", type=int, default=128)
    ap.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", ""))
    args = ap.parse_args()

    # Load corpora
    train_lines = load_lines(args.train)
    new_lines = load_lines(args.new)

    # Build MinHash estimator if possible; else fallback to exact n-gram Jaccard
    train_mh, jac_est = try_datasketch_minhash(train_lines, args.ngram, args.nperm)
    have_minhash = train_mh is not None

    # Optional embedding cosine
    enc = try_sentence_transformers(args.embed_model) if args.embed_model else None

    violations = []
    for i, line in enumerate(new_lines, 1):
        try:
            obj = json.loads(line)
            text = obj.get("text") or obj.get("prompt") or json.dumps(obj)
        except Exception:
            text = line

        # MinHash or n-gram Jaccard
        if have_minhash:
            jac = jac_est(text)
        else:
            base = "\n".join(train_lines)
            jac = jaccard(set(ngrams(text.lower(), args.ngram)), set(ngrams(base.lower(), args.ngram)) if base else set())
        minhash_ok = (jac <= args.jac)

        # Embedding cosine (optional)
        if enc:
            base_text = "\n".join(train_lines)
            cos = enc.cosine(text, base_text) if base_text else 0.0
            cos_ok = (cos <= args.cos)
        else:
            cos = 0.0
            cos_ok = True

        if not (cos_ok and minhash_ok):
            violations.append({"idx": i, "jac": float(jac), "cos": float(cos), "cos_ok": bool(cos_ok), "minhash_ok": bool(minhash_ok)})

    ok = len(violations) == 0
    out = {
        "ok": ok,
        "checked": len(new_lines),
        "violations": violations,
        "embed_checked": bool(enc is not None),
        "minhash_checked": bool(have_minhash),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
