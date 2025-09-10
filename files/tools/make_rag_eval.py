#!/usr/bin/env python3
"""
Create a small, deterministic RAG eval artifact at the spec's path and pin it.
This avoids heavy deps (no parquet writer); content is plain text but lives at
the expected path, and we pin its sha256 for CI reproducibility.

Usage:
  python files/tools/make_rag_eval.py --spec UCBxTOT-gold.json [--rows 100] [--force]
"""

from __future__ import annotations
import argparse, hashlib, json, os, pathlib, random, sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="UCBxTOT-gold.json")
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    spec = json.load(open(args.spec, "r", encoding="utf-8"))
    rag = spec.get("rag", {}).get("eval_set", {})
    out_path = rag.get("path", "data/rag10k.parquet")
    pin_path = rag.get("sha256_from_file", "pins/rag.sha256")

    out_p = pathlib.Path(out_path)
    pin_p = pathlib.Path(pin_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    pin_p.parent.mkdir(parents=True, exist_ok=True)

    if out_p.exists() and out_p.stat().st_size > 0 and not args.force:
        # Still ensure pin exists
        if not pin_p.exists() or pin_p.stat().st_size == 0:
            pin_p.write_text(sha256_file(str(out_p)) + "\n", encoding="utf-8")
            print(f"[make-rag] wrote missing pin: {pin_p}")
        else:
            print("[make-rag] artifact exists; nothing to do")
        return

    random.seed(7)
    with open(out_p, "wb") as f:
        for i in range(max(1, int(args.rows))):
            line = {
                "id": i,
                "title": f"doc-{i}",
                "text": f"synthetic rag doc {i} seed 7 {random.randint(0, 1_000_000)}",
            }
            data = (json.dumps(line) + "\n").encode("utf-8")
            f.write(data)
    h = sha256_file(str(out_p))
    pin_p.write_text(h + "\n", encoding="utf-8")
    print(f"[make-rag] wrote {out_p} and pin {pin_p}: {h}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(2)

