#!/usr/bin/env python3
"""
Prepare the environment for strict novelty checks:
 - Installs sentence-transformers and datasketch into the current Python env
 - Loads the embed model from the spec to pre-download/cache it

Usage:
  python files/tools/novelty_setup.py --spec UCBxTOT-gold.json
"""

from __future__ import annotations
import argparse, json, sys, subprocess


def install(pkgs):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    except subprocess.CalledProcessError as e:
        print(f"[novelty-setup] pip install failed: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="UCBxTOT-gold.json")
    args = ap.parse_args()

    spec = json.load(open(args.spec, "r", encoding="utf-8"))
    model = spec.get("novelty", {}).get("embed_model", "")
    if not model:
        print("[novelty-setup] no novelty.embed_model in spec; nothing to do")
        return

    # Ensure deps
    try:
        import sentence_transformers  # type: ignore
        import datasketch  # type: ignore
        deps_ok = True
    except Exception:
        deps_ok = False
    if not deps_ok:
        print("[novelty-setup] installing sentence-transformers and datasketch ...")
        install(["sentence-transformers", "datasketch"])

    # Pre-download/cache the embed model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        print(f"[novelty-setup] downloading/caching embed model: {model}")
        SentenceTransformer(model)
        print("[novelty-setup] embed model ready")
    except Exception as e:
        print(f"[novelty-setup] failed to load embed model '{model}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

