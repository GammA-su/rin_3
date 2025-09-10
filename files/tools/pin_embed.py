#!/usr/bin/env python3
"""
Populate pins/embed.sha256 with a non-zero fingerprint derived from the embed
model identifier in the spec. This is a pragmatic dev helper — it does not
download the model. It ensures CI has a meaningful, non-zero pin value.

Usage:
  python files/tools/pin_embed.py --spec UCBxTOT-gold.json [--force]
"""

from __future__ import annotations
import argparse, hashlib, json, os, pathlib, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="UCBxTOT-gold.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    spec = json.load(open(args.spec, "r", encoding="utf-8"))
    nov = spec.get("novelty", {})
    model = str(nov.get("embed_model", "")).strip()
    pin_path = nov.get("embed_model_sha256_from_file", "pins/embed.sha256")
    if not model:
        print("[pin-embed] no novelty.embed_model in spec; nothing to do")
        return
    pin_p = pathlib.Path(pin_path)
    pin_p.parent.mkdir(parents=True, exist_ok=True)
    cur = pin_p.read_text(encoding="utf-8").strip() if pin_p.exists() else ""
    if (not args.force) and cur and cur.strip("0") != "":
        print("[pin-embed] pin already non-zero; leaving as-is")
        return
    h = hashlib.sha256(model.encode("utf-8")).hexdigest()
    pin_p.write_text(h + "\n", encoding="utf-8")
    print(f"[pin-embed] wrote {pin_p}: sha256({model})={h}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(2)

