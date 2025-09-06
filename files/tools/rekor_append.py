#!/usr/bin/env python3
"""
Append a transparency-log entry with a hash chain to logs/rekor-local.jsonl.

Usage:
  python files/tools/rekor_append.py logs/rekor-local.jsonl out/payload.hash

Where 'out/payload.hash' contains a line like: 'sha256:<hex>'.
"""

from __future__ import annotations
import sys, json, time, hashlib, pathlib


def last_chain(path: str) -> tuple[int, str]:
    ts = -1
    chain = "0" * 64
    p = pathlib.Path(path)
    if not p.exists():
        return ts, chain
    try:
        with p.open("r", encoding="utf-8") as f:
            last = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                last = line
            if last:
                e = json.loads(last)
                ts = int(e.get("timestamp", -1))
                chain = e.get("prev_chain", "0" * 64)
                # recompute head from last entry for safety
                pay = e.get("payload_hash", "")
                h = hashlib.sha256((chain + pay + str(ts)).encode()).hexdigest()
                chain = h
    except Exception:
        pass
    return ts, chain


def load_payload_hash(path: str) -> str:
    text = pathlib.Path(path).read_text(encoding="utf-8").strip()
    if not text.startswith("sha256:"):
        raise SystemExit(f"payload hash must start with 'sha256:', got: {text[:16]}...")
    return text


def main():
    if len(sys.argv) != 3:
        print("usage: rekor_append.py <log.jsonl> <payload.hash>", file=sys.stderr)
        sys.exit(2)
    log_path = sys.argv[1]
    payload_hash = load_payload_hash(sys.argv[2])
    _, prev_head = last_chain(log_path)
    ts = int(time.time())
    entry = {"timestamp": ts, "prev_chain": prev_head, "payload_hash": payload_hash}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps({"appended": True, "ts": ts}))


if __name__ == "__main__":
    main()

