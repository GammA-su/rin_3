#!/usr/bin/env python3
import sys, json, pathlib


def main():
    if len(sys.argv) != 2:
        print("usage: atf_assert.py <logs/atf.daily.jsonl>", file=sys.stderr)
        sys.exit(2)
    p = pathlib.Path(sys.argv[1])
    if not p.exists():
        print(f"ATF-ASSERT-FAIL: missing {p}", file=sys.stderr)
        sys.exit(1)
    last = None
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except Exception:
                continue
    if not last:
        print("ATF-ASSERT-FAIL: empty atf log", file=sys.stderr)
        sys.exit(1)
    if not bool(last.get('accepted', False)):
        print(f"ATF-ASSERT-FAIL: last admission not accepted: {last}")
        sys.exit(1)
    print("atf-ok")


if __name__ == '__main__':
    main()

