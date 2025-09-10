#!/usr/bin/env python3
import sys, os, pathlib

def main():
    if len(sys.argv) < 2:
        print("usage: require_file.py <path> [<path> ...]", file=sys.stderr)
        sys.exit(2)
    missing = []
    for p in sys.argv[1:]:
        path = pathlib.Path(p)
        if not (path.exists() and path.is_file() and path.stat().st_size > 0):
            missing.append(p)
    if missing:
        print("MISSING:", ", ".join(missing))
        sys.exit(1)
    print("require-file-ok")

if __name__ == "__main__":
    main()

