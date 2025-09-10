#!/usr/bin/env python3
import sys, json

def get(doc, path):
    cur = doc
    for k in path.split('.'):
        cur = cur[k]
    return cur

def main():
    if len(sys.argv) != 4:
        print("usage: json_eq.py <json-file> <path.a> <path.b>", file=sys.stderr)
        sys.exit(2)
    doc = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
    a = get(doc, sys.argv[2])
    b = get(doc, sys.argv[3])
    if a != b:
        print(f"EQ-FAIL: {sys.argv[2]} != {sys.argv[3]} :: {a} != {b}")
        sys.exit(1)
    print("eq-ok")

if __name__ == '__main__':
    main()

