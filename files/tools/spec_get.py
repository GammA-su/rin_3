#!/usr/bin/env python3
import sys, json

def get_path(doc, path):
    cur = doc
    for k in path.split('.'):
        cur = cur[k]
    return cur

def main():
    if len(sys.argv) != 3:
        print("usage: spec_get.py <json-file> <dot.path>", file=sys.stderr)
        sys.exit(2)
    doc = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
    val = get_path(doc, sys.argv[2])
    if isinstance(val, (dict, list)):
        print(json.dumps(val))
    else:
        print(str(val))

if __name__ == '__main__':
    main()

