#!/usr/bin/env python3
import sys, json

def jload(p):
    return json.load(open(p, 'r', encoding='utf-8'))

def main():
    if len(sys.argv) not in (2, 3):
        print("usage: novelty_assert.py <novelty.json> [<spec.json>]", file=sys.stderr)
        sys.exit(2)
    doc = jload(sys.argv[1])
    ok = bool(doc.get('ok', False))
    embed_checked = bool(doc.get('embed_checked', False))
    minhash_checked = bool(doc.get('minhash_checked', False))

    require_both = False
    if len(sys.argv) == 3:
        try:
            spec = jload(sys.argv[2])
            require_both = bool(spec.get('novelty', {}).get('require_both', False))
        except Exception:
            require_both = False

    if require_both and (not embed_checked or not minhash_checked):
        print(f"NOVELTY-ASSERT-FAIL: require_both set but checks missing: embed_checked={embed_checked} minhash_checked={minhash_checked}")
        sys.exit(1)

    if not ok:
        print(f"NOVELTY-ASSERT-FAIL: {doc}")
        sys.exit(1)
    print("novelty-ok")

if __name__ == '__main__':
    main()
