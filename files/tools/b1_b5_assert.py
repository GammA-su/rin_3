#!/usr/bin/env python3
import sys, json


def main():
    if len(sys.argv) not in (2, 3):
        print("usage: b1_b5_assert.py <reports/b1_b5_report.json> [<spec.json>]", file=sys.stderr)
        sys.exit(2)
    rep = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
    gain = float(rep.get('oecg_gain_abs_pct', 0.0))
    drop = float(rep.get('drops_abs_pct_max', 999.0))
    want_gain = 10.0
    want_drop = 2.0
    if len(sys.argv) == 3:
        try:
            spec = json.load(open(sys.argv[2], 'r', encoding='utf-8'))
            b1 = spec.get('bars', {}).get('B1', {})
            want_gain = float(b1.get('delta_abs_min_pct', want_gain))
            want_drop = float(b1.get('max_domain_drop_abs_pct', want_drop))
        except Exception:
            pass
    ok = (gain >= want_gain) and (drop <= want_drop)
    if not ok:
        print(f"B1B5-ASSERT-FAIL: gain={gain} want>={want_gain} drop={drop} want<={want_drop}")
        sys.exit(1)
    print("b1b5-ok")


if __name__ == '__main__':
    main()

