#!/usr/bin/env python3
import math

from guardian_agi_min import qtile, WMPHead


def test_qtile_ceiling_index_basic():
    a = [1, 2, 3, 4, 5]
    assert qtile(a, 0.50) == 3.0  # ceil(0.5*5)=3 -> idx=2 -> 3
    assert qtile(a, 0.80) == 4.0  # ceil(0.8*5)=4 -> idx=3 -> 4
    assert qtile(a, 0.99) == 5.0


def test_qtile_empty_and_edges():
    assert math.isnan(qtile([], 0.95))
    a = [10]
    assert qtile(a, 0.95) == 10.0
    assert qtile(a, 0.01) == 10.0


def test_wmphead_cap_enforced():
    h = WMPHead(params_m=5.0)
    assert WMPHead.validate_caps(h) is h

    try:
        bad = WMPHead(params_m=5.0000001)
        WMPHead.validate_caps(bad)
        assert False, "Expected cap assertion"
    except AssertionError:
        pass
