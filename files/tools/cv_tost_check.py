#!/usr/bin/env python3
"""
Compute coarse CV and TOST-equivalence checks over probe summaries.

Inputs:
  --probe <file1> <file2> ...   JSON files with keys: p95_s, p99_s, j_per_inf
  --cv    '{"p95":7,"p99":10,"j":6}' OR 'p95=7,p99=10,j=6' (optional)
  --tost  '{"p95":0.25,"p99":0.35,"j_pct":4}' OR 'p95=0.25,p99=0.35,j_pct=4' (optional)
  --spec  UCBxTOT-gold.json (optional; used if --cv/--tost omitted)

Output:
  Writes 'out/cv_tost.json' and prints a JSON summary to stdout.

Notes:
  - Uses aggregate metrics from each probe. CV is computed across the set of
    probes provided (usually two: ctx8k and ctx16k). With <2 probes, CV checks
    are treated as passing (insufficient data).
  - TOST-equivalence is approximated as absolute difference <= delta for p95/p99
    and percent difference <= j_pct for energy.
  - If --cv/--tost are not provided, thresholds are read from spec.budgets
    (stability_cv_caps_pct, tost_equivalence).
"""

from __future__ import annotations
import argparse, json, math, os, pathlib
import numpy as np


def _load(p: str) -> dict:
    return json.load(open(p, "r", encoding="utf-8"))


def _cv_pct(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    if mean == 0.0:
        return 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    sd = (var ** 0.5) if var > 0 else 0.0
    return 100.0 * sd / abs(mean)


def _load_samples(meta_path: str) -> dict | None:
    # meta may include a pointer to samples file; else try replacing suffix
    try:
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
    except Exception:
        return None
    sp = meta.get("samples_file")
    if sp and os.path.exists(sp):
        try:
            return json.load(open(sp, "r", encoding="utf-8"))
        except Exception:
            return None
    # Fallback to sibling path
    base, ext = os.path.splitext(meta_path)
    cand = f"{base}.samples.json"
    if os.path.exists(cand):
        try:
            return json.load(open(cand, "r", encoding="utf-8"))
        except Exception:
            return None
    return None


def _parse_kv_or_json(s: str | None) -> dict | None:
    if not s:
        return None
    st = s.strip()
    if not st:
        return None
    # JSON path
    if st.startswith("{"):
        return json.loads(st)
    # key=value,key=value fallback
    out: dict[str, float] = {}
    for tok in st.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"bad token '{tok}', expected key=value")
        k, v = tok.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def _load_spec_thresholds(spec_path: str | None) -> tuple[dict, dict]:
    if not spec_path:
        return ({"p95": 7, "p99": 10, "j": 6}, {"p95": 0.25, "p99": 0.35, "j_pct": 4})
    try:
        spec = json.load(open(spec_path, "r", encoding="utf-8"))
        cv = spec.get("budgets", {}).get("stability_cv_caps_pct", {})
        tost = spec.get("budgets", {}).get("tost_equivalence", {})
        cv_out = {"p95": float(cv.get("p95", 7)), "p99": float(cv.get("p99", 10)), "j": float(cv.get("j_per_inf", 6))}
        to_out = {"p95": float(tost.get("p95_s_delta", 0.25)), "p99": float(tost.get("p99_s_delta", 0.35)), "j_pct": float(tost.get("j_per_inf_pct", 4))}
        return cv_out, to_out
    except Exception:
        return ({"p95": 7, "p99": 10, "j": 6}, {"p95": 0.25, "p99": 0.35, "j_pct": 4})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", nargs="+", required=True)
    ap.add_argument("--cv", default="")
    ap.add_argument("--tost", default="")
    ap.add_argument("--spec", default="")
    ap.add_argument("--out", default="out/cv_tost.json")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap replicates for quantile TOST if samples are available")
    args = ap.parse_args()

    # thresholds: parse flags if provided; else load from --spec
    cv_caps = _parse_kv_or_json(args.cv)
    tost = _parse_kv_or_json(args.tost)
    if cv_caps is None or tost is None:
        spec_cv, spec_tost = _load_spec_thresholds(args.spec)
        cv_caps = cv_caps or spec_cv
        tost = tost or spec_tost

    p95s, p99s, js = [], [], []
    j_samples = []  # optional per-probe energy arrays
    for path in args.probe:
        d = _load(path)
        if "p95_s" in d:
            p95s.append(float(d.get("p95_s", 0.0)))
        if "p99_s" in d:
            p99s.append(float(d.get("p99_s", 0.0)))
        if "j_per_inf" in d and d.get("j_per_inf") is not None:
            js.append(float(d.get("j_per_inf", 0.0)))
        s = _load_samples(path)
        if s and isinstance(s.get("j"), list):
            try:
                arr = [float(x) for x in s.get("j")]
                if arr:
                    j_samples.append(arr)
            except Exception:
                pass

    cv_p95 = _cv_pct(p95s)
    cv_p99 = _cv_pct(p99s)
    cv_j = _cv_pct(js)
    cv_ok = {
        "p95_ok": (cv_p95 <= float(cv_caps.get("p95", 7))),
        "p99_ok": (cv_p99 <= float(cv_caps.get("p99", 10))),
        "j_ok": (cv_j <= float(cv_caps.get("j", 6))),
    }

    tost_p95_ok = True
    tost_p99_ok = True
    tost_j_ok = True
    if len(p95s) >= 2:
        # compare first two probes; conservative for >2
        tost_p95_ok = (abs(p95s[0] - p95s[1]) <= float(tost.get("p95", 0.25)))
    if len(p99s) >= 2:
        tost_p99_ok = (abs(p99s[0] - p99s[1]) <= float(tost.get("p99", 0.35)))
    if len(j_samples) >= 2:
        # TOST for percent difference of means with normal approx at 90% CI
        a, b = j_samples[0], j_samples[1]
        n1, n2 = len(a), len(b)
        m1 = sum(a) / n1
        m2 = sum(b) / n2
        v1 = sum((x - m1) ** 2 for x in a) / max(1, n1 - 1)
        v2 = sum((x - m2) ** 2 for x in b) / max(1, n2 - 1)
        se = (v1 / n1 + v2 / n2) ** 0.5
        # convert to percent diff relative to m1
        if abs(m1) < 1e-9 or se == 0.0:
            tost_j_ok = True  # degenerate case; treat as ok
        else:
            diff = m2 - m1
            z = 1.6448536269514722  # 90% two one-sided
            lo = diff - z * se
            hi = diff + z * se
            lo_pct = 100.0 * lo / abs(m1)
            hi_pct = 100.0 * hi / abs(m1)
            thr = float(tost.get("j_pct", 4))
            tost_j_ok = (lo_pct >= -thr and hi_pct <= thr)
    elif len(js) >= 2 and js[0] != 0:
        pct = 100.0 * abs(js[1] - js[0]) / max(1e-9, abs(js[0]))
        tost_j_ok = (pct <= float(tost.get("j_pct", 4)))

    # If latency samples exist and boot>0, refine p95/p99 TOST using bootstrap CI of deltas
    if args.boot > 0 and len(j_samples) >= 0:
        # try to load latency arrays from sample files parallel to energy samples detection
        lat_samples = []
        for path in args.probe:
            s = _load_samples(path)
            if s and isinstance(s.get("lat_s"), list) and s.get("lat_s"):
                lat_samples.append(np.asarray(s.get("lat_s"), dtype=float))
        if len(lat_samples) >= 2:
            a, b = lat_samples[0], lat_samples[1]
            n = int(args.boot)
            rng = np.random.default_rng(7)
            idx_a = rng.integers(0, len(a), size=(n, len(a)))
            idx_b = rng.integers(0, len(b), size=(n, len(b)))
            # bootstrap quantiles
            qa = np.quantile(a[idx_a], 0.95, axis=1)
            qb = np.quantile(b[idx_b], 0.95, axis=1)
            diff = qb - qa
            lo, hi = np.quantile(diff, [0.05, 0.95])
            tost_p95_ok = (abs(lo) <= float(tost.get("p95", 0.25))) and (abs(hi) <= float(tost.get("p95", 0.25)))
            # p99
            qa99 = np.quantile(a[idx_a], 0.99, axis=1)
            qb99 = np.quantile(b[idx_b], 0.99, axis=1)
            diff99 = qb99 - qa99
            lo99, hi99 = np.quantile(diff99, [0.05, 0.95])
            tost_p99_ok = (abs(lo99) <= float(tost.get("p99", 0.35))) and (abs(hi99) <= float(tost.get("p99", 0.35)))

    out = {
        "cv": {
            "p95_cv_pct": cv_p95,
            "p99_cv_pct": cv_p99,
            "j_cv_pct": cv_j,
            **cv_ok,
        },
        "tost": {
            "p95_ok": bool(tost_p95_ok),
            "p99_ok": bool(tost_p99_ok),
            "j_ok": bool(tost_j_ok),
        },
    }
    out["pass"] = bool(all(cv_ok.values()) and tost_p95_ok and tost_p99_ok and tost_j_ok)

    os.makedirs(os.path.dirname(args.out) or "out", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
