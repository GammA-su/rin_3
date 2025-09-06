#!/usr/bin/env python3
"""
Gate v2 orchestrator (caps + tails + Holm-Bonferroni + tie-break).

Usage:
  python files/tools/gate_v2.py \
    --spec UCBxTOT-gold.json \
    --candidates files/configs/gatev2_candidates.json \
    --pvals-out out/pvals.json \
    --out out/gate_v2.json

Notes:
  - This is a lightweight, data-driven orchestrator. It enforces caps/tails
    and reduces multiple comparisons using Holm-Bonferroni over per-seed p-values
    supplied in the candidates manifest.
  - It does not compute p-values itself; supply them via the manifest so the
    existing tools/mcomp_holm.py and schema checks remain consistent.
"""

from __future__ import annotations
import argparse, json, sys, pathlib


def _load_json(path: str):
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"failed to load JSON: {path}: {e}")


def _holm(pvals: list[float], alpha: float) -> list[bool]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    rejects = [False] * m
    for k, i in enumerate(order):
        thresh = alpha / (m - k)
        if pvals[i] <= thresh:
            rejects[i] = True
        else:
            break
    return rejects


def _as_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def enforce_caps_and_tails(spec: dict, stage: str, cand: dict) -> tuple[bool, dict]:
    g = spec.get("gate_v2", {})
    slo = spec.get("budgets", {}).get("slo", {})
    out = {"caps_ok": True, "tails_ok": True}
    ok = True
    if stage == "micro":
        caps = g.get("micro", {}).get("caps", {})
        tails = g.get("micro", {}).get("tails", {})
        c = cand.get("caps", {})
        # caps
        tests = {
            "flops_pct": (_as_float(c.get("flops_pct")), _as_float(caps.get("flops_pct"))),
            "vram_gb_delta": (_as_float(c.get("vram_gb_delta")), _as_float(caps.get("vram_gb_delta"))),
            "params_active_m": (_as_float(c.get("params_active_m")), _as_float(caps.get("params_active_m"))),
            "tps_delta_pct": (_as_float(c.get("tps_delta_pct")), _as_float(caps.get("tps_delta_pct"))),
            "j_per_inf_delta_pct": (_as_float(c.get("j_per_inf_delta_pct")), _as_float(caps.get("j_per_inf_delta_pct"))),
        }
        # <= for all caps
        for key, (val, lim) in tests.items():
            if key == "flops_pct" and val > lim:
                out["caps_ok"] = False
            if key == "vram_gb_delta" and val > lim:
                out["caps_ok"] = False
            if key == "params_active_m" and val > lim:
                out["caps_ok"] = False
            if key == "tps_delta_pct" and abs(val) > lim:
                out["caps_ok"] = False
            if key == "j_per_inf_delta_pct" and val > lim:
                out["caps_ok"] = False
        # tails
        t = cand.get("tails", {})
        p95_ok = _as_float(t.get("p95_s")) <= _as_float(tails.get("p95_s_max", slo.get("p95_s", 3.5)))
        p99_ok = _as_float(t.get("p99_s")) <= _as_float(tails.get("p99_s_max", slo.get("p99_s", 4.5)))
        out["tails_ok"] = bool(p95_ok and p99_ok)
        ok = bool(out["caps_ok"] and out["tails_ok"])
    elif stage == "mael":
        caps = g.get("mael", {}).get("caps", {})
        tails = g.get("mael", {}).get("tails", {})
        c = cand.get("caps", {})
        tests = {
            "params_delta_m": (_as_float(c.get("params_delta_m")), _as_float(caps.get("params_delta_m"))),
            "flops_pct": (_as_float(c.get("flops_pct")), _as_float(caps.get("flops_pct"))),
            "vram_gb_delta": (_as_float(c.get("vram_gb_delta")), _as_float(caps.get("vram_gb_delta"))),
        }
        for key, (val, lim) in tests.items():
            if val > lim:
                out["caps_ok"] = False
        t = cand.get("tails", {})
        p95_ok = _as_float(t.get("p95_s")) <= _as_float(tails.get("p95_s_max", slo.get("p95_s", 3.5)))
        p99_ok = _as_float(t.get("p99_s")) <= _as_float(tails.get("p99_s_max", slo.get("p99_s", 4.5)))
        out["tails_ok"] = bool(p95_ok and p99_ok)
        ok = bool(out["caps_ok"] and out["tails_ok"])
    else:
        ok = False
        out["error"] = f"unknown stage: {stage}"
    return ok, out


def seeds_criteria_ok(spec: dict, stage: str, cand: dict, seeds: list[int]) -> tuple[bool, dict]:
    g = spec.get("gate_v2", {})
    crit = g.get(stage, {})
    acc_min = _as_float(crit.get("acc_delta_abs_min_pct", 0.0))
    ece_max = _as_float(crit.get("ece_delta_abs_max", 0.0))
    tails_ok = True
    acc_ok = True
    ece_ok = True
    veto = False
    per_seed = cand.get("per_seed", {})
    for s in seeds:
        row = per_seed.get(str(s)) or per_seed.get(int(s)) or {}
        acc_ok = acc_ok and (_as_float(row.get("acc_delta_abs_pct", acc_min)) >= acc_min)
        ece_ok = ece_ok and (_as_float(row.get("ece_delta_abs", ece_max)) <= ece_max)
        if bool(row.get("catastrophic_veto", False)):
            veto = True
        # tails within SLO
        p95 = _as_float(row.get("p95_s", cand.get("tails", {}).get("p95_s", 0.0)))
        p99 = _as_float(row.get("p99_s", cand.get("tails", {}).get("p99_s", 0.0)))
        tails_ok = tails_ok and (p95 <= _as_float(crit.get("tails", {}).get("p95_s_max", 3.5))) and (p99 <= _as_float(crit.get("tails", {}).get("p99_s_max", 4.5)))
    ok = bool(acc_ok and ece_ok and tails_ok and not veto)
    return ok, {"acc_ok": acc_ok, "ece_ok": ece_ok, "tails_ok": tails_ok, "veto": veto}


def write_pvals(pvals_out: str, cand: dict, alpha: float = 0.05) -> dict:
    # Manifest may contain pvals: { suite: { seed: p } }
    pman = cand.get("pvals", {})
    suites = []
    for suite_name, per_seed in pman.items():
        items = []
        for seed_str, p in per_seed.items():
            try:
                seed = int(seed_str)
            except Exception:
                continue
            items.append({"seed": seed, "p": float(p)})
        if items:
            suites.append({"name": suite_name, "per_seed": items})
    if not suites:
        # still write a valid doc with no pvals to satisfy schema consumer
        doc = {"method": "paired_t", "alpha": alpha, "suites": []}
    else:
        doc = {"method": "paired_t", "alpha": alpha, "suites": suites}
    pathlib.Path(pvals_out).parent.mkdir(parents=True, exist_ok=True)
    open(pvals_out, "w", encoding="utf-8").write(json.dumps(doc, indent=2))
    return doc


def tie_break_key(spec: dict, cand: dict) -> tuple:
    tb = spec.get("gate_v2", {}).get("tie_break", ["acc_mean_desc","jinf_asc","p99_asc"])  # default
    # compute aggregates used in tie-breaks
    acc_mean = _as_float(cand.get("acc_mean", 0.0))
    # jinf: prefer lower energy delta pct if provided
    jinf = _as_float(cand.get("caps", {}).get("j_per_inf_delta_pct", 0.0))
    p99 = _as_float(cand.get("tails", {}).get("p99_s", 1e9))
    key = []
    for k in tb:
        if k == "acc_mean_desc":
            key.append(-acc_mean)
        elif k == "jinf_asc":
            key.append(jinf)
        elif k == "p99_asc":
            key.append(p99)
        else:
            key.append(0)
    return tuple(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="Path to UCBxTOT-gold.json")
    ap.add_argument("--candidates", required=True, help="Path to candidates manifest JSON")
    ap.add_argument("--out", required=True, help="Path to write gate result JSON")
    ap.add_argument("--pvals-out", default="out/pvals.json", help="Where to write pvals.json for Holm step")
    args = ap.parse_args()

    spec = _load_json(args.spec)
    manifest = _load_json(args.candidates)
    stage = str(manifest.get("stage", "micro")).lower()
    seeds = spec.get("gate_v2", {}).get("seeds", [5, 7, 11])
    alpha = 0.05

    passed = []
    rejected = []
    reasons = {}
    for cand in manifest.get("candidates", []):
        name = cand.get("name", "unnamed")
        caps_ok, caps_info = enforce_caps_and_tails(spec, stage, cand)
        seeds_ok, seeds_info = seeds_criteria_ok(spec, stage, cand, seeds)
        if not caps_ok or not seeds_ok:
            rejected.append(name)
            reasons[name] = {"caps": caps_info, "seeds": seeds_info}
            continue

        # Write/update pvals for this candidate so stats_checks can consume
        pdoc = write_pvals(args.pvals_out, cand, alpha=alpha)
        # Evaluate Holm survivorship locally for summary
        flat_ps = []
        for suite in pdoc.get("suites", []):
            for row in suite.get("per_seed", []):
                flat_ps.append(float(row.get("p", 1.0)))
        if flat_ps:
            rejects = _holm(flat_ps, alpha)
            survivors = sum(1 for r in rejects if not r)
        else:
            survivors = 0
        cand_summary = {
            "name": name,
            "caps": caps_info,
            "seeds": seeds_info,
            "holm_survivors": survivors,
            "acc_mean": _as_float(cand.get("acc_mean", 0.0)),
            "tails": cand.get("tails", {}),
        }
        passed.append((cand_summary, cand))

    promote_exactly = int(spec.get("gate_v2", {}).get("promote_exactly", 1))
    # Rank passed by survivors then tie-break key
    passed.sort(key=lambda t: (-(t[0]["holm_survivors"]), tie_break_key(spec, t[1])))
    selected = passed[:promote_exactly]

    out = {
        "stage": stage,
        "seeds": seeds,
        "candidates_total": len(manifest.get("candidates", [])),
        "rejected": rejected,
        "selected": [s[0] for s in selected],
        "ok": bool(len(selected) == promote_exactly and promote_exactly > 0)
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

