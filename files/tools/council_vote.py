#!/usr/bin/env python3
"""
Council aggregator for promotions.

Decision rule (from spec):
  Approve if all 5 roles pass OR
  (>=4/5 pass with Safety and Adversarial mandatory AND p99_margin_s <= -0.05).

Inputs:
  --spec UCBxTOT-gold.json
  --rubric files/configs/council_rubric.json (optional; defaults from spec)
  --input files/configs/council_input.json (role booleans and metrics)
  --out out/council.json

Example council_input.json:
{
  "roles": {"Unit": true, "Property": true, "Adversarial": true, "Runtime-SLO": true, "Safety": true},
  "p99_margin_s": -0.08,
  "ece_delta": -0.012,
  "policy_violations": 0,
  "tails_ok": true,
  "energy_ok": true,
  "shadow_deploy_pass": true
}
"""

from __future__ import annotations
import argparse, json, pathlib


def _load(p: str) -> dict:
    return json.load(open(p, "r", encoding="utf-8"))


def _norm_role(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    return name.replace(" ", "").replace("-", "").strip().lower()


def decide(spec: dict, rubric: dict, inp: dict) -> dict:
    roles_spec = spec.get("council", {}).get("roles", ["Unit", "Property", "Adversarial", "Runtime-SLO", "Safety"])
    # Normalize role names
    roles_norm = [_norm_role(r) for r in roles_spec]
    role_pass_map = {}
    ri = inp.get("roles", {})
    for k, v in ri.items():
        role_pass_map[_norm_role(k)] = bool(v)

    n_pass = sum(1 for r in roles_norm if role_pass_map.get(r, False))
    all_pass = (n_pass == len(roles_norm))
    safety_ok = role_pass_map.get(_norm_role("Safety"), False)
    adv_ok = role_pass_map.get(_norm_role("Adversarial"), False)
    p99_margin_ok = float(inp.get("p99_margin_s", 0.0)) <= -0.05

    approve = bool(all_pass or (n_pass >= 4 and safety_ok and adv_ok and p99_margin_ok))
    shadow_hours = int(spec.get("council", {}).get("shadow_deploy_hours", 24))
    shadow_pass = bool(inp.get("shadow_deploy_pass", False))

    decision = "rejected"
    if approve and shadow_pass:
        decision = "approved"
    elif approve and not shadow_pass:
        decision = "approved_pending_shadow"

    return {
        "roles": ri,
        "n_pass": n_pass,
        "required": len(roles_norm),
        "p99_margin_s": float(inp.get("p99_margin_s", 0.0)),
        "safety_ok": safety_ok,
        "adversarial_ok": adv_ok,
        "tails_ok": bool(inp.get("tails_ok", True)),
        "energy_ok": bool(inp.get("energy_ok", True)),
        "policy_violations": int(inp.get("policy_violations", 0)),
        "shadow_hours": shadow_hours,
        "shadow_deploy_pass": shadow_pass,
        "approve_rule": "all5 OR (>=4/5 & Safety+Adversarial & p99_margin<=-0.05)",
        "decision": decision,
        "ok": bool(decision in ("approved", "approved_pending_shadow")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--rubric", default="")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = _load(args.spec)
    rubric = _load(args.rubric) if args.rubric else {}
    inp = _load(args.input)
    out = decide(spec, rubric, inp)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

