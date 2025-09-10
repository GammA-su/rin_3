#!/usr/bin/env python3
"""
Convert an eval artifact (CSV/JSON/JSONL/Parquet) into a tidy CSV with
columns: suite, seed, base, cand.

Usage examples:
  # Auto-detect columns, set suite and seed explicitly
  python files/tools/evals_to_csv.py --in artifacts/suite_full.json \
    --out evals.csv --suite my-suite --seed 7

  # Explicit column mapping (CSV)
  python files/tools/evals_to_csv.py --in results.csv --out evals.csv \
    --suite-col bench --seed-col run_seed --base-col base_acc --cand-col cand_acc

  # JSON lines with nested fields (dot paths)
  python files/tools/evals_to_csv.py --in results.jsonl --out evals.csv \
    --suite my-suite --seed 11 --base-col metrics.base.acc --cand-col metrics.cand.acc

Notes:
  - Parquet requires either pyarrow or pandas; if unavailable, the script will
    suggest installing pyarrow.
  - Heuristics try common column names when mappings are not provided: base|cand,
    base_acc|cand_acc, base_score|cand_score, y_base|y_cand, ref|hyp_acc.
"""

from __future__ import annotations
import argparse, csv, json, sys, pathlib
from typing import Any, Dict, Iterable, List, Tuple


def flatten(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key, sep))
        else:
            out[key] = v
    return out


HEURISTICS: List[Tuple[str, str]] = [
    ("base", "cand"),
    ("base_acc", "cand_acc"),
    ("base_score", "cand_score"),
    ("y_base", "y_cand"),
    ("acc_base", "acc_cand"),
    ("ref", "hyp_acc"),
]


def choose_cols(cols: Iterable[str]) -> Tuple[str, str]:
    s = {c.lower() for c in cols}
    for a, b in HEURISTICS:
        if a in s and b in s:
            return a, b
    # if not found, try contains
    for a, b in HEURISTICS:
        cand_a = next((c for c in s if a in c), None)
        cand_b = next((c for c in s if b in c), None)
        if cand_a and cand_b:
            return cand_a, cand_b
    raise SystemExit("Could not infer base/cand columns; please pass --base-col and --cand-col")


def load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return [dict(row) for row in rdr]


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # First, try full JSON (object/array)
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        # try common containers: {"results": [...]}, {"samples": [...]}
        for k in ("results", "samples", "data"):
            if isinstance(obj, dict) and isinstance(obj.get(k), list):
                return obj[k]
        return [obj] if isinstance(obj, dict) else []
    except Exception:
        # Fallback: parse as JSONL (each non-empty line is a JSON object)
        out: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                # ignore malformed lines
                continue
        return out


def load_parquet(path: str) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
        table = pq.read_table(path)
        return [dict(zip(table.column_names, row)) for row in zip(*[table[c].to_pylist() for c in table.column_names])]
    except Exception:
        try:
            import pandas as pd  # type: ignore
        except Exception:
            raise SystemExit("Parquet support requires pyarrow or pandas; install pyarrow and retry")
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")


def coerce_float(v: Any) -> float:
    if v is True:
        return 1.0
    if v is False:
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to eval artifact (csv/json/jsonl/parquet)")
    ap.add_argument("--out", default="evals.csv")
    ap.add_argument("--suite", default="", help="Suite name to stamp (or use --suite-col)")
    ap.add_argument("--suite-col", default="", help="Column to use for suite")
    ap.add_argument("--seed", type=int, default=None, help="Seed value to stamp (or use --seed-col)")
    ap.add_argument("--seed-col", default="", help="Column to use for seed")
    ap.add_argument("--base-col", default="", help="Column (or dot path) for base")
    ap.add_argument("--cand-col", default="", help="Column (or dot path) for cand")
    args = ap.parse_args()

    p = pathlib.Path(args.inp)
    if not p.exists():
        raise SystemExit(f"input not found: {p}")
    ext = p.suffix.lower()

    if ext == ".csv":
        rows = load_csv(str(p))
    elif ext in (".json", ".jsonl"):
        rows = load_json_or_jsonl(str(p))
    elif ext in (".parquet", ".pq"):
        rows = load_parquet(str(p))
    else:
        # try json as fallback
        rows = load_json_or_jsonl(str(p))

    # Normalize to flattened dicts for column access
    flat_rows = [flatten(r) if isinstance(r, dict) else {} for r in rows]
    if not flat_rows:
        raise SystemExit("no records found in artifact")

    # Determine suite and seed per row
    suite_vals: List[str] = []
    seed_vals: List[int] = []

    # Choose base/cand columns
    base_col = args.base_col
    cand_col = args.cand_col
    if not base_col or not cand_col:
        base_col, cand_col = choose_cols(flat_rows[0].keys())

    for r in flat_rows:
        # suite
        if args.suite_col:
            suite_vals.append(str(r.get(args.suite_col, args.suite or "suite")))
        else:
            suite_vals.append(args.suite or "suite")
        # seed
        if args.seed_col:
            try:
                seed_vals.append(int(r.get(args.seed_col)))
            except Exception:
                seed_vals.append(int(args.seed or 0))
        else:
            seed_vals.append(int(args.seed or 0))

    # Write tidy CSV
    out_p = pathlib.Path(args.out)
    with out_p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "seed", "base", "cand"])
        for r, suite, seed in zip(flat_rows, suite_vals, seed_vals):
            base_v = coerce_float(r.get(base_col))
            cand_v = coerce_float(r.get(cand_col))
            w.writerow([suite, seed, base_v, cand_v])

    print(json.dumps({"out": str(out_p), "rows": len(flat_rows), "base_col": base_col, "cand_col": cand_col}))


if __name__ == "__main__":
    main()
