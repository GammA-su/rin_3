#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pathlib, re, sys
from typing import Any, Dict, Iterable


def norm_text(s: str) -> str:
    return " ".join((s or "").lower().split())


def guess_domain(rec: Dict[str, Any]) -> str:
    txt = str(rec.get('text','')).lower()
    srcs = rec.get('sources') or []
    paths = ' '.join(str(s.get('url','')).lower() for s in srcs)
    if '/brain/' in paths or any(w in txt for w in ('neuron','cortex','synapse','hippocamp','dopamine','predictive coding','gw theory','global workspace')):
        return 'BRAIN'
    if '/agi/' in paths or any(w in txt for w in ('agi','world model','planner','meta-learning','autoencoder','sparse autoencoder','alignment','page rank','pagerank')):
        return 'AGI'
    return 'UNKNOWN'


def has_strong_source(rec: Dict[str, Any], tier_max: int) -> bool:
    srcs = rec.get('sources') or []
    for s in srcs:
        try:
            if int(s.get('domain_tier', 99)) <= tier_max:
                return True
        except Exception:
            continue
    return False


def iter_lines(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default='.guardian_mem/claims.jsonl')
    ap.add_argument('--out', dest='out', default='.guardian_mem/claims.pruned.jsonl')
    ap.add_argument('--max', dest='max_keep', type=int, default=100000, help='max records to keep')
    ap.add_argument('--min-sources', type=int, default=1, help='min number of sources required')
    ap.add_argument('--tier-max', type=int, default=3, help='require at least one source with domain_tier <= tier-max')
    ap.add_argument('--drop-slugs', default='^(en-wikipedia-org|arxiv-org-abs|en-wikipedia)', help='regex to drop sluggy topics')
    ap.add_argument('--domains', default='', help='comma list of domains to keep (BRAIN,AGI). empty=all')
    ap.add_argument('--include', default='', help='regex; keep only claims whose text matches (optional)')
    args = ap.parse_args()

    inp = pathlib.Path(args.inp)
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    drop_re = re.compile(args.drop_slugs) if args.drop_slugs else None
    include_re = re.compile(args.include) if args.include else None
    keep_domains = {d.strip().upper() for d in args.domains.split(',') if d.strip()} if args.domains else set()

    seen_text: set[str] = set()
    kept = 0
    with outp.open('w', encoding='utf-8') as w:
        for rec in iter_lines(inp):
            if kept >= args.max_keep:
                break
            text = rec.get('text') or ''
            if drop_re and drop_re.search(text):
                continue
            if include_re and not include_re.search(text):
                continue
            if len(rec.get('sources') or []) < args.min_sources:
                continue
            if args.tier_max is not None and args.tier_max >= 0 and not has_strong_source(rec, args.tier_max):
                continue
            dom = guess_domain(rec)
            if keep_domains and dom not in keep_domains:
                continue
            key = norm_text(text)
            if key in seen_text:
                continue
            seen_text.add(key)
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(json.dumps({
        'in': str(inp), 'out': str(outp), 'kept': kept,
        'max': args.max_keep, 'domains': sorted(list(keep_domains))
    }, indent=2))


if __name__ == '__main__':
    main()
