#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pathlib, re
from typing import List, Dict, Any, Tuple


def load_mem_claims(memdir: pathlib.Path) -> List[Dict[str, Any]]:
    path = memdir / 'claims.jsonl'
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def dedupe_latest(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    last: Dict[str, Dict[str, Any]] = {}
    for c in claims:
        cid = str(c.get('id') or '')
        if not cid:
            continue
        last[cid] = c
    return list(last.values())


def tokenize(s: str) -> List[str]:
    import re as _re
    return _re.findall(r'[a-z0-9]+', (s or '').lower())


def hash_embed(text: str, dim: int = 384):
    import numpy as _np, hashlib as _h
    D = int(dim)
    v = _np.zeros(D, dtype=_np.float32)
    toks = tokenize(text)
    last = None
    for t in toks:
        h = int(_h.sha256(t.encode()).hexdigest()[:8], 16)
        i = h % D
        s = 1.0 if (h >> 31) & 1 else -1.0
        v[i] += s
        if last is not None:
            b = last + '/' + t
            hb = int(_h.sha256(b.encode()).hexdigest()[:8], 16)
            ib = hb % D
            sb = 1.0 if (hb >> 31) & 1 else -1.0
            v[ib] += sb
        last = t
    n = float(_np.linalg.norm(v))
    if n > 0:
        v /= n
    return v


def guess_domain(c: Dict[str, Any]) -> str:
    # Prefer folder-based hints from sources
    srcs = c.get('sources') or []
    paths = ' '.join(str(s.get('url','')) for s in srcs).lower()
    txt = str(c.get('text','')).lower()
    if '/brain/' in paths or any(w in txt for w in ['neuron','cortex','synapse','hippocamp','dopamine','predictive coding','gw theory']):
        return 'BRAIN'
    if '/agi/' in paths or any(w in txt for w in ['agi','world model','planner','meta-learning','autoencoder','sparse autoencoder','alignment']):
        return 'AGI'
    return 'UNKNOWN'


def pairwise_topk(brain: List[Tuple[Dict[str,Any], Any]], agi: List[Tuple[Dict[str,Any], Any]], k: int, min_sim: float) -> List[Dict[str, Any]]:
    import numpy as _np
    if not brain or not agi:
        return []
    B = _np.stack([v for _, v in brain], axis=0)
    A = _np.stack([v for _, v in agi], axis=0)
    S = B @ A.T  # cosine since vectors are normalized
    out: List[Dict[str, Any]] = []
    for i, (bc, bv) in enumerate(brain):
        sims = S[i]
        idx = sims.argsort()[::-1]
        taken = 0
        for j in idx:
            s = float(sims[j])
            if s < min_sim:
                break
            ac, av = agi[int(j)]
            out.append({
                'brain': {'id': bc.get('id'), 'text': bc.get('text'), 'sources': bc.get('sources', [])},
                'agi': {'id': ac.get('id'), 'text': ac.get('text'), 'sources': ac.get('sources', [])},
                'sim': round(s, 4)
            })
            taken += 1
            if taken >= k:
                break
    # Sort all pairs by sim desc and cap global top 500 for readability
    out.sort(key=lambda r: r['sim'], reverse=True)
    return out[:500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memdir', default='.guardian_mem')
    ap.add_argument('--include-brain', default='', help='regex to force include for brain side (optional)')
    ap.add_argument('--include-agi', default='', help='regex to force include for agi side (optional)')
    ap.add_argument('--k', type=int, default=3, help='top-k AGI matches per brain claim')
    ap.add_argument('--min-sim', type=float, default=0.25)
    ap.add_argument('--out', default='logs/claims.similarities.json')
    args = ap.parse_args()

    memdir = pathlib.Path(args.memdir)
    claims = dedupe_latest(load_mem_claims(memdir))

    incb = re.compile(args.include_brain) if args.include_brain else None
    inca = re.compile(args.include_agi) if args.include_agi else None

    brain, agi = [], []
    for c in claims:
        dom = guess_domain(c)
        if dom == 'BRAIN':
            if incb and not incb.search(c.get('text','')):
                continue
            brain.append((c, hash_embed(c.get('text',''))))
        elif dom == 'AGI':
            if inca and not inca.search(c.get('text','')):
                continue
            agi.append((c, hash_embed(c.get('text',''))))

    pairs = pairwise_topk(brain, agi, k=max(1,int(args.k)), min_sim=float(args.min_sim))
    out_doc = {
        'brain_n': len(brain),
        'agi_n': len(agi),
        'pairs': pairs,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, 'w', encoding='utf-8').write(json.dumps(out_doc, ensure_ascii=False, indent=2))
    print(json.dumps({'out': args.out, 'pairs': len(pairs), 'brain': len(brain), 'agi': len(agi)}, indent=2))


if __name__ == '__main__':
    main()

