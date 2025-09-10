#!/usr/bin/env python3
"""
Ground existing ACL claims with sources and stance from local docs.

Heuristics:
  - Match *.txt under DOCS by token overlap with the claim text.
  - Infer domain tier from folder name (primary=1, official/peer=2, media=3, blog/community=4, dissent=3).
  - Infer stance per doc: 'con' if path or text contains dissent markers; 'neutral' for media/blog; else 'pro'.
  - Attach top-K sources to each claim and set stance by majority vote over sources (fallback neutral).

Appends updated claim lines into mem JSONL so the latest version wins when loaded.
"""
from __future__ import annotations
import argparse, pathlib, json, re
from typing import Dict, Any, List


AUTH_TIER_BY_FOLDER = {
    'primary': 1,
    'official': 2,
    'peer': 2,
    'peerreview': 2,
    'media': 3,
    'reputable': 3,
    'blog': 4,
    'community': 4,
    'forum': 5,
    'dissent': 3,
}


def read_text(path: pathlib.Path, limit: int = 20000) -> str:
    try:
        return path.read_bytes()[:limit].decode('utf-8', errors='ignore')
    except Exception:
        return ''


def tier_from_path(path: pathlib.Path) -> int:
    s = str(path).replace('\\', '/').lower()
    for key, tier in AUTH_TIER_BY_FOLDER.items():
        if f'/{key}/' in s or s.endswith(f'/{key}'):
            return tier
    return 4


def stance_from_text_or_path(text: str, path: pathlib.Path) -> str:
    p = str(path).lower()
    if 'dissent' in p or re.search(r'\b(however|contradict|misleading|not simply)\b', text.lower()):
        return 'con'
    if any(k in p for k in ('media', 'blog', 'forum')):
        return 'neutral'
    return 'pro'


def tokenize(s: str) -> List[str]:
    return re.findall(r'[a-z0-9]+', (s or '').lower())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return 0.0 if u == 0 else len(a & b) / u


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


def append_mem_claim(memdir: pathlib.Path, claim: Dict[str, Any]) -> None:
    path = memdir / 'claims.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(claim, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memdir', default='.guardian_mem')
    ap.add_argument('--docs', default='docs')
    ap.add_argument('-k', '--k', type=int, default=3)
    args = ap.parse_args()

    memdir = pathlib.Path(args.memdir)
    docsdir = pathlib.Path(args.docs)
    claims = load_mem_claims(memdir)

    # Build doc index
    docs: List[pathlib.Path] = [p for p in docsdir.rglob('*.txt') if p.is_file()]
    docs_tokens: Dict[pathlib.Path, set[str]] = {p: set(tokenize(p.stem.replace('_', ' '))) for p in docs}

    updated = 0
    for base in claims:
        cid = base.get('id')
        text = base.get('text', '')
        if not cid or not text:
            continue
        cq = set(tokenize(text))
        scored: List[tuple[float, pathlib.Path]] = []
        for p in docs:
            score = jaccard(cq, docs_tokens.get(p, set()))
            if score > 0:
                scored.append((score, p))
        if not scored:
            continue
        scored.sort(key=lambda t: t[0], reverse=True)

        # Select top-K and build sources
        sel = [p for _, p in scored[: max(1, int(args.k))]]
        sources = []
        st_votes = {'pro': 0, 'neutral': 0, 'con': 0}
        for p in sel:
            txt = read_text(p)
            st = stance_from_text_or_path(txt, p)
            st_votes[st] = st_votes.get(st, 0) + 1
            sources.append({'url': str(p), 'domain_tier': tier_from_path(p)})

        # Majority stance; fallback neutral
        stance = max(st_votes.items(), key=lambda kv: kv[1])[0] if any(st_votes.values()) else 'neutral'

        # Append updated claim
        new_claim = dict(base)
        new_claim['sources'] = sources
        new_claim['stance'] = stance
        append_mem_claim(memdir, new_claim)
        updated += 1

    print(json.dumps({'updated': updated, 'memdir': str(memdir), 'docs': str(docsdir)}))


if __name__ == '__main__':
    main()

