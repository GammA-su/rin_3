#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pathlib, re, time, urllib.request, urllib.error
from typing import List, Tuple, Dict, Any


def tokenize(s: str) -> List[str]:
    return re.findall(r'[a-z0-9]+', (s or '').lower())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return 0.0 if u == 0 else len(a & b) / u


def scan_docs_for_topic(docs_dir: str, topic: str, k: int = 8, max_each: int = 600) -> List[Tuple[str, str]]:
    root = pathlib.Path(docs_dir)
    cand = [p for p in root.rglob('*.txt') if p.is_file()]
    q = set(tokenize(topic))
    scored: List[Tuple[float, pathlib.Path]] = []
    for p in cand:
        stem = p.stem.replace('_', ' ')
        s = set(tokenize(stem))
        score = jaccard(q, s)
        if score <= 0:
            continue
        scored.append((score, p))
    scored.sort(key=lambda t: t[0], reverse=True)
    out: List[Tuple[str, str]] = []
    for _, p in scored[:max(1,int(k))]:
        try:
            t = p.read_bytes()[: max_each * 4].decode('utf-8', errors='ignore')
        except Exception:
            continue
        # drop leading URL line
        lines = [ln for ln in t.splitlines() if not (ln.startswith('http') and '://' in ln)]
        snip = '\n'.join(lines)
        if len(snip) > max_each:
            snip = snip[:max_each] + '…'
        out.append((str(p), snip))
    return out


def ask_llamacpp(server: str, prompt: dict, timeout: int = 180) -> dict:
    data = json.dumps(prompt).encode('utf-8')
    req = urllib.request.Request(server.rstrip('/') + '/completion', data=data, headers={"Content-Type":"application/json", "User-Agent":"Triforce-Prophet/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode('utf-8', errors='ignore')
    try:
        return json.loads(raw)
    except Exception:
        return {"response": raw}


def has_meta_instructions(s: str) -> bool:
    t = (s or '').lower()
    return any(x in t for x in [
        'return strict json', 'only json', 'we need json', 'provide only json',
        'strict json with keys', 'json with keys'
    ])


def extract_cite_numbers(s: str) -> List[str]:
    nums: List[str] = []
    for m in re.finditer(r"\[(\d+)\]", s or ''):
        n = m.group(1)
        if n not in nums:
            nums.append(n)
    return nums


def build_prompt(topic: str, snippets: List[Tuple[str, str]]) -> Tuple[str, str]:
    system = (
        'You are a precise tutor. Return STRICT JSON with keys {"text": "string", "cites": ["string"]}. '
        'Write 3–5 sentences (≤120 words total) that define and explain the concept clearly, citing sources with [1],[2],…. Only output JSON.'
    )
    parts = [f"Topic: {topic}"]
    for i, (path, snip) in enumerate(snippets, 1):
        parts.append(f"[{i}] {path}\n{snip}")
    user = "\n\n".join(parts)
    return system, user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topics', default='files/configs/topics.brain.txt')
    ap.add_argument('--docs', default='docs/brain')
    ap.add_argument('--memdir', default='.guardian_mem_brain')
    ap.add_argument('--llama', default='http://127.0.0.1:11435')
    ap.add_argument('--k', type=int, default=8, help='evidence items per topic')
    ap.add_argument('--max_each', type=int, default=600)
    ap.add_argument('--out', default='.guardian_mem_brain/knowledge.cards.jsonl')
    args = ap.parse_args()

    topics = [ln.strip() for ln in pathlib.Path(args.topics).read_text(encoding='utf-8').splitlines() if ln.strip() and not ln.strip().startswith('#')]
    memdir = pathlib.Path(args.memdir)
    memdir.mkdir(parents=True, exist_ok=True)
    claims_path = memdir / 'claims.jsonl'
    cards_path = pathlib.Path(args.out)
    cards_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for t in topics:
        snips = scan_docs_for_topic(args.docs, t, k=args.k, max_each=args.max_each)
        if not snips:
            continue
        sys_msg, user_msg = build_prompt(t, snips)
        # Build llama.cpp payload
        payload = {
            "prompt": f"{sys_msg}\n\nUser:\n{user_msg}\n\nAssistant:",
            "temperature": 0.2,
            "top_p": 0.8,
            "repeat_penalty": 1.12,
            "n_predict": 400,
            "stream": False,
            "cache_prompt": True,
            # Stop at blank line to reduce drift
            "stop": ["\n\n"]
        }
        try:
            # Retry loop to reduce meta-instruction bleed
            tries = 0
            while True:
                obj = ask_llamacpp(args.llama, payload)
                text = obj.get('content') or obj.get('response') or ''
                try:
                    j = json.loads(text)
                except Exception:
                    j = {"text": text, "cites": []}
                if has_meta_instructions(str(j.get('text',''))) or len(str(j.get('text',''))) < 60:
                    if tries >= 1:
                        break
                    rep = (
                        'Rewrite as STRICT JSON: {"text":"string","cites":["string"]}. '
                        'Do not mention JSON or instructions. 3–5 sentences (≤120 words) with [1],[2] citations from snippets.'
                    )
                    payload["prompt"] = f"{rep}\n\nUser:\n{user_msg}\n\nAssistant:"
                    tries += 1
                    continue
                break
            # Write knowledge card with repaired cites if possible
            text_out = j.get('text','')
            cites_out = j.get('cites',[])
            if not isinstance(cites_out, list):
                cites_out = []
            nums = extract_cite_numbers(text_out)
            if nums:
                cites_out = nums
            card = {"topic": t, "text": text_out, "cites": cites_out, "sources": [p for p,_ in snips]}
            cards_path.open('a', encoding='utf-8').write(json.dumps(card, ensure_ascii=False) + "\n")
            # Append a claim with sources (so the system can retrieve it later)
            claim = {
                "id": f"card-{int(time.time()*1000)}",
                "text": t,
                "q": 0.6,
                "stance": "neutral",
                "sources": [{"url": p, "domain_tier": 2 if '/peer/' in p else 3} for p,_ in snips]
            }
            claims_path.open('a', encoding='utf-8').write(json.dumps(claim, ensure_ascii=False) + "\n")
            n_ok += 1
        except Exception as e:
            err = {"topic": t, "error": str(e)}
            cards_path.open('a', encoding='utf-8').write(json.dumps(err) + "\n")
            continue

    print(json.dumps({"topics": len(topics), "cards": n_ok, "out": str(cards_path)}, indent=2))


if __name__ == '__main__':
    main()
