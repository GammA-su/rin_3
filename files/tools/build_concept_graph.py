#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pathlib, re, urllib.request
from typing import Any, Dict, List, Tuple


RELATIONS = [
    "part_of", "located_in", "projects_to", "modulates", "excites", "inhibits",
    "associated_with", "causes", "implements", "supports", "contradicts"
]


def ask_llamacpp(server: str, prompt: str, n_predict: int = 400, temperature: float = 0.2) -> str:
    payload = {
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": 0.8,
        "repeat_penalty": 1.12,
        "n_predict": int(n_predict),
        "stream": False,
        "cache_prompt": True,
        "stop": ["\n\n"]
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(server.rstrip('/') + '/completion', data=data, headers={"Content-Type":"application/json", "User-Agent":"Triforce-Prophet/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = r.read().decode('utf-8', errors='ignore')
    try:
        obj = json.loads(raw)
        return obj.get('content') or obj.get('response') or ''
    except Exception:
        return raw


def extract_json_object(s: str) -> dict:
    """Best-effort: extract the largest balanced {...} and parse it."""
    if not s:
        return {}
    # 1) Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2) Find longest balanced braces
    start_idx = None
    depth = 0
    best = None
    for i, ch in enumerate(s):
        if ch == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    cand = s[start_idx:i+1]
                    best = cand
    if best:
        try:
            return json.loads(best)
        except Exception:
            pass
    # 3) Last resort: replace common trailing commas and try
    t = s.replace(",]", "]").replace(",}", "}")
    try:
        return json.loads(t)
    except Exception:
        return {}


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    out = []
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


def build_graph_prompt(text: str) -> str:
    rels = ", ".join(RELATIONS)
    return (
        'Extract a small concept graph from the TEXT. Return STRICT JSON with keys '
        '{"nodes":["string"], "edges":[{"src":"string","rel":"string","dst":"string"}]}. '
        f'Use only these relations: [{rels}]. Keep 5–12 edges.\n\nTEXT:\n{text}'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cards', default='.guardian_mem_brain/knowledge.cards.jsonl')
    ap.add_argument('--notes', default='.guardian_mem_brain/knowledge.notes.jsonl')
    ap.add_argument('--llama', default='http://127.0.0.1:11435')
    ap.add_argument('--out', default='.guardian_mem_brain/concepts.graph.json')
    args = ap.parse_args()

    cards = load_jsonl(pathlib.Path(args.cards))
    notes = load_jsonl(pathlib.Path(args.notes))

    nodes_set = set()
    edges: List[Dict[str, Any]] = []

    # Build prompts from cards and note sections
    texts: List[Tuple[str, str]] = []  # (topic, text)
    for c in cards:
        t = c.get('topic') or ''
        txt = str(c.get('text') or '')
        if t and txt:
            texts.append((t, txt))
    for n in notes:
        t = n.get('topic') or ''
        sec = n.get('note', {}).get('sections', {})
        txt = "\n".join([str(sec.get('definition','')), str(sec.get('mechanism','')), str(sec.get('evidence',''))])
        if t and txt.strip():
            texts.append((t, txt))

    for topic, txt in texts:
        prompt = build_graph_prompt(txt)
        raw = ask_llamacpp(args.llama, prompt)
        g = extract_json_object(raw)
        if not isinstance(g, dict):
            continue
        for n in g.get('nodes', []) or []:
            if isinstance(n, str):
                nodes_set.add(n.strip())
        for e in g.get('edges', []) or []:
            src = e.get('src'); rel = e.get('rel'); dst = e.get('dst')
            if isinstance(src, str) and isinstance(rel, str) and isinstance(dst, str) and rel in RELATIONS:
                edges.append({"src": src.strip(), "rel": rel.strip(), "dst": dst.strip(), "topic": topic})

    nodes = sorted(n for n in nodes_set if n)
    out = {"nodes": nodes, "edges": edges, "relations": RELATIONS}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({"nodes": len(nodes), "edges": len(edges), "out": args.out}, indent=2))


if __name__ == '__main__':
    main()
