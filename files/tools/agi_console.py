#!/usr/bin/env python3
from __future__ import annotations
import argparse, pathlib, re, json, sys, os
from typing import List, Dict, Any, Tuple


def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return 0.0 if u == 0 else len(a & b) / u


def scan_docs(docs: str, query: str, k: int = 5, max_each: int = 500) -> List[Tuple[str, str]]:
    root = pathlib.Path(docs)
    cand: List[pathlib.Path] = [p for p in root.rglob('*.txt') if p.is_file()]
    q = set(tokenize(query))
    scored: List[Tuple[float, pathlib.Path]] = []
    for p in cand:
        stem = p.stem.replace('_', ' ')
        s = set(tokenize(stem))
        score = jaccard(q, s)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda t: t[0], reverse=True)
    out: List[Tuple[str, str]] = []
    for _, p in scored[: max(1, int(k))]:
        try:
            t = p.read_bytes()[: max_each * 4].decode('utf-8', errors='ignore')
        except Exception:
            continue
        # drop leading URL
        lines = [ln for ln in t.splitlines() if not (ln.startswith('http') and '://' in ln)]
        snip = '\n'.join(lines)[: max_each]
        if len('\n'.join(lines)) > max_each:
            snip += '…'
        out.append((str(p), snip))
    return out


def build_prompt(goal: str, snippets: List[Tuple[str, str]]) -> Tuple[str, str]:
    system = (
        "You are an autonomous, precise assistant. Return STRICT JSON with keys: "
        "{\"text\": string, \"cites\": [string]}. The 'text' must be ≤150 words and reference "
        "the snippets using bracketed ids like [1], [2]. Only output JSON."
    )
    parts = [f"Goal: {goal}"]
    for i, (path, snip) in enumerate(snippets, 1):
        parts.append(f"[{i}] {path}\n{snip}")
    user = "\n\n".join(parts)
    return system, user


def ask_llamacpp(server: str, system_msg: str, user_msg: str, *, temperature: float = 0.3, top_p: float = 0.8, repeat_penalty: float = 1.12, n_predict: int = 400, grammar: str | None = None) -> str:
    import urllib.request, urllib.error
    import json as _json
    prompt = f"{system_msg}\n\nUser:\n{user_msg}\n\nAssistant:"
    base_payload = {
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "repeat_penalty": float(repeat_penalty),
        "n_predict": int(n_predict),
        "stream": False,
        "cache_prompt": True,
        "stop": ["\n\n"],
    }
    url = server.rstrip('/') + '/completion'
    def _post(pay):
        req = urllib.request.Request(url, data=_json.dumps(pay).encode('utf-8'), headers={"Content-Type":"application/json", "User-Agent":"Triforce-Prophet/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.read().decode('utf-8', errors='ignore')
    # Try with grammar first (if provided), then fallback without grammar on 400
    payload = dict(base_payload)
    if grammar:
        payload["grammar"] = grammar
    try:
        data = _post(payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='ignore') if hasattr(e, 'read') else ''
        # Fallback if grammar not supported by this llama.cpp build
        if e.code == 400 and grammar:
            payload = dict(base_payload)  # drop grammar
            data = _post(payload)
        else:
            raise RuntimeError(f"llama.cpp HTTP {e.code}: {e.reason} body={body[:240]}")
    obj = json.loads(data) if data else {}
    return obj.get('content') or obj.get('response') or ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--llama-server', default='http://127.0.0.1:11435')
    ap.add_argument('--docs', default='docs')
    ap.add_argument('--memdir', default='.guardian_mem')
    ap.add_argument('--gbnf', default='files/grammars/json_text_cites.gbnf')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--max_each', type=int, default=500)
    ap.add_argument('--out', default='logs/console.sessions.jsonl')
    ap.add_argument('--goal', default='', help='optional one-shot goal; if empty, runs interactive console')
    args = ap.parse_args()

    grammar = None
    if args.gbnf and pathlib.Path(args.gbnf).exists():
        try:
            grammar = pathlib.Path(args.gbnf).read_text(encoding='utf-8')
        except Exception:
            grammar = None

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    def run_goal(goal: str):
        snips = scan_docs(args.docs, goal, k=args.k, max_each=args.max_each)
        sys_msg, user_msg = build_prompt(goal, snips)
        raw = ask_llamacpp(args.llama_server, sys_msg, user_msg, grammar=grammar)
        # Attempt to parse; on fail, store raw
        try:
            result = json.loads(raw)
        except Exception:
            result = {"text": raw, "cites": []}
        rec = {"t": int(pathlib.time.time()) if hasattr(pathlib, 'time') else None, "goal": goal, "snippets": [p for p,_ in snips], "result": result}
        outp.open('a', encoding='utf-8').write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps({"goal": goal, "text": result.get('text',''), "cites": result.get('cites',[])}, ensure_ascii=False, indent=2))

    if args.goal:
        run_goal(args.goal)
        return

    print("AGI Console — type a goal and press Enter (Ctrl+C to exit)\n")
    try:
        while True:
            goal = input("goal> ").strip()
            if not goal:
                continue
            run_goal(goal)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == '__main__':
    main()
