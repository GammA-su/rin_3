#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pathlib, re, sys
from typing import Any, Dict, List, Tuple

# Reuse the repo's client and JSON extractor
try:
    from guardian_agi_min import LLMClient, MockLLM, extract_json_object
except Exception:
    LLMClient = None  # type: ignore
    MockLLM = None    # type: ignore
    def extract_json_object(s: str) -> dict:
        try:
            return json.loads(s)
        except Exception:
            return {"text": s}


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


def select_claims(claims: List[Dict[str, Any]], include: str, exclude: str, limit: int) -> List[Dict[str, Any]]:
    inc_re = re.compile(include) if include else None
    exc_re = re.compile(exclude) if exclude else None
    sel: List[Dict[str, Any]] = []
    for c in claims:
        text = str(c.get('text') or '')
        if exc_re and exc_re.search(text):
            continue
        if inc_re and not inc_re.search(text):
            continue
        sel.append(c)
        if 0 < limit <= len(sel):
            break
    return sel


def read_snippets(sources: List[Dict[str, Any]], max_each: int = 360) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for s in sources or []:
        p = pathlib.Path(str(s.get('url', '')))
        if not p.exists():
            continue
        try:
            t = p.read_bytes()[: max_each * 4].decode('utf-8', errors='ignore')
            # Drop leading URL line if present
            t = '\n'.join([ln for ln in t.splitlines() if not (ln.startswith('http') and '://' in ln)])
        except Exception:
            continue
        snippet = (t[:max_each] + '…') if len(t) > max_each else t
        out.append((str(p), snippet))
        if len(out) >= 3:
            break
    return out


def build_prompt(claim_text: str, snippets: List[Tuple[str, str]]) -> Tuple[str, str]:
    system = (
        "You are a precise tutor. Return STRICT JSON with keys: "
        "{\"text\": string, \"cites\": [string]}. The 'text' must be ≤120 words and reference "
        "the snippets using bracketed ids like [1], [2].\n"
        "Only output JSON."
    )
    parts = [f"Topic: {claim_text}"]
    for i, (path, snip) in enumerate(snippets, 1):
        parts.append(f"[{i}] {path}\n{snip}")
    user = "\n\n".join(parts)
    return system, user


def evaluate_json(j: Dict[str, Any]) -> Dict[str, Any]:
    text = str(j.get('text') or '')
    cites = j.get('cites') or []
    if not isinstance(cites, list):
        cites = []
    words = len([w for w in text.strip().split() if w])
    # re.search returns a match or None; coerce to bool
    has_refs = bool(re.search(r'\[[0-9]+\]', text)) and len(cites) >= 1
    ok = (words > 0 and words <= 120 and has_refs)
    return {"words": words, "has_refs": has_refs, "ok": ok}


def get_llm(args) -> Any:
    if args.mock:
        return MockLLM(debug=args.debug) if MockLLM else None
    return LLMClient(args.model, host=args.host, port=args.port, debug=args.debug) if LLMClient else None


def ask_llamacpp(server: str, system_msg: str, user_msg: str, *, temperature: float, top_p: float, repeat_penalty: float, n_predict: int, grammar: str | None = None) -> str:
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
    payload = dict(base_payload)
    if grammar:
        payload["grammar"] = grammar
    try:
        data = _post(payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='ignore') if hasattr(e, 'read') else ''
        if e.code == 400 and grammar:
            payload = dict(base_payload)
            data = _post(payload)
        else:
            raise RuntimeError(f"llama.cpp HTTP {e.code}: {e.reason} body={body[:240]}")
    except Exception as e:
        raise RuntimeError(f"llama.cpp error: {e}")
    obj = _json.loads(data) if data else {}
    return obj.get('content') or obj.get('response') or ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memdir', default='.guardian_mem')
    ap.add_argument('--include', default='', help='regex to include claim texts')
    ap.add_argument('--exclude', default='', help='regex to exclude claim texts')
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--model', default='gpt-oss:20b')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=11434)
    ap.add_argument('--llama-server', default='', help='http://host:port of llama.cpp server (uses /completion)')
    ap.add_argument('--mock', action='store_true')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--out', default='logs/learned.tests.jsonl')
    ap.add_argument('--gbnf', default=os.getenv('GBNF', ''), help='optional GBNF grammar file for llama.cpp')
    args = ap.parse_args()

    memdir = pathlib.Path(args.memdir)
    claims = dedupe_latest(load_mem_claims(memdir))
    targets = select_claims(claims, args.include, args.exclude, args.limit)

    llm = None if args.llama_server else get_llm(args)
    if args.llama_server:
        # We'll call llama.cpp directly later
        pass
    elif llm is None:
        print(json.dumps({"error": "LLM client not available; use --mock or --llama-server"}))
        sys.exit(2)

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    # Load grammar if requested
    grammar = None
    if args.gbnf:
        try:
            grammar = pathlib.Path(args.gbnf).read_text(encoding='utf-8')
        except Exception:
            grammar = None

    for c in targets:
        text = str(c.get('text') or '')
        snippets = read_snippets(c.get('sources') or [], max_each=400)
        sys_msg, user_msg = build_prompt(text, snippets)
        try:
            if args.llama_server:
                raw = ask_llamacpp(args.llama_server, sys_msg, user_msg, temperature=0.3, top_p=0.8, repeat_penalty=1.12, n_predict=300, grammar=grammar)
            else:
                raw = llm.ask(sys_msg, user_msg, temperature=0.3, top_p=0.8, repeat_penalty=1.12, num_predict=300, force_json=True, attempts_log=[], phase_label='learned')
        except Exception as e:
            rec = {"id": c.get('id'), "text": text, "error": str(e), "ok": False}
            outp.open('a', encoding='utf-8').write(json.dumps(rec) + "\n")
            continue
        try:
            j = extract_json_object(raw)
        except Exception:
            j = {"text": raw, "cites": []}
        evalr = evaluate_json(j)
        rec = {
            "id": c.get('id'),
            "text": text,
            "result": j,
            "eval": evalr,
        }
        n_ok += 1 if evalr.get('ok') else 0
        outp.open('a', encoding='utf-8').write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {"tested": len(targets), "pass": n_ok, "out": str(outp)}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
