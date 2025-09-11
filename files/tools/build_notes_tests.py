#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pathlib, re, urllib.request, urllib.error
from typing import List, Tuple, Dict


def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return 0.0 if u == 0 else len(a & b) / u


def scan_docs_for_topic(docs_dir: str, topic: str, k: int = 12, max_each: int = 1200) -> List[Tuple[str, str]]:
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
    for _, p in scored[:max(1, int(k))]:
        try:
            t = p.read_bytes()[: max_each * 4].decode('utf-8', errors='ignore')
        except Exception:
            continue
        lines = [ln for ln in t.splitlines() if not (ln.startswith('http') and '://' in ln)]
        snip = '\n'.join(lines)
        if len(snip) > max_each:
            snip = snip[:max_each] + '…'
        out.append((str(p), snip))
    return out


def ask_llamacpp(server: str, payload: dict, timeout: int = 240) -> dict:
    data = json.dumps(payload).encode('utf-8')
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
        'keys {"sections"', 'cites array'
    ])


def extract_cite_numbers(s: str) -> List[str]:
    # Find [1], [2], … and return unique string numbers in order
    nums: List[str] = []
    for m in re.finditer(r"\[(\d+)\]", s or ''):
        n = m.group(1)
        if n not in nums:
            nums.append(n)
    return nums


def build_note_prompt(topic: str, snippets: List[Tuple[str, str]]) -> Tuple[str, str]:
    system = (
        'You are a precise scientific tutor. Return STRICT JSON with keys '
        '{"sections": {"definition": "string", "mechanism": "string", "evidence": "string", "open_questions": ["string"]}, "cites": ["string"]}. '
        'Write a compact note of 400–800 words (3–5 sentences per section). Use bracketed citations [1],[2],[3] that refer to the provided snippets. Only output JSON.'
    )
    parts = [f"Topic: {topic}"]
    for i, (path, snip) in enumerate(snippets, 1):
        parts.append(f"[{i}] {path}\n{snip}")
    user = "\n\n".join(parts)
    return system, user


def build_qa_prompt(topic: str, note_text: str) -> str:
    return (
        'You are a tutor. Based only on the NOTE below, make 10 Q/A pairs for spaced repetition. '
        'Return STRICT JSON with key {"qa":[{"q":"string","a":"string"}, {"q":"string","a":"string"} ... ]}. '
        'Keep answers short (≤2 sentences).\n\n'
        f'NOTE:\n{note_text}'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topics', default='files/configs/topics.brain.txt')
    ap.add_argument('--docs', default='docs/brain')
    ap.add_argument('--memdir', default='.guardian_mem_brain')
    ap.add_argument('--llama', default='http://127.0.0.1:11435')
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--max_each', type=int, default=1200)
    ap.add_argument('--out-notes', default='.guardian_mem_brain/knowledge.notes.jsonl')
    ap.add_argument('--out-qa', default='.guardian_mem_brain/knowledge.qa.jsonl')
    args = ap.parse_args()

    topics = [ln.strip() for ln in pathlib.Path(args.topics).read_text(encoding='utf-8').splitlines() if ln.strip() and not ln.strip().startswith('#')]
    out_notes = pathlib.Path(args.out_notes)
    out_qa = pathlib.Path(args.out_qa)
    out_notes.parent.mkdir(parents=True, exist_ok=True)
    out_qa.parent.mkdir(parents=True, exist_ok=True)

    for t in topics:
        snips = scan_docs_for_topic(args.docs, t, k=args.k, max_each=args.max_each)
        if not snips:
            continue

        sys_msg, user_msg = build_note_prompt(t, snips)
        payload = {
            "prompt": f"{sys_msg}\n\nUser:\n{user_msg}\n\nAssistant:",
            "temperature": 0.2,
            "top_p": 0.8,
            "repeat_penalty": 1.12,
            "n_predict": 800,
            "stream": False,
            "cache_prompt": True,
            "stop": ["\n\n"]
        }
        # Try up to 2 attempts to get a clean note
        attempts = 0
        note: Dict[str, any] = {}
        while attempts < 2:
            obj = ask_llamacpp(args.llama, payload)
            text = obj.get('content') or obj.get('response') or ''
            try:
                note = json.loads(text)
            except Exception:
                # Build a minimal wrapper if model returned plain text
                note = {"sections": {"definition": text, "mechanism": "", "evidence": "", "open_questions": []}, "cites": []}

            # Validate and repair: no meta-instructions, has sections and some length
            sec = note.get('sections', {}) if isinstance(note, dict) else {}
            definition = str(sec.get('definition', ''))
            mechanism = str(sec.get('mechanism', ''))
            evidence = str(sec.get('evidence', ''))
            if has_meta_instructions(definition + mechanism + evidence) or len(definition) < 80:
                # Repair prompt to restate the note without meta commentary
                rep_prompt = (
                    'Rewrite the note as STRICT JSON with keys '
                    '{"sections": {"definition": "string", "mechanism": "string", "evidence": "string", "open_questions": ["string"]}, "cites": ["string"]}. '
                    'Do not mention JSON, instructions, or formatting. Keep 400–800 words (3–5 sentences per section). '
                    'Use bracketed citations [1],[2],[3] from the provided snippets. '
                )
                payload["prompt"] = f"{rep_prompt}\n\nUser:\n{user_msg}\n\nAssistant:"
                attempts += 1
                continue
            break

        # Ensure cites array reflects brackets used in text
        cites = note.get('cites') if isinstance(note, dict) else []
        if not isinstance(cites, list):
            cites = []
        all_txt = " ".join([definition, mechanism, evidence])
        nums = extract_cite_numbers(all_txt)
        if nums:
            note['cites'] = nums
        else:
            note['cites'] = cites
        card = {"topic": t, "note": note, "sources": [p for p,_ in snips]}
        out_notes.open('a', encoding='utf-8').write(json.dumps(card, ensure_ascii=False) + "\n")

        # Build Q/A
        note_text = "\n".join([
            str(note.get('sections', {}).get('definition', '')),
            str(note.get('sections', {}).get('mechanism', '')),
            str(note.get('sections', {}).get('evidence', '')),
            "\n".join(note.get('sections', {}).get('open_questions', []) or []),
        ])
        # Repair meta instructions in note_text if present
        if has_meta_instructions(note_text) or len(note_text) < 120:
            note_text = f"Definition: {definition}\nMechanism: {mechanism}\nEvidence: {evidence}"
        qa_prompt = build_qa_prompt(t, note_text)
        payload2 = {
            "prompt": qa_prompt,
            "temperature": 0.2,
            "top_p": 0.8,
            "repeat_penalty": 1.12,
            "n_predict": 600,
            "stream": False,
            "cache_prompt": True,
            "stop": ["\n\n"]
        }
        obj2 = ask_llamacpp(args.llama, payload2)
        text2 = obj2.get('content') or obj2.get('response') or ''
        try:
            qa = json.loads(text2)
        except Exception:
            qa = {"qa": []}
        out_qa.open('a', encoding='utf-8').write(json.dumps({"topic": t, "qa": qa.get('qa', [])}, ensure_ascii=False) + "\n")

    print(json.dumps({"topics": len(topics), "notes": str(out_notes), "qa": str(out_qa)}, indent=2))


if __name__ == '__main__':
    main()
