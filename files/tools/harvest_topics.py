#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, pathlib, urllib.parse, urllib.request, urllib.error, time
from typing import List, Dict, Tuple, Set


def must_allow_net():
    if os.getenv('ALLOW_NET') != '1':
        print('[harvest-topics] networking disabled. Set ALLOW_NET=1.', file=sys.stderr)
        sys.exit(2)


def ua_headers() -> dict:
    return {
        'User-Agent': os.getenv('UA', 'Triforce-Prophet/1.0 (+local; contact: local)') ,
        'Accept': 'application/json, text/plain;q=0.9, */*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'close',
    }


def http_get_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers=ua_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read().decode('utf-8', errors='ignore')
        return json.loads(data) if data else {}
    except Exception as e:
        print(f"[harvest-topics] GET JSON failed {url}: {e}", file=sys.stderr)
        return {}


def http_get_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers=ua_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"[harvest-topics] GET TEXT failed {url}: {e}", file=sys.stderr)
        return ''


def slug(s: str) -> str:
    import re
    return re.sub(r'[^a-zA-Z0-9]+', '-', s.strip()).strip('-').lower() or 'doc'


def wiki_search_titles(query: str, limit: int) -> List[str]:
    qs = urllib.parse.urlencode({'action':'query','list':'search','srsearch':query,'format':'json','srlimit':max(1,int(limit))})
    url = f'https://en.wikipedia.org/w/api.php?{qs}'
    obj = http_get_json(url)
    out: List[str] = []
    for hit in obj.get('query',{}).get('search',[])[:limit]:
        t = hit.get('title')
        if t:
            out.append(t)
    return out


def wiki_extract_intro_sentences(title: str, sentences: int = 4) -> Tuple[str, str]:
    tenc = urllib.parse.quote(title)
    # Use REST summary first (usually first paragraph)
    summ = http_get_json(f'https://en.wikipedia.org/api/rest_v1/page/summary/{tenc}')
    extract = summ.get('extract') or ''
    url = f'https://en.wikipedia.org/wiki/{tenc}'
    if extract:
        return url, extract
    # Fallback to extracts API
    qs = urllib.parse.urlencode({'action':'query','prop':'extracts','exintro':1,'explaintext':1,'exsentences':max(1,int(sentences)),'format':'json','titles':title})
    obj = http_get_json(f'https://en.wikipedia.org/w/api.php?{qs}')
    pages = obj.get('query',{}).get('pages',{})
    for _, page in pages.items():
        ext = page.get('extract')
        if ext:
            return url, ext
    return url, ''


def wiki_related_titles(title: str, limit: int) -> List[str]:
    tenc = urllib.parse.quote(title)
    obj = http_get_json(f'https://en.wikipedia.org/api/rest_v1/page/related/{tenc}')
    out: List[str] = []
    items = obj.get('pages') or obj.get('related') or obj.get('suggested') or []
    for it in items[:limit]:
        tt = it.get('title') or it.get('display') or it.get('extract')
        if isinstance(tt, str):
            out.append(tt)
    return out


def arxiv_search(query: str, limit: int) -> List[Tuple[str,str]]:
    q = urllib.parse.quote(query)
    url = f'http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={max(1,int(limit))}'
    text = http_get_text(url)
    if not text:
        return []
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(text)
    except Exception:
        return []
    ns = {'a':'http://www.w3.org/2005/Atom'}
    out: List[Tuple[str,str]] = []
    for ent in root.findall('a:entry', ns)[:limit]:
        link = ent.find('a:id', ns)
        summ = ent.find('a:summary', ns)
        href = link.text.strip() if link is not None and link.text else ''
        extract = summ.text.strip() if summ is not None and summ.text else ''
        if href and extract:
            out.append((href, extract))
    return out


def write_doc(base: pathlib.Path, tier: str, name: str, url: str, text: str) -> None:
    dest = base / 'web' / tier / f'{slug(name)}.txt'
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = f'{url}\n\n{text.strip()}'
    dest.write_text(content, encoding='utf-8')


def main():
    must_allow_net()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--topics', default='files/configs/topics.brain.txt')
    ap.add_argument('--docs', default='docs/brain')
    ap.add_argument('--wiki-k', type=int, default=12)
    ap.add_argument('--related-k', type=int, default=6)
    ap.add_argument('--arxiv-k', type=int, default=6)
    args = ap.parse_args()

    topics = [ln.strip() for ln in pathlib.Path(args.topics).read_text(encoding='utf-8').splitlines() if ln.strip() and not ln.strip().startswith('#')]
    base = pathlib.Path(args.docs)
    total = 0

    for t in topics:
        seen: Set[str] = set()
        # Wikipedia search
        titles = wiki_search_titles(t, args.wiki_k)
        for title in titles:
            url, extract = wiki_extract_intro_sentences(title, sentences=4)
            if not extract:
                continue
            key = f'media:{title}'
            if key in seen:
                continue
            seen.add(key)
            write_doc(base, 'media', f'en-wikipedia-org-wiki-{title}', url, extract)
            total += 1
            time.sleep(0.2)
        # Related pages for the best title
        if titles:
            rel = wiki_related_titles(titles[0], args.related_k)
            for title in rel:
                url, extract = wiki_extract_intro_sentences(title, sentences=4)
                if not extract:
                    continue
                key = f'media:{title}'
                if key in seen:
                    continue
                seen.add(key)
                write_doc(base, 'media', f'en-wikipedia-org-wiki-{title}', url, extract)
                total += 1
                time.sleep(0.2)
        # arXiv abstracts
        for href, summ in arxiv_search(t, args.arxiv_k):
            name = href.split('/')[-1]
            write_doc(base, 'peer', f'arxiv-org-abs-{name}', href, summ)
            total += 1
            time.sleep(0.2)

    print(json.dumps({'topics': len(topics), 'docs_written': total, 'docs_dir': str(base)}, indent=2))


if __name__ == '__main__':
    main()

