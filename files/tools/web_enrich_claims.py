#!/usr/bin/env python3
"""
Fetch supporting evidence from the web for ACL claims and append updated
claims (with sources and stance) into the memory JSONL.

Safety/replicability:
- Opt-in networking: requires ALLOW_NET=1 in the environment.
- Writes fetched texts under docs/web/* so future runs can operate offline.
- Heuristic tier mapping by domain; stance derived from text/path.

Providers (no API keys required):
- Wikipedia (MediaWiki REST summary API)
- arXiv (export API; Atom XML)
"""
from __future__ import annotations
import os, sys, json, pathlib, re, time, urllib.parse, urllib.request, urllib.error, xml.etree.ElementTree as ET
from typing import Dict, Any, List, Tuple


def must_allow_net() -> None:
    if os.getenv('ALLOW_NET') != '1':
        print('[web-enrich] networking disabled. Set ALLOW_NET=1 to enable.')
        raise SystemExit(2)


def slugify(s: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9]+', '-', (s or '').strip())
    return re.sub(r'-+', '-', s).strip('-').lower() or 'doc'


def _ua_headers() -> dict:
    ua = os.getenv('UA', 'Triforce-Prophet/1.0 (+local; contact: local)')
    return {
        'User-Agent': ua,
        'Accept': 'application/json, text/plain;q=0.9, */*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'close',
    }


def http_get_json(url: str, timeout: int = 15) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=_ua_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read().decode('utf-8', errors='ignore')
        try:
            return json.loads(data)
        except Exception:
            return {}
    except urllib.error.HTTPError as e:
        print(f"[web-enrich] HTTP {e.code} for {url}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"[web-enrich] fetch error for {url}: {e}", file=sys.stderr)
        return {}


def http_get_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers=_ua_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='ignore')
    except urllib.error.HTTPError as e:
        print(f"[web-enrich] HTTP {e.code} for {url}", file=sys.stderr)
        return ''
    except Exception as e:
        print(f"[web-enrich] fetch error for {url}: {e}", file=sys.stderr)
        return ''


def wiki_search(query: str, k: int = 2) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    k = max(1, int(k))
    # Primary: action=query search
    try:
        qs = urllib.parse.urlencode({'action': 'query', 'list': 'search', 'srsearch': query, 'format': 'json', 'srlimit': k})
        url = f'https://en.wikipedia.org/w/api.php?{qs}'
        obj = http_get_json(url)
        hits = obj.get('query', {}).get('search', [])
        for hit in hits[:k]:
            title = hit.get('title')
            if not title:
                continue
            tenc = urllib.parse.quote(title)
            summ = http_get_json(f'https://en.wikipedia.org/api/rest_v1/page/summary/{tenc}')
            extract = summ.get('extract') or ''
            if extract:
                out.append((f'https://en.wikipedia.org/wiki/{tenc}', extract))
                time.sleep(0.3)
        if out:
            return out
    except Exception:
        pass
    # Fallback: REST search API
    try:
        qs2 = urllib.parse.urlencode({'q': query, 'limit': k})
        url2 = f'https://en.wikipedia.org/w/rest.php/v1/search/title?{qs2}'
        obj2 = http_get_json(url2)
        items = obj2.get('pages', []) or obj2.get('results', []) or []
        for it in items[:k]:
            title = it.get('title') or it.get('key')
            if not title:
                continue
            tenc = urllib.parse.quote(title)
            summ = http_get_json(f'https://en.wikipedia.org/api/rest_v1/page/summary/{tenc}')
            extract = summ.get('extract') or ''
            if extract:
                out.append((f'https://en.wikipedia.org/wiki/{tenc}', extract))
                time.sleep(0.3)
    except Exception:
        return out
    return out


def arxiv_search(query: str, k: int = 1) -> List[Tuple[str, str]]:
    # Atom XML feed
    q = urllib.parse.quote(query)
    url = f'http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={max(1,int(k))}'
    text = http_get_text(url)
    if not text:
        return []
    try:
        root = ET.fromstring(text)
    except Exception:
        return []
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    out: List[Tuple[str, str]] = []
    for ent in root.findall('a:entry', ns)[:k]:
        link = ent.find("a:id", ns)
        summ = ent.find("a:summary", ns)
        href = link.text.strip() if link is not None and link.text else ''
        extract = summ.text.strip() if summ is not None and summ.text else ''
        if href and extract:
            out.append((href, extract))
    return out


AUTH_TIER_BY_DOMAIN = {
    'wikipedia.org': 3,
    'arxiv.org': 2,
}


def tier_from_url(url: str) -> int:
    try:
        host = urllib.parse.urlparse(url).hostname or ''
    except Exception:
        return 4
    host = host.lower()
    for dom, tier in AUTH_TIER_BY_DOMAIN.items():
        if host.endswith(dom):
            return tier
    return 4


def stance_from_text(url: str, text: str) -> str:
    p = url.lower()
    if 'dissent' in p or re.search(r'\b(however|contradict|misleading|not simply)\b', (text or '').lower()):
        return 'con'
    if any(k in p for k in ('media', 'blog', 'forum', 'wikipedia.org')):
        return 'neutral'
    return 'pro'


def write_doc_copy(base_dir: pathlib.Path, url: str, text: str) -> pathlib.Path:
    # Place under docs/web/<tier_name>/slug.txt
    tier = tier_from_url(url)
    tier_name = {1:'primary',2:'peer',3:'media',4:'blog',5:'community'}.get(tier, 'blog')
    slug = slugify(url.split('//')[-1])
    dest = base_dir / 'web' / tier_name / f'{slug}.txt'
    dest.parent.mkdir(parents=True, exist_ok=True)
    # prefix with URL for traceability
    dest.write_text(f'{url}\n\n{text}', encoding='utf-8')
    return dest


def load_claims(memdir: pathlib.Path) -> List[Dict[str, Any]]:
    path = memdir / 'claims.jsonl'
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def append_claim(memdir: pathlib.Path, claim: Dict[str, Any]) -> None:
    path = memdir / 'claims.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(claim, ensure_ascii=False) + '\n')


def enrich_one(claim: Dict[str, Any], out_docs: pathlib.Path, k_total: int = 3) -> Dict[str, Any]:
    text = claim.get('text', '')
    sources: List[Dict[str, Any]] = []
    # 1) Wikipedia summaries
    for url, extract in wiki_search(text, k=2):
        p = write_doc_copy(out_docs, url, extract)
        sources.append({'url': str(p), 'domain_tier': tier_from_url(url)})
        if len(sources) >= k_total:
            break
    if len(sources) < k_total:
        # 2) arXiv abstracts
        for url, extract in arxiv_search(text, k=2):
            p = write_doc_copy(out_docs, url, extract)
            sources.append({'url': str(p), 'domain_tier': tier_from_url(url)})
            if len(sources) >= k_total:
                break
    if not sources:
        return claim
    # stance vote
    votes = {'pro':0,'neutral':0,'con':0}
    for s in sources:
        try:
            txt = pathlib.Path(s['url']).read_text(encoding='utf-8')
        except Exception:
            txt = ''
        st = stance_from_text(s['url'], txt)
        votes[st] = votes.get(st, 0) + 1
    stance = max(votes.items(), key=lambda kv: kv[1])[0]
    newc = dict(claim)
    newc['sources'] = sources
    newc['stance'] = stance
    return newc


def main():
    must_allow_net()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--memdir', default='.guardian_mem')
    ap.add_argument('--docs', default='docs')
    ap.add_argument('-k', '--k', type=int, default=3)
    ap.add_argument('--only-missing', action='store_true', help='enrich only claims without sources')
    args = ap.parse_args()

    memdir = pathlib.Path(args.memdir)
    docs = pathlib.Path(args.docs)
    claims = load_claims(memdir)
    updated = 0
    for c in claims:
        if args.only_missing and (c.get('sources')):
            continue
        nc = enrich_one(c, docs, k_total=max(1,int(args.k)))
        if nc is not c and nc.get('sources'):
            append_claim(memdir, nc)
            updated += 1
    print(json.dumps({'updated': updated, 'memdir': str(memdir), 'docs': str(docs)}))


if __name__ == '__main__':
    main()
