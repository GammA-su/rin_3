#!/usr/bin/env python3
"""
Autopilot: continuous discovery + promotion cycles.

Loop:
  1) Harvest candidate topics (docs/ and optional topics file).
  2) Run ACL novelty gate and promote novel concepts (persist to memory).
  3) Append daily evidence (metrics, ATF) and aggregate p-values from manifest.
  4) Run Gate v2 + Council (proof-cycle-multi) and bundle proof.
  5) Sleep, repeat.

This is a light autonomy layer over existing repo tools. It does not train models;
it accumulates concepts, admissions (ATF), and promotion decisions over time.
"""
from __future__ import annotations
import argparse, os, time, subprocess, pathlib, sys


def sh(cmd: list[str] | str, env: dict | None = None, check: bool = False) -> int:
    if isinstance(cmd, str):
        p = subprocess.run(cmd, shell=True, env=env)
    else:
        p = subprocess.run(cmd, env=env)
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.returncode


def scan_docs_for_topics(docs_dir: str) -> list[str]:
    out: list[str] = []
    p = pathlib.Path(docs_dir)
    if not p.exists():
        return out
    for f in p.rglob('*.txt'):
        name = f.stem.replace('_', ' ').strip()
        if name:
            out.append(name)
    return sorted(set(out))


def load_topics_file(path: str) -> list[str]:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    items = []
    for line in p.read_text(encoding='utf-8').splitlines():
        t = line.strip()
        if t:
            items.append(t)
    return items


def run_acl_for_topics(topics: list[str], memdir: str, novel_theta: float) -> None:
    pathlib.Path(memdir).mkdir(parents=True, exist_ok=True)
    for t in topics:
        cmd = [
            sys.executable, 'guardian_agi_min.py', '--task', 'acl',
            '--memdir', memdir, '--concept', t, '--novel-theta', str(novel_theta),
        ]
        print(f"[autopilot] ACL: {t}")
        sh(cmd)


def do_evidence_cycle(manifest: str, server: str) -> None:
    # Aggregate p-values from manifest
    sh(['make', 'pvals-from-manifest', f'MANIFEST={manifest}', 'OUT=files/configs/pvals.input.json'])
    # Metrics and ATF from artifact (if present)
    sh(['make', 'metrics-from-artifact', 'IN=artifacts/suite_full.json'])
    sh(['make', 'atf-from-artifact', 'IN=artifacts/suite_full.json'])
    # Promote + council + bundle
    env = os.environ.copy(); env['SERVER'] = server
    sh(['make', 'proof-cycle-multi', 'PVALS=files/configs/pvals.input.json'], env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memdir', default='.guardian_mem')
    ap.add_argument('--docs', default='docs', help='scan for *.txt as topic seeds')
    ap.add_argument('--topics', default='files/configs/topics.txt', help='optional topics file')
    ap.add_argument('--novel-theta', type=float, default=0.7)
    ap.add_argument('--manifest', default='files/configs/eval_sources.json')
    ap.add_argument('--server', default='http://127.0.0.1:8090')
    ap.add_argument('--interval-min', type=int, default=15, help='sleep minutes between cycles (0 = no sleep)')
    ap.add_argument('--cycles', type=int, default=-1, help='-1 for infinite')
    ap.add_argument('--hours', type=float, default=0.0, help='stop after H hours (overrides cycles if > 0)')
    args = ap.parse_args()

    # Seed status print
    print('[autopilot] starting')
    start = time.time()
    n = 0

    def time_left_ok() -> bool:
        if args.hours and args.hours > 0:
            return (time.time() - start) < (args.hours * 3600.0)
        return True

    import re
    inc_pat = os.getenv('TOPIC_INCLUDE', '').strip()
    exc_pat = os.getenv('TOPIC_EXCLUDE', '').strip()
    include_re = re.compile(inc_pat) if inc_pat else None
    exclude_re = re.compile(exc_pat) if exc_pat else None

    def _filter_topics(ts: list[str]) -> list[str]:
        out: list[str] = []
        for t in ts:
            if exclude_re and exclude_re.search(t):
                continue
            if include_re and not include_re.search(t):
                continue
            out.append(t)
        return out

    while (args.cycles < 0 or n < args.cycles) and time_left_ok():
        n += 1
        print(f"[autopilot] cycle {n}")
        # 1) harvest topics
        if os.getenv('TOPICS_ONLY', '') == '1':
            topics = sorted(set(load_topics_file(args.topics)))
        else:
            topics = sorted(set(scan_docs_for_topics(args.docs) + load_topics_file(args.topics)))
        topics = _filter_topics(topics)
        if topics:
            run_acl_for_topics(topics, args.memdir, args.novel_theta)
        else:
            print('[autopilot] no topics found (docs/topics file empty)')
        # 2) optional web enrichment + grounding (opt-in)
        if os.getenv('HARVEST_WEB') == '1' and os.getenv('ALLOW_NET') == '1':
            sh(['make', 'web-enrich-claims', f"MEMDIR={args.memdir}", f"DOCS={args.docs}", 'ONLY_MISSING=1'])
            sh(['make', 'ground-claims', f"MEMDIR={args.memdir}", f"DOCS={args.docs}"])
        # 3) evidence + promotion
        do_evidence_cycle(args.manifest, args.server)
        # 4) bundle (already in proof-cycle-multi) and status
        sh(['make', 'status'])
        # sleep (or continuous if interval==0)
        if args.interval_min > 0:
            print(f"[autopilot] sleep {args.interval_min} min")
            try:
                time.sleep(args.interval_min * 60)
            except KeyboardInterrupt:
                break
        else:
            print("[autopilot] continuous mode (no sleep)")

    print('[autopilot] done')


if __name__ == '__main__':
    main()
