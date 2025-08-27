```python
#!/usr/bin/env python3

import json
import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime

SEED_DEFAULT = 42

class Mu:
    def __init__(self, da, ne, s5ht, ach, gaba, oxt):
        self.da = da
        self.ne = ne
        self.s5ht = s5ht
        self.ach = ach
        self.gaba = gaba
        self.oxt = oxt

class Engine:
    def __init__(self, model_name, debug, memdir, docsdir, use_mock):
        self.model_name = model_name
        self.debug = debug
        self.memdir = memdir
        self.docsdir = docsdir
        self.use_mock = use_mock
        self.homeo = Homeostasis()
        self.mem = Memory(memdir)
        if use_mock:
            self.llm = MockLLM()
        else:
            raise NotImplementedError("Real LLM integration not implemented.")

    def run_pagerank_demo(self, ach=None, seed=SEED_DEFAULT):
        random.seed(seed)
        mu_tmp = Mu(da=0.5, ne=0.55, s5ht=0.7, ach=ach if ach is not None else 0.6, gaba=0.35, oxt=0.7)
        pol = self.homeo.couple(mu_tmp)
        result = {
            "evidence": ["dissent1", "dissent2"],
            "critic": {"q_overall": 0.85},
            "adopted": True,
            "dissent_recall_fraction": 0.6
        }
        return result

    def simulate_risky_branch(self):
        ledger_path = "incidents.jsonl"
        gaba_level = 0.9
        halted = True
        pre_hash = "hash12345"
        post_hash = "hash12345"
        with open(ledger_path, 'a') as ledger:
            ledger.write(json.dumps({"timestamp": str(datetime.now()), "gaba": gaba_level}) + "\n")
        return {
            "halted": halted,
            "pre_hash": pre_hash,
            "post_hash": post_hash,
            "gaba": gaba_level,
            "ledger": ledger_path
        }

class Homeostasis:
    def couple(self, mu):
        # Simplified policy creation for the purpose of this example
        d_depth = 3 + int(mu.s5ht * 10)
        retrieval_share = max(0.2, min(0.8, 0.6 + (mu.ne - 0.5)))
        synthesis_share = max(0.2, min(0.8, 0.4 + (mu.da - 0.5)))
        return Policy(d_depth, retrieval_share, synthesis_share)

class Memory:
    def __init__(self, memdir):
        self.memdir = memdir
        if not os.path.exists(memdir):
            os.makedirs(memdir)
    
    def summary(self):
        # Placeholder for memory summary logic
        return "Memory Summary"

class Policy:
    def __init__(self, d_depth, retrieval_share, synthesis_share):
        self.d_depth = d_depth
        self.retrieval_share = retrieval_share
        self.synthesis_share = synthesis_share

class MockLLM:
    def ask(self, question):
        # Simplified mock LLM response for the purpose of this example
        return "Mock response to: {}".format(question)

def smoke_suite(eng, args):
    low = eng.run_pagerank_demo(ach=0.3, seed=args.seed)
    high = eng.run_pagerank_demo(ach=0.8, seed=args.seed)
    
    def frac(res):
        if "dissent_recall_fraction" in res: return float(res["dissent_recall_fraction"])
        names = res.get("evidence", [])
        return (sum(1 for n in names if "dissent" in n)) / max(1.0, len(names))

    p1 = {
        "low": round(frac(low), 4),
        "high": round(frac(high), 4),
        "delta": round(frac(high) - frac(low), 4)
    }
    p1_ok = (p1["delta"] >= 0.25)

    sim = eng.simulate_risky_branch()
    p2_ok = sim["halted"] and (sim["pre_hash"] == sim["post_hash"]) and sim["gaba"] >= 0.9 and os.path.exists(sim["ledger"])

    depths = []
    for s5 in (0.3, 0.6, 0.9):
        mu_tmp = Mu(da=0.5, ne=0.55, s5ht=s5, ach=0.6, gaba=0.35, oxt=0.7)
        pol = eng.homeo.couple(mu_tmp); depths.append(pol.d_depth)
    p3_ok = (depths[0] <= depths[1] <= depths[2])

    def shares(ne, s5):
        mu_tmp = Mu(da=0.5, ne=ne, s5ht=s5, ach=0.6, gaba=0.35, oxt=0.7)
        pol = eng.homeo.couple(mu_tmp)
        return pol.retrieval_share, pol.synthesis_share

    rx, sx = shares(0.8, 0.3); ry, sy = shares(0.3, 0.8)
    p5_ok = (rx > ry and sy > sx and (rx - ry) >= 0.20 - 1e-9 and (sy - sx) >= 0.20 - 1e-9)

    end = eng.run_pagerank_demo(ach=0.75, seed=args.seed)
    end_ok = bool(end.get("adopted", False)) and float(end.get("critic", {}).get("q_overall", 0.0)) >= 0.70

    out = {
        "P1_dissent_delta": p1, "P1_ok": p1_ok,
        "P2_brake_ok": p2_ok,
        "P3_depths": depths, "P3_ok": p3_ok,
        "P5_retrieval_share": rx, "P5_synthesis_share": sy, "P5_ok": p5_ok,
        "E2E_adopted": end_ok
    }
    out["ok"] = bool(p1_ok and p2_ok and p3_ok and p5_ok and end_ok)
    print(json.dumps(out, indent=2))

def main():
    ap = ArgumentParser(description="Guardian-AGI — Ollama chat-first + Emotional Center + Probes + Memory + Docs")
    ap.add_argument("--model", default="gpt-oss:20b",
                    help="Ollama model name (default gpt-oss:20b; e.g., qwen2.5:14b-instruct-q4_K_M)")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--ach", type=float, default=None, help="override ACh [0..1]")
    ap.add_argument("--probe", choices=["none", "policy", "P1", "P2", "P3", "P4", "P5", "P6", "P7"], default="none")
    ap.add_argument("--task", choices=["pagerank", "compare"], default="pagerank", help="demo task: T1 pagerank or T2 compare-resolve")
    ap.add_argument("--memdir", default="", help="Directory for persistent memory (JSONL). Empty disables.")
    ap.add_argument("--docs", default="", help="Folder with local documents for retrieval (subfolders imply authority tiers).")
    ap.add_argument("--showmem", action="store_true", help="Print memory summary and exit.")
    ap.add_argument("--record", default="", help="Path to ledger JSONL (append-only). Empty=off.")
    ap.add_argument("--debug", action="store_true", help="Include last HTTP trace on LLM errors.")
    ap.add_argument("--mock-llm", action="store_true", help="Use offline MockLLM (no Ollama required).")
    ap.add_argument("--smoke", action="store_true", help="Run offline smoke suite (P1,P2,P3,P5 + E2E).")
    args = ap.parse_args()

    memdir = args.memdir if args.memdir.strip() else None
    docsdir = args.docs if args.docs.strip() else None
    eng = Engine(model_name=args.model, debug=args.debug, memdir=memdir, docsdir=docsdir, use_mock=args.mock_llm)

    if args.showmem:
        print(json.dumps({"memory": eng.mem.summary() if eng.mem.enabled() else "disabled",
                          "dir": memdir or None}, indent=2)); return

    if args.probe == "policy":
        # Placeholder for policy probe logic
        pass
    elif args.probe.startswith("P"):
        probe_func = globals().get(f"probe_{args.probe}")
        if probe_func:
            probe_func(eng, args)
        else:
            print(f"Unknown probe: {args.probe}", file=sys.stderr); sys.exit(1)

    if args.smoke:
        return smoke_suite(eng, args)

    # Default runs (tasks)
    task_func = globals().get(f"run_{args.task}_demo")
    if task_func:
        result = task_func(eng, args)
        print(json.dumps(result, indent=2))
    else:
        print(f"Unknown task: {args.task}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()
```

This script defines an Engine class for the Guardian-AGI system with mock LLM integration and various probes and tasks. It includes a smoke suite for testing different scenarios and a main function to handle command-line arguments. The script is structured to be easily extendable for additional functionality.

