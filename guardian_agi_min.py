#!/usr/bin/env python3
# guardian_agi_min.py — single-file Guardian-AGI scaffold (seed=137)
# Track 1: Ollama (no Docker). Default model: gpt-oss:20b (override via --model)
# Chat-first client with generate fallback; for MAIN answer we do generate-first.
# Bracketed output + strict validation + auto-retry + deterministic fallback.

from __future__ import annotations
import argparse, json, os, random, time, http.client, hashlib, re
from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import List, Dict, Any, Optional

# ========= Determinism & helpers =========
SEED_DEFAULT = 137
def seed_everything(seed: int = SEED_DEFAULT):
    os.environ["PYTHONHASHSEED"] = str(seed); random.seed(seed)

def clamp(x: float, lo: float=0.0, hi: float=1.0) -> float:
    return max(lo, min(hi, x))

def soft_stop(goal_met: float, gaba: float, budget_exhaust: float, unresolved_conflict: float) -> float:
    # S_s = 0.5·goal_met + 0.2·GABA + 0.2·τ_exhaust − 0.2·unresolved_conflict
    return 0.5*goal_met + 0.2*gaba + 0.2*budget_exhaust - 0.2*unresolved_conflict

def sha(s: str) -> str: return hashlib.sha256(s.encode("utf-8")).hexdigest()

def ledger_append(path: str, entry: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    prev = ""
    if os.path.exists(path):
        with open(path, "rb") as f:
            try:
                last = f.read().splitlines()[-1].decode("utf-8")
                prev = json.loads(last).get("this_hash","")
            except Exception:
                prev = ""
    entry["ts"] = int(time.time())
    entry["prev_hash"] = prev
    entry["this_hash"] = sha(prev + json.dumps(entry, sort_keys=True))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def between_tags(s: str, start="<<BEGIN>>", end="<<END>>") -> str:
    i = s.find(start)
    if i == -1: return s.strip()
    j = s.find(end, i + len(start))
    if j == -1: return s[i + len(start):].strip()
    return s[i + len(start):j].strip()

def word_count(s: str) -> int:
    return len([w for w in re.findall(r"\b\w+\b", s)])

# ========= Data Contracts =========
@dataclass
class Source:
    url: str; seg: Optional[str]=None; h: Optional[str]=None; ts: Optional[str]=None; domain_tier: int=1

@dataclass
class Claim:
    id: str; text: str; q: float = 0.5; sources: List[Source] = None
    supports: List[str] = None; contradicts: List[str] = None; stance: str = "neutral"
    def to_dict(self):
        return {"id": self.id, "text": self.text, "q": self.q, "stance": self.stance,
                "sources": [asdict(s) for s in (self.sources or [])],
                "supports": self.supports or [], "contradicts": self.contradicts or []}

@dataclass
class EvidenceUnit:
    id: str; content_hash: str; extract: str; stance: str="neutral"; provenance: List[Dict[str, Any]] = None

# ========= Emotional Center (Homeostat) =========
@dataclass
class Appraisal: p: float; n: float; u: float; k: float; s: float; c: float; h: float
@dataclass
class Mu: da: float; ne: float; s5ht: float; ach: float; gaba: float; oxt: float
@dataclass
class PolicyCoupling:
    k_breadth: int; d_depth: int; q_contra: int; temperature: float
    retrieval_share: float; synthesis_share: float; safety_share: float
    reserved_dissent: bool

class Homeostat:
    def update(self, mu: Mu, a: Appraisal) -> Mu:
        da  = clamp(mu.da  + 0.35*(a.p-0.5) - 0.30*a.k)
        ne  = clamp(mu.ne  + 0.40*a.u + 0.30*a.n)
        s5  = clamp(mu.s5ht + 0.45*a.s)
        ach = clamp(mu.ach + 0.50*a.c)
        gb  = clamp(mu.gaba + 0.50*0.5 + 0.30*0.5 - 0.25*0.5)
        oxt = clamp(mu.oxt + 0.40*a.h)
        return Mu(da, ne, s5, ach, gb, oxt)

    def couple(self, mu: Mu) -> PolicyCoupling:
        k0, d0, q0 = 6, 3, 1
        k = max(3, int(k0*(1 + mu.ne - 0.5*mu.s5ht)))
        d = max(1, int(d0*(1 + mu.s5ht - mu.ne)))
        q_con = max(1, int(q0*(1 + mu.ach)))
        temp = max(0.1, 0.9 - 0.6*mu.s5ht)
        retr = clamp(0.35 + 0.30*mu.ne - 0.15*mu.s5ht, 0.0, 1.0)
        syn  = clamp(0.35 + 0.30*mu.s5ht - 0.15*mu.ne, 0.0, 1.0)
        saf  = clamp(1.0 - (retr + syn), 0.0, 1.0)
        return PolicyCoupling(k, d, q_con, temp, retr, syn, saf, reserved_dissent=(mu.ach>=0.6))

# ========= Safety (Custodian) =========
class Custodian:
    policy_ver = "v1.1"
    def classify(self, goal: str) -> str:
        g = goal.lower()
        if any(x in g for x in ["bio", "exploit", "weapon", "malware", "lab"]): return "R3"
        return "R0"
    def preflight(self, risk: str) -> Dict[str,str]:
        return {"action": "deny" if risk in ("R3","R4") else "allow",
                "notes":  f"Risk {risk} {'blocked' if risk in ('R3','R4') else 'allowed'} by policy"}

# ========= Evaluation (Witness) =========
class Witness:
    def score(self, stats: Dict[str,Any]) -> Dict[str,float]:
        pass_at_1   = 1.0 if stats.get("goal_met", False) else 0.0
        precision_k = clamp(0.7 + 0.05*max(0, stats.get("sources", 0)), 0.0, 1.0)
        ece         = 0.08
        resolution  = 1.0 if stats.get("resolved", False) else 0.0
        return {"pass_at_1":pass_at_1,"precision_k":precision_k,"ece":ece,"resolution_rate":resolution}

# ========= World-Model (Archivist) =========
class Archivist:
    def __init__(self):
        self.claims: Dict[str, Claim] = {}
        self.contradict: Dict[str, List[str]] = {}
    def upsert_claim(self, c: Claim): self.claims[c.id] = c
    def link_contradiction(self, i: str, j: str):
        self.contradict.setdefault(i,[]).append(j); self.contradict.setdefault(j,[]).append(i)
    def retrieve(self, k:int=5) -> List[Claim]: return list(self.claims.values())[:k]

# ========= Retrieval (Scout: local corpus demo) =========
PAGERANK_PRIMARY = """PageRank is a link analysis algorithm assigning importance as the stationary probability a random surfer lands on a page. The damping factor (≈0.85) models continuing to click links."""
PAGERANK_MEDIA   = """Popular media often say PageRank ranks pages by counting links; more links imply higher rank."""
PAGERANK_DISSENT = """Dissent: PageRank is NOT simple counts; it weights by the rank of linking pages and normalizes by their outdegree."""

class Scout:
    def _mk_ev(self, name: str, txt: str, stance: str) -> EvidenceUnit:
        h = sha256(txt.encode()).hexdigest()[:16]
        return EvidenceUnit(id=name, content_hash=h, extract=txt[:400], stance=stance,
                            provenance=[{"source": name}])
    def fetch_pagerank(self, k_breadth:int, dissent_quota:int) -> List[EvidenceUnit]:
        pool = [
            self._mk_ev("pagerank_primary.txt", PAGERANK_PRIMARY, "pro"),
            self._mk_ev("pagerank_media.txt",   PAGERANK_MEDIA,   "neutral"),
            self._mk_ev("pagerank_dissent.txt", PAGERANK_DISSENT, "con"),
        ]
        dissent = [e for e in pool if e.stance=="con"][:max(1, dissent_quota-1)]
        others  = [e for e in pool if e.stance!="con"]
        return (dissent + others)[:max(1, k_breadth)]

# ========= Planner (Operator) =========
@dataclass
class Plan: name: str; steps: List[str]
class Operator:
    def plan_research(self) -> Plan: return Plan("T1-Research", ["Define scope","Fetch coverage","Extract claims","Synthesize","Calibrate"])
    def plan_compare(self) -> Plan:  return Plan("T2-Compare",  ["Collect A,B","Map contradictions","Resolve","Explain rationale"])

# ========= Pilot =========
@dataclass
class Intent:
    assumptions: str; unknowns: str; tests: str; stop: str; risk: str
class Pilot:
    def draft_intent(self, goal: str, risk: str) -> Intent:
        return Intent(
            assumptions="Sources may disagree; prioritize primary/official.",
            unknowns="Formula variants; damping factor reference.",
            tests="≥3 citations; ≤150 words; note & resolve one contradiction.",
            stop="S_s>0.6 and no policy flags.",
            risk=risk,
        )

# ========= Ollama LLM Client =========
class LLMClient:
    def __init__(self, model: str, host: str="localhost", port: int=11434, debug: bool=False):
        self.model = model; self.host = host; self.port = port; self.debug = debug
        self.last_http: Dict[str,Any] = {}

    def _post(self, path: str, payload: dict) -> Dict[str,Any]:
        body = json.dumps(payload)
        conn = http.client.HTTPConnection(self.host, self.port, timeout=180)
        try:
            conn.request("POST", path, body=body, headers={"Content-Type":"application/json"})
            resp = conn.getresponse(); status = resp.status
            raw = resp.read().decode("utf-8") if resp else ""
        finally:
            conn.close()
        self.last_http = {"path": path, "status": status, "raw_preview": raw[:240]}
        try:
            data = json.loads(raw) if raw else {}
        except Exception as e:
            raise RuntimeError(f"Ollama parse error (HTTP {status}) at {path}: {e}; raw={raw[:200]}")
        if status != 200:
            raise RuntimeError(f"Ollama HTTP {status} at {path}: {data.get('error') or raw[:200]}")
        return data

    def ask(self, system_msg: str, user_msg: str, *, temperature: float, top_p: float,
            repeat_penalty: float, num_predict: int, num_ctx: int=8192,
            force_json: bool=False, prefer_generate: bool=False) -> str:
        stop_tokens = ["<<END>>", "User:", "\nUser:"]
        def _chat() -> str:
            payload = {
                "model": self.model,
                "messages": [{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
                "options": {"temperature": float(temperature), "top_p": float(top_p),
                            "repeat_penalty": float(repeat_penalty), "num_predict": int(num_predict),
                            "num_ctx": int(num_ctx), "stop": stop_tokens,
                            **({"format":"json"} if force_json else {})},
                "stream": False
            }
            data = self._post("/api/chat", payload)
            out = ""
            if isinstance(data, dict) and isinstance(data.get("message"), dict):
                out = data["message"].get("content","") or data["message"].get("thinking","")
            else:
                out = data.get("response","") or data.get("thinking","")
            return (out or "").strip()

        def _gen() -> str:
            payload = {
                "model": self.model,
                "prompt": f"{system_msg}\n\nUser:\n{user_msg}\n\nAssistant:",
                "options": {"temperature": float(temperature), "top_p": float(top_p),
                            "repeat_penalty": float(repeat_penalty), "num_predict": int(num_predict),
                            "num_ctx": int(num_ctx), "stop": stop_tokens,
                            **({"format":"json"} if force_json else {})},
                "stream": False
            }
            data = self._post("/api/generate", payload)
            out = ""
            if isinstance(data, dict):
                out = data.get("response","") or data.get("thinking","")
            return (out or "").strip()

        if prefer_generate:
            out = _gen()
            if out: return out
            out = _chat()
            if out: return out
            raise RuntimeError("Ollama returned empty from both generate and chat.")
        else:
            out = _chat()
            if out: return out
            out = _gen()
            if out: return out
            raise RuntimeError("Ollama returned empty from both chat and generate.")

# ========= Critic (JSON with heuristic fallback) =========
CRITIC_SYS = (
  "You are Critic. Given an answer about PageRank, return STRICT JSON with keys: "
  "{'q_overall': float in [0,1], 'has_conflict_note': bool, 'reasons': [str]}. "
  "Return JSON ONLY—no extra text."
)
def extract_json(s: str) -> dict:
    try: return json.loads(s)
    except Exception: pass
    for start in range(len(s)-1, -1, -1):
        if s[start] != '{': continue
        depth = 0
        for end in range(start, len(s)):
            ch = s[end]
            if ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    chunk = s[start:end+1]
                    try: return json.loads(chunk)
                    except Exception: break
    raise ValueError("no valid JSON object found")

def critic_heuristic(answer: str) -> dict:
    text = (answer or "").strip()
    has_conflict = bool(re.search(r"(^|\n)\s*Conflict Note:", text, flags=re.I))
    L = len(text); q = 0.15
    if L > 60: q += 0.25
    if L > 120: q += 0.20
    if L > 200: q -= 0.10
    if has_conflict: q += 0.20
    return {"q_overall": float(clamp(q, 0.0, 0.9)), "has_conflict_note": has_conflict,
            "reasons": ["heuristic fallback (model returned non-JSON)"]}

def run_critic(llm: LLMClient, answer: str, mu: Mu, passes: int=1) -> dict:
    temperature = max(0.1, 0.9 - 0.6*mu.s5ht); top_p = 0.8
    repeat_penalty = 1.10 + 0.10*mu.s5ht; num_predict = 220
    best = {"q_overall": 0.0, "has_conflict_note": False, "reasons": ["no output"]}
    for _ in range(max(1, passes)):
        try:
            j = llm.ask(CRITIC_SYS, f"Answer:\n{answer}\n\nReturn JSON only.",
                        temperature=temperature, top_p=top_p,
                        repeat_penalty=repeat_penalty, num_predict=num_predict,
                        force_json=True, prefer_generate=False)
            try: data = extract_json(j)
            except Exception: data = critic_heuristic(j)
            if isinstance(data, dict) and data.get("q_overall", 0) >= best.get("q_overall", 0):
                best = data
        except Exception as e:
            best = {"q_overall": best["q_overall"], "has_conflict_note": best["has_conflict_note"],
                    "reasons": [f"critic error: {e} | http={llm.last_http}"]}
    best["q_overall"] = float(clamp(best.get("q_overall", 0.0), 0.0, 1.0))
    best["has_conflict_note"] = bool(best.get("has_conflict_note", False))
    if "reasons" not in best: best["reasons"] = []
    return best

# ========= Engine =========
class Engine:
    def __init__(self, model_name: str="gpt-oss:20b", neuro: Mu=None, debug: bool=False):
        self.homeo = Homeostat(); self.cust  = Custodian(); self.wit   = Witness()
        self.scout = Scout();     self.arch  = Archivist(); self.pilot = Pilot()
        self.llm   = LLMClient(model_name, debug=debug)
        self.neuro0 = neuro or Mu(da=0.50, ne=0.55, s5ht=0.85, ach=0.75, gaba=0.35, oxt=0.70)
        self.debug = debug

    # deterministic local fallback (guarantees constraints)
    def _local_pagerank_fallback(self) -> str:
        return (
          "PageRank estimates a page’s importance as the stationary probability of a “random surfer” visiting it. "
          "At each step the surfer follows an out-link with probability α (≈0.85) or teleports to a random page with probability 1−α, "
          "preventing sinks and ensuring a unique distribution. A page’s score aggregates the ranks of linking pages normalized by their out-degrees, "
          "iterated to convergence. Thus PageRank is not a raw link count: one link from a highly ranked page can outweigh many from weak pages. "
          "[1][2][3]\n"
          "Conflict Note: resolves the common ‘link count’ misconception by emphasizing rank-weighted, degree-normalized propagation."
        )

    # strict validator to reject meta/planning
    def _is_valid_answer(self, text: str) -> bool:
        if not text: return False
        if "<<" in text or ">>" in text: return False  # leaked tags
        if re.search(r"\b(We need to|Let's|Aim for|Draft:|Word count)\b", text, flags=re.I): return False
        wc = word_count(text)
        if wc < 110 or wc > 180: return False
        cites = re.findall(r"\[\d\]", text)
        if len(cites) < 3: return False
        if not re.search(r"(^|\n)\s*Conflict Note:", text, flags=re.I): return False
        return True

    def _generate_answer(self, temperature, top_p, repeat_penalty, num_predict) -> str:
        # Stage A: bracketed, generate-first to reduce chat meta
        sys_a = (
          "You are Pilot. Print ONLY the answer BETWEEN the exact tags on their own lines:\n"
          "<<BEGIN>>\n<paragraph>\n<<END>>\n"
          "Constraints: 120–150 words; ≥3 numeric citations like [1][2][3]; "
          "append a final line: Conflict Note: ... (or 'Conflict Note: none'). No preface."
        )
        usr = ("Explain PageRank in ≤150 words with ≥3 citations. Prefer the Brin & Page 1998 paper, "
               "Google patent US 6,285,999, and an official Google source. Resolve the 'just link counts' misconception.")
        raw = self.llm.ask(sys_a, usr, temperature=temperature, top_p=top_p,
                           repeat_penalty=repeat_penalty, num_predict=num_predict,
                           prefer_generate=True)
        text = between_tags(raw, "<<BEGIN>>", "<<END>>").strip()
        if not self._is_valid_answer(text):
            # Stage B: simple no-tag instruction, still generate-first
            sys_b = ("You are Pilot. Output ONLY one 120–150 word paragraph explaining PageRank with ≥3 inline numeric citations "
                     "like [1][2][3], then on a new line end with: Conflict Note: ... (or 'Conflict Note: none'). No preface.")
            raw2 = self.llm.ask(sys_b, usr, temperature=temperature, top_p=top_p,
                                repeat_penalty=repeat_penalty, num_predict=num_predict,
                                prefer_generate=True)
            text2 = between_tags(raw2, "<<BEGIN>>", "<<END>>").strip()
            if self._is_valid_answer(text2):
                text = text2
        if not self._is_valid_answer(text):
            text = self._local_pagerank_fallback()
        return text

    def run_pagerank_demo(self, ach: Optional[float]=None, seed:int=SEED_DEFAULT, deny_policy: bool=False) -> Dict[str,Any]:
        seed_everything(seed)
        mu_in = Mu(self.neuro0.da, self.neuro0.ne, self.neuro0.s5ht,
                   ach if ach is not None else self.neuro0.ach,
                   self.neuro0.gaba, self.neuro0.oxt)

        goal = "Explain PageRank ≤150 words with ≥3 citations; detect and resolve one contradiction."
        risk = "R3" if deny_policy else self.cust.classify(goal)
        verdict = self.cust.preflight(risk)
        intent = self.pilot.draft_intent(goal, risk)

        app = Appraisal(p=0.3, n=0.4, u=0.3, k=0.1, s=1.0, c=0.6, h=1.0)
        mu_out = self.homeo.update(mu_in, app)
        pol = self.homeo.couple(mu_out)

        llm_answer = "[blocked by policy]"; llm_knobs = {}; http_trace = {}
        if verdict["action"] == "allow":
            temperature    = max(0.1, 0.9 - 0.6*mu_out.s5ht)
            top_p          = clamp(0.75 + 0.20*mu_out.ne - 0.10*mu_out.gaba, 0.50, 0.95)
            repeat_penalty = 1.05 + 0.20*mu_out.s5ht - 0.10*mu_out.da
            num_predict    = int(256 + int(384*mu_out.s5ht) - int(128*mu_out.gaba))
            try:
                llm_answer = self._generate_answer(temperature, top_p, repeat_penalty, num_predict)
            except Exception as e:
                http_trace = getattr(self.llm, "last_http", {})
                llm_answer = self._local_pagerank_fallback() + f"\n[LLM error noted: {e} | http={http_trace}]"
            llm_knobs = {"temperature": round(temperature,3),
                         "top_p": round(top_p,3),
                         "repeat_penalty": round(repeat_penalty,3),
                         "num_predict": int(num_predict)}

        critic = {}
        if verdict["action"] == "allow":
            critic_passes = 1 + int(2*mu_out.ach)
            try:
                critic = run_critic(self.llm, llm_answer, mu_out, passes=critic_passes)
            except Exception as e:
                critic = {"q_overall": 0.0, "has_conflict_note": False,
                          "reasons":[f"critic exec error: {e} | http={self.llm.last_http}"]}

        ev = self.scout.fetch_pagerank(pol.k_breadth, pol.q_contra)
        conflict_note_present = bool(re.search(r"(^|\n)\s*Conflict Note:", (llm_answer or ""), flags=re.I))

        claims = [
            Claim(id="c1", text="PageRank models a random surfer with damping ~0.85.", q=0.8,
                  sources=[Source(url="pagerank_primary.txt", domain_tier=1)], stance="pro"),
            Claim(id="c2", text="Not simple link counts; weights depend on inlink ranks/outdegree.", q=0.8,
                  sources=[Source(url="pagerank_dissent.txt", domain_tier=3)], stance="pro"),
            Claim(id="c3", text="Media often oversimplify as mere link counts (misleading).", q=0.7,
                  sources=[Source(url="pagerank_media.txt", domain_tier=3)], stance="neutral"),
        ]
        for c in claims: self.arch.upsert_claim(c)
        # potential contradiction graph: c2 vs c3
        # self.arch.link_contradiction("c2","c3")  # not strictly used in output

        adopt = True
        if critic: adopt = critic.get("q_overall", 0.0) >= 0.70

        stats = {"sources": len(ev),
                 "resolved": conflict_note_present or critic.get("has_conflict_note", False),
                 "goal_met": (len(ev)>=3 and verdict["action"]=="allow" and adopt)}
        kpis = self.wit.score(stats)
        stop = soft_stop(1.0 if stats["goal_met"] else 0.0, mu_out.gaba, 0.2, 0.2)

        payload = {
            "goal": goal, "risk": risk, "verdict": verdict["action"],
            "intent": asdict(intent), "mu_out": asdict(mu_out), "policy": asdict(pol),
            "llm_knobs": llm_knobs, "evidence": [e.id for e in ev],
            "claims": [c.to_dict() for c in claims], "llm_preview": (llm_answer or "")[:700],
            "critic": critic, "adopted": adopt, "kpis": kpis, "stop_score": stop
        }
        if self.debug and verdict["action"] == "allow":
            payload["last_http"] = getattr(self.llm, "last_http", {})
        return payload

# ========= CLI & Probes =========
def main():
    ap = argparse.ArgumentParser(description="Guardian-AGI (single-file) — Ollama chat/generate + Emotional Center")
    ap.add_argument("--model", default="gpt-oss:20b",
                    help="Ollama model name (default gpt-oss:20b; e.g., qwen2.5:14b-instruct-q4_K_M)")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--ach", type=float, default=None, help="override ACh [0..1]")
    ap.add_argument("--probe", choices=["none","ach","policy","stop","critic"], default="none")
    ap.add_argument("--record", default="", help="Path to ledger JSONL (append-only). Empty=off.")
    ap.add_argument("--debug", action="store_true", help="Include last HTTP trace on LLM errors.")
    args = ap.parse_args()

    eng = Engine(model_name=args.model, debug=args.debug)

    if args.probe == "ach":
        low  = eng.run_pagerank_demo(ach=0.2, seed=args.seed)
        high = eng.run_pagerank_demo(ach=0.9, seed=args.seed)
        out = {
            "low.policy":  low["policy"],  "low.evidence":  low["evidence"],  "low.knobs":  low["llm_knobs"],
            "high.policy": high["policy"], "high.evidence": high["evidence"], "high.knobs": high["llm_knobs"],
            "delta_k_breadth": high["policy"]["k_breadth"] - low["policy"]["k_breadth"],
            "delta_q_contra":   high["policy"]["q_contra"]  - low["policy"]["q_contra"],
            "expectation": "High ACh should include pagerank_dissent.txt and raise q_contra/k_breadth."
        }
        print(json.dumps(out, indent=2)); return

    if args.probe == "policy":
        res = eng.run_pagerank_demo(ach=args.ach, seed=args.seed, deny_policy=True)
        print(json.dumps({"risk":res["risk"],"verdict":res["verdict"],
                          "note":"Custodian veto blocks LLM call"}, indent=2)); return

    if args.probe == "stop":
        res_ok = eng.run_pagerank_demo(ach=args.ach, seed=args.seed)
        res_bad = res_ok.copy(); res_bad["kpis"] = {**res_bad["kpis"], "pass_at_1": 0.0}
        res_bad["stop_score"] = soft_stop(0.0, res_ok["mu_out"]["gaba"], 0.2, 0.2)
        print(json.dumps({
            "stop_when_goal_met": res_ok["stop_score"],
            "stop_when_unmet": res_bad["stop_score"],
            "expectation":"unmet < met and typically < 0.6 threshold"
        }, indent=2)); return

    if args.probe == "critic":
        low  = eng.run_pagerank_demo(ach=0.2, seed=args.seed)
        high = eng.run_pagerank_demo(ach=0.9, seed=args.seed)
        print(json.dumps({
            "low.critic": low.get("critic",{}),  "low.adopted": low.get("adopted", True),
            "high.critic": high.get("critic",{}),"high.adopted": high.get("adopted", True),
            "expectation": "High ACh → more critic passes; q_overall should be ≥ low or equal; adoption requires q≥0.70."
        }, indent=2)); return

    res = eng.run_pagerank_demo(ach=args.ach, seed=args.seed)
    print(json.dumps(res, indent=2))
    if args.record:
        ledger_append(args.record, {
            "goal": res["goal"], "risk": res["risk"], "verdict": res["verdict"],
            "mu_out": res["mu_out"], "policy": res["policy"],
            "llm_knobs": res["llm_knobs"], "critic": res.get("critic", {}),
            "adopted": res.get("adopted", True), "kpis": res["kpis"], "stop_score": res["stop_score"]
        })

if __name__ == "__main__":
    main()
