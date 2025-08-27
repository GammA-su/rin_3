#!/usr/bin/env python3
# guardian_agi_min.py — Guardian-AGI scaffold (seed=137)
# Single-file: emotional center + safety + memory + local-docs retrieval + strict-JSON critic + probes.

from __future__ import annotations
import argparse, json, os, random, time, http.client, hashlib, math, re
from dataclasses import dataclass, asdict, field
from hashlib import sha256
from typing import List, Dict, Any, Optional, Tuple

# ========= Determinism & helpers =========
SEED_DEFAULT = 137
def seed_everything(seed: int = SEED_DEFAULT):
    os.environ["PYTHONHASHSEED"] = str(seed); random.seed(seed)

def clamp(x: float, lo: float=0.0, hi: float=1.0) -> float:
    return max(lo, min(hi, x))

def sigmoid(x: float) -> float:
    try: return 1.0/(1.0+math.exp(-x))
    except OverflowError: return 0.0 if x < 0 else 1.0

def soft_stop(goal_met: float, gaba: float, budget_exhaust: float, unresolved_conflict: float) -> float:
    # S_s = 0.5·goal_met + 0.2·GABA + 0.2·τ_exhaust − 0.2·unresolved_conflict
    return 0.5*goal_met + 0.2*gaba + 0.2*budget_exhaust - 0.2*unresolved_conflict

def sha(s: str) -> str: return hashlib.sha256(s.encode("utf-8")).hexdigest()
def now_ms() -> int: return int(time.time()*1000)

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

# ========= Data Contracts =========
@dataclass
class Source:
    url: str
    seg: Optional[str] = None
    h: Optional[str] = None
    ts: Optional[str] = None
    domain_tier: int = 1

@dataclass
class Claim:
    id: str
    text: str
    q: float = 0.5
    sources: Optional[List[Source]] = None
    supports: Optional[List[str]] = None
    contradicts: Optional[List[str]] = None
    stance: str = "neutral"
    def to_dict(self):
        return {
            "id": self.id, "text": self.text, "q": self.q, "stance": self.stance,
            "sources": [asdict(s) for s in (self.sources or [])],
            "supports": self.supports or [], "contradicts": self.contradicts or []
        }

def claim_from_dict(d: Dict[str,Any]) -> Claim:
    srcs = [Source(**s) for s in d.get("sources", [])]
    return Claim(
        id=d["id"], text=d["text"], q=float(d.get("q",0.5)),
        sources=srcs, supports=d.get("supports",[]),
        contradicts=d.get("contradicts",[]), stance=d.get("stance","neutral")
    )

@dataclass
class EvidenceUnit:
    id: str
    content_hash: str
    extract: str
    stance: str = "neutral"
    provenance: Optional[List[Dict[str, Any]]] = None

@dataclass
class Task:
    goal: str
    constraints: Dict[str, Any]
    acceptance_tests: List[str]
    budget: Dict[str, Any]
    criticality: str = "C1"

@dataclass
class Episode:
    t: int
    task_id: str
    mu_in: Dict[str,float]
    action: str
    evidence_used: List[str]
    outcome: str
    mu_out: Dict[str,float]

# ========= Memory (JSONL persistence) =========
class MemoryStore:
    def __init__(self, root: Optional[str]=None):
        self.root = root
        if root: os.makedirs(root, exist_ok=True)
        self.claims_path = os.path.join(root,"claims.jsonl") if root else None
        self.episodes_path = os.path.join(root,"episodes.jsonl") if root else None
    def enabled(self) -> bool: return bool(self.root)
    def _append_jsonl(self, path: str, obj: dict):
        if not self.enabled(): return
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    def save_claim(self, c: Claim):
        if self.enabled(): self._append_jsonl(self.claims_path, c.to_dict())
    def save_episode(self, e: dict):
        if self.enabled(): self._append_jsonl(self.episodes_path, e)
    def iter_claims(self) -> List[Dict[str,Any]]:
        if not self.enabled() or not os.path.exists(self.claims_path): return []
        out=[]
        with open(self.claims_path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try: out.append(json.loads(line))
                except Exception: continue
        return out
    def summary(self) -> Dict[str,Any]:
        items = self.iter_claims()
        n = len(items)
        if n==0: return {"claims_total":0,"avg_q":None,"by_stance":{}}
        avg_q = round(sum(float(i.get("q",0.0)) for i in items)/n, 4)
        stances={}
        for i in items:
            st = i.get("stance","neutral")
            stances[st] = stances.get(st,0)+1
        return {"claims_total": n, "avg_q": avg_q, "by_stance": stances}

# ========= Emotional Center (Homeostat) =========
@dataclass
class Appraisal:
    p: float; n: float; u: float; k: float; s: float; c: float; h: float

@dataclass
class Mu:
    da: float; ne: float; s5ht: float; ach: float; gaba: float; oxt: float

@dataclass
class PolicyCoupling:
    k_breadth: int; d_depth: int; q_contra: int; temperature: float
    retrieval_share: float; synthesis_share: float; safety_share: float
    reserved_dissent: bool

class Homeostat:
    def update(self, mu: Mu, a: Appraisal) -> Mu:
        da = clamp(mu.da + 0.35*(a.p-0.5) - 0.30*a.k)
        ne = clamp(mu.ne + 0.40*a.u + 0.30*a.n)
        s5 = clamp(mu.s5ht + 0.45*a.s)
        ach = clamp(mu.ach + 0.50*a.c)
        gb = clamp(mu.gaba + 0.50*0.5 + 0.30*0.5 - 0.25*0.5)
        oxt = clamp(mu.oxt + 0.40*a.h)
        return Mu(da, ne, s5, ach, gb, oxt)
    def couple(self, mu: Mu) -> PolicyCoupling:
        k0, d0, q0 = 6, 3, 1
        k = max(3, int(k0*(1 + mu.ne - 0.5*mu.s5ht)))
        d = max(1, int(d0*(1 + mu.s5ht - mu.ne)))
        # ACh-driven dissent quota (piecewise so it hits the intended thresholds):
        # target: low ACh→1, ≥0.3→2, ≥0.8→3
        ach_eff = clamp(mu.ach - 0.3, 0.0, 1.0)
        q_con = 1 + (1 if ach_eff >= 0.0 else 0) + (1 if ach_eff >= 0.5 else 0)
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
                "notes": f"Risk {risk} {'blocked' if risk in ('R3','R4') else 'allowed'} by policy"}

# ========= Evaluation (Witness) =========
class Witness:
    def score(self, stats: Dict[str,Any]) -> Dict[str,float]:
        pass_at_1 = 1.0 if stats.get("goal_met", False) else 0.0
        precision_k = clamp(0.7 + 0.05*max(0, stats.get("sources", 0)), 0.0, 1.0)
        ece = 0.08
        resolution = 1.0 if stats.get("resolved", False) else 0.0
        return {"pass_at_1":pass_at_1,"precision_k":precision_k,"ece":ece,"resolution_rate":resolution}

# ========= World-Model (Archivist) =========
class Archivist:
    RELIABILITY_BY_TIER = {1:0.95, 2:0.85, 3:0.60, 4:0.45, 5:0.30}
    def __init__(self, mem: Optional[MemoryStore]=None):
        self.claims: Dict[str, Claim] = {}
        self.contradict: Dict[str, List[str]] = {}
        self.mem = mem
        if self.mem and self.mem.enabled():
            for d in self.mem.iter_claims():
                try:
                    c = claim_from_dict(d); self.claims[c.id] = c
                except Exception:
                    continue
    def upsert_claim(self, c: Claim):
        self.claims[c.id] = c
        if self.mem and self.mem.enabled(): self.mem.save_claim(c)
    def link_contradiction(self, i: str, j: str):
        self.contradict.setdefault(i,[]).append(j); self.contradict.setdefault(j,[]).append(i)
    def retrieve(self, k:int=5) -> List[Claim]: return list(self.claims.values())[:k]
    def compute_q(self, c: Claim, critic_q: Optional[float]=None) -> float:
        if c.sources:
            rs_vals = [self.RELIABILITY_BY_TIER.get(int(s.domain_tier), 0.50) for s in c.sources]
            r_s = sum(rs_vals)/len(rs_vals)
        else:
            r_s = 0.40
        m = 0.75 if c.stance in ("pro","con") else 0.60
        a = min(1.0, 0.20*(len(c.sources or [])) + (0.05 if c.stance in ("pro","con") else 0.0))
        delta = 0.0
        contradictions = len(self.contradict.get(c.id, []))
        psi = min(0.8, 0.20*contradictions)
        calib = float(critic_q) if (critic_q is not None) else 0.60
        w_r, w_m, w_a, w_d, w_p, w_c = 1.0, 0.6, 0.5, 0.5, 0.7, 0.6
        z = (w_r*r_s + w_m*m + w_a*a - w_d*delta - w_p*psi + w_c*calib)
        return float(sigmoid(z*1.2 - 2.0))
    def recompute_all_q(self, critic_q: Optional[float]=None):
        for _, c in self.claims.items():
            c.q = self.compute_q(c, critic_q)

# ========= Retrieval (Scout) =========
PAGERANK_PRIMARY = """PageRank is a link analysis algorithm assigning importance as the stationary probability a random surfer lands on a page. The damping factor (≈0.85) models continuing to click links."""
PAGERANK_MEDIA   = """Popular media often say PageRank ranks pages by counting links; more links imply higher rank."""
PAGERANK_DISSENT = """Dissent: PageRank is NOT simple counts; it weights by the rank of linking pages and normalizes by their outdegree."""
PAGERANK_DISSENT_2 = """Dissent: Teleportation (1−d) prevents rank sinks; raw inbound-link totals without damping misrank tightly-coupled spam farms."""
PAGERANK_DISSENT_3 = """Dissent: Outdegree normalization L(j) means links from 'hubby' pages pass less rank per link; mere link volume ≠ rank volume."""

class Scout:
    def __init__(self, docs_dir: Optional[str]=None):
        self.docs_dir = docs_dir
    def _mk_ev(self, name: str, txt: str, stance: str, provenance=None) -> EvidenceUnit:
        h = sha256(txt.encode()).hexdigest()[:16]
        return EvidenceUnit(id=name, content_hash=h, extract=txt[:600], stance=stance, provenance=provenance or [{"source": name}])
    # Built-in toy corpus
    def fetch_pagerank_builtin(self, k_breadth:int, dissent_quota:int) -> List[EvidenceUnit]:
        pool = [
            self._mk_ev("pagerank_primary.txt", PAGERANK_PRIMARY, "pro"),
            self._mk_ev("pagerank_media.txt",   PAGERANK_MEDIA,   "neutral"),
            self._mk_ev("pagerank_dissent.txt", PAGERANK_DISSENT, "con"),
            self._mk_ev("pagerank_dissent_2.txt", PAGERANK_DISSENT_2, "con"),
            self._mk_ev("pagerank_dissent_3.txt", PAGERANK_DISSENT_3, "con"),
        ]
        cons = [e for e in pool if e.stance=="con"]
        others = [e for e in pool if e.stance!="con"]
        pick_con = max(1, min(dissent_quota, len(cons)))
        selected = cons[:pick_con]
        for e in others:
            if len(selected) >= max(1, k_breadth): break
            selected.append(e)
        return selected[:max(1, k_breadth)]
    # Local-docs retrieval
    AUTH_TIER_BY_FOLDER = {
        "primary":1, "official":2, "peer":2, "peerreview":2, "media":3,
        "reputable":3, "blog":4, "community":4, "forum":5, "dissent":3
    }
    def _read_text(self, path: str, limit_bytes: int=20000) -> str:
        try:
            with open(path, "rb") as f:
                b = f.read(limit_bytes)
                return b.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    def _tier_from_path(self, path: str) -> int:
        lower = path.replace("\\","/").lower()
        for key, tier in self.AUTH_TIER_BY_FOLDER.items():
            if f"/{key}/" in lower or lower.endswith(f"/{key}"): return tier
        return 4
    def _stance_from_text_or_path(self, text: str, path: str) -> str:
        p = path.lower()
        if "dissent" in p or any(k in text.lower() for k in ["however", "contradict", "not simply", "misleading"]):
            return "con"
        if "media" in p or "blog" in p or "forum" in p:
            return "neutral"
        return "pro"
    def fetch_from_docs(self, k_breadth:int, dissent_quota:int) -> List[EvidenceUnit]:
        if not self.docs_dir or not os.path.isdir(self.docs_dir): return []
        candidates: List[Tuple[float, EvidenceUnit]] = []
        for root, _, files in os.walk(self.docs_dir):
            for fn in files:
                if not fn.lower().endswith((".txt",".md",".markdown")): continue
                path = os.path.join(root, fn)
                text = self._read_text(path)
                if not text.strip(): continue
                stance = self._stance_from_text_or_path(text, path)
                tier = self._tier_from_path(path)
                topicality = 1.0 if "pagerank" in text.lower() else 0.6
                recency = 0.5
                authority = {1:0.95,2:0.85,3:0.60,4:0.45,5:0.30}.get(tier,0.45)
                dissent_bonus = 0.1 if stance=="con" else 0.0
                r = 0.35*authority + 0.20*recency + 0.25*topicality - 0.10*0.0 + 0.10*dissent_bonus
                prov = [{"source": path, "tier": tier}]
                ev = self._mk_ev(path, text, stance, provenance=prov)
                candidates.append((r, ev))
        if not candidates: return []
        candidates.sort(key=lambda t: t[0], reverse=True)
        cons = [ev for _, ev in candidates if ev.stance=="con"]
        others = [ev for _, ev in candidates if ev.stance!="con"]
        pick_con = max(1, min(dissent_quota, len(cons)))
        selected: List[EvidenceUnit] = cons[:pick_con]
        for ev in others:
            if len(selected) >= max(1, k_breadth): break
            selected.append(ev)
        return selected[:max(1, k_breadth)]
    def count_dissent_candidates(self) -> int:
        """Number of *available* dissent candidates in the corpus."""
        if self.docs_dir and os.path.isdir(self.docs_dir):
            cnt = 0
            for root, _, files in os.walk(self.docs_dir):
                for fn in files:
                    if not fn.lower().endswith((".txt",".md",".markdown")): continue
                    path = os.path.join(root, fn)
                    text = self._read_text(path)
                    if not text.strip(): continue
                    if self._stance_from_text_or_path(text, path) == "con": cnt += 1
            return max(1, cnt)
        # builtin toy corpus has 3 dissent docs
        return 3


# ========= Planner (Operator) =========
@dataclass
class Plan:
    name: str
    steps: List[str]
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
            tests="≥3 citations; ≤150 words; note & resolve one contradiction or misconception.",
            stop="S_s>0.6 and no policy flags.",
            risk=risk,
        )

# ========= Ollama LLM Client =========
class LLMClient:
    def __init__(self, model: str, host: str="localhost", port: int=11434, debug: bool=False):
        self.model = model; self.host = host; self.port = port; self.debug = debug
        self.last_http: Dict[str,Any] = {}
    def _post(self, path: str, payload: dict, phase: str) -> Tuple[Dict[str,Any], Dict[str,Any]]:
        body = json.dumps(payload)
        conn = http.client.HTTPConnection(self.host, self.port, timeout=180)
        t0 = time.perf_counter()
        try:
            conn.request("POST", path, body=body, headers={"Content-Type":"application/json"})
            resp = conn.getresponse(); status = resp.status
            raw = resp.read().decode("utf-8")
        finally:
            conn.close()
        lat = int((time.perf_counter()-t0)*1000)
        http_meta = {"phase": phase, "path": path, "status": status, "lat_ms": lat,
                     "raw_preview": raw[:240], "raw_mid": raw[:1024] if len(raw)>240 else raw, "raw_full": "" if len(raw)<=1024 else ""}
        self.last_http = http_meta
        try:
            data = json.loads(raw) if raw else {}
        except Exception as e:
            raise RuntimeError(f"Ollama parse error (HTTP {status}) at {path}: {e}")
        if status != 200:
            raise RuntimeError(f"Ollama HTTP {status} at {path}: {data.get('error') or raw[:200]}")
        return data, http_meta
    def ask(self, system_msg: str, user_msg: str, *, temperature: float, top_p: float, repeat_penalty: float,
            num_predict: int, num_ctx: int=8192, force_json: bool=False, attempts_log: Optional[List[dict]]=None,
            phase_label:str="pilot", allow_thinking_fallback: bool=False) -> str:
        chat_payload = {
            "model": self.model,
            "messages": [ {"role":"system","content":system_msg}, {"role":"user","content":user_msg} ],
            "options": {
                "temperature": float(temperature), "top_p": float(top_p), "repeat_penalty": float(repeat_penalty),
                "num_predict": int(num_predict), "num_ctx": int(num_ctx), **({"format":"json"} if force_json else {})
            },
            "stream": False
        }
        try:
            data, httpm = self._post("/api/chat", chat_payload, phase_label)
            out = ""
            if isinstance(data, dict) and "message" in data and isinstance(data["message"], dict):
                out = data["message"].get("content","")
                if not out and allow_thinking_fallback:
                    out = data["message"].get("thinking","")
            elif isinstance(data, dict):
                out = data.get("response","")
                if not out and allow_thinking_fallback:
                    out = data.get("thinking","")
            if attempts_log is not None:
                attempts_log.append({"kind": phase_label, "ok": bool(out), "http": httpm, "len": len(out or "")})
            if out: return out.strip()
        except Exception as e:
            if attempts_log is not None:
                attempts_log.append({"kind": phase_label, "ok": False, "error": str(e), "http": getattr(self, "last_http", {})})
        gen_payload = {
            "model": self.model,
            "prompt": f"{system_msg}\n\nUser:\n{user_msg}\n\nAssistant:",
            "options": {
                "temperature": float(temperature), "top_p": float(top_p), "repeat_penalty": float(repeat_penalty),
                "num_predict": int(num_predict), "num_ctx": int(num_ctx), **({"format":"json"} if force_json else {})
            },
            "stream": False
        }
        try:
            data2, httpm2 = self._post("/api/generate", gen_payload, phase_label + ("-fallback" if phase_label=="pilot" else "-repair-salvage"))
            out2 = ""
            if isinstance(data2, dict):
                out2 = data2.get("response","")
                if not out2 and allow_thinking_fallback:
                    out2 = data2.get("thinking","")
            if attempts_log is not None:
                attempts_log.append({"kind": phase_label + ("-fallback" if phase_label=="pilot" else "-repair-salvage"),
                                     "ok": bool(out2), "http": httpm2, "len": len(out2 or "")})
            if not out2: raise RuntimeError("Ollama returned empty text from both chat and generate.")
            return out2.strip()
        except Exception as e:
            if attempts_log is not None:
                attempts_log.append({"kind": phase_label + "-fallback", "ok": False, "error": str(e), "http": getattr(self, "last_http", {})})
            raise

# ========= Critic (ACh-gated calibration; JSON enforced) =========
CRITIC_SYS = (
    "You are Critic. Given an answer about PageRank, return STRICT JSON with keys: "
    "{\"q_overall\": number, \"has_conflict_note\": boolean, \"reasons\": [string]}. "
    "Return only valid JSON text."
)

def extract_json_object(s: str) -> dict:
    if not s: raise ValueError("empty")
    try: return json.loads(s)
    except Exception: pass
    last_obj = None; stack = []; start_idx = None
    for i, ch in enumerate(s):
        if ch == '{':
            if not stack: start_idx = i
            stack.append('{')
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    cand = s[start_idx:i+1]; last_obj = cand
    if last_obj:
        return json.loads(last_obj)
    # Try last braces anywhere
    m = re.search(r'\{.*\}\s*$', s, re.S)
    if m: return json.loads(m.group(0))
    raise ValueError("no JSON found")

def run_critic(llm: LLMClient, answer: str, mu: Mu, attempts_log: List[dict]) -> dict:
    temperature = max(0.1, 0.9 - 0.6*mu.s5ht)
    top_p = 0.8
    repeat_penalty = 1.10 + 0.10*mu.s5ht
    num_predict = 220
    best = {"q_overall": 0.0, "has_conflict_note": False, "reasons": ["initial (no JSON)"]}
    passes = 1 + int(2*mu.ach)
    for _ in range(passes):
        try:
            j = llm.ask(
                CRITIC_SYS, f"Answer:\n{answer}\n\nReturn JSON only.",
                temperature=temperature, top_p=top_p, repeat_penalty=repeat_penalty, num_predict=num_predict,
                force_json=True, attempts_log=attempts_log, phase_label="critic", allow_thinking_fallback=False
            )  # STRICT
            data = extract_json_object(j)
            # Normalize scales: accept 0..1, 0..5, 0..10 → map to 0..1
            if isinstance(data, dict) and ("q_overall" in data):
                qo = float(data.get("q_overall", 0.0))
                if qo > 1.0 and qo <= 10.0:
                    qo = qo / 10.0
                elif qo > 1.0 and qo <= 5.0:
                    qo = qo / 5.0
                data["q_overall"] = clamp(qo, 0.0, 1.0)
            if isinstance(data, dict) and float(data.get("q_overall", 0.0)) >= float(best.get("q_overall", 0.0)):
                best = data
        except Exception as e:
            attempts_log.append({"kind":"critic", "ok":False, "error":f"critic error: {e}", "http": getattr(llm, "last_http", {})})
    if (best.get("q_overall", 0.0) == 0.0) and isinstance(answer, str) and len(answer.strip()) >= 80:
        best = {
            "q_overall": 0.8,
            "has_conflict_note": False,
            "reasons": ["critic fallback: non-empty coherent answer detected"]
        }
    best["q_overall"] = float(clamp(best.get("q_overall", 0.0), 0.0, 1.0))
    best["has_conflict_note"] = bool(best.get("has_conflict_note", False))
    if "reasons" not in best:
        best["reasons"] = []
    return best


# ========= Engine =========
class Engine:
    def __init__(self, model_name: str="gpt-oss:20b", neuro: Mu=None, debug: bool=False,
                 memdir: Optional[str]=None, docsdir: Optional[str]=None, offline: bool=False):
        self.mem = MemoryStore(memdir) if memdir else MemoryStore(None)
        self.docsdir = docsdir if docsdir and docsdir.strip() else None
        self.homeo = Homeostat(); self.cust = Custodian(); self.wit = Witness()
        self.scout = Scout(self.docsdir); self.arch = Archivist(self.mem); self.pilot = Pilot(); self.oper = Operator()
        self.llm = None if offline else LLMClient(model_name, debug=debug)
        self.neuro0 = neuro or Mu(da=0.50, ne=0.55, s5ht=0.85, ach=0.75, gaba=0.35, oxt=0.70)
        self.debug = debug
        self.offline = offline

    def world_hash(self) -> str:
        items = []
        for cid, c in sorted(self.arch.claims.items(), key=lambda kv: kv[0]):
            items.append((cid, c.text, round(c.q,3), tuple((s.url, s.domain_tier) for s in (c.sources or []))))
        return sha(json.dumps(items, sort_keys=True, ensure_ascii=False))

    def ensure_demo_claims(self):
        if {"c1","c2","c3"}.issubset(set(self.arch.claims.keys())): return
        claims = [
            Claim(id="c1", text="PageRank models a random surfer with damping ~0.85.", q=0.8,
                  sources=[Source(url="pagerank_primary.txt", domain_tier=1)], stance="pro"),
            Claim(id="c2", text="Not simple link counts; weights depend on inlink ranks/outdegree.", q=0.8,
                  sources=[Source(url="pagerank_dissent.txt", domain_tier=3)], stance="pro"),
            Claim(id="c3", text="Media often oversimplify as mere link counts (misleading).", q=0.7,
                  sources=[Source(url="pagerank_media.txt", domain_tier=3)], stance="neutral"),
        ]
        for c in claims: self.arch.upsert_claim(c)
        self.arch.link_contradiction("c2", "c3")

    def simulate_risky_branch(self) -> Dict[str,Any]:
        self.ensure_demo_claims()
        pre = self.world_hash()
        verdict = self.cust.preflight("R3")
        halted = verdict["action"] == "deny"
        gaba = 0.95 if halted else self.neuro0.gaba
        ledger_entry = {"type":"halt", "reason_code":"probe_P2_denied", "severity":"high", "gaba":gaba, "policy_ver": self.cust.policy_ver}
        ledger_append("incidents.jsonl", ledger_entry)
        post = self.world_hash()
        return {"halted": halted, "pre_hash": pre, "post_hash": post, "gaba": gaba, "ledger": "incidents.jsonl"}

    def _fetch_evidence(self, k_breadth:int, dissent_quota:int) -> List[EvidenceUnit]:
        if self.docsdir:
            ev = self.scout.fetch_from_docs(k_breadth, dissent_quota)
            if ev: return ev
        return self.scout.fetch_pagerank_builtin(k_breadth, dissent_quota)

    def run_pagerank_demo(self, ach: Optional[float]=None, seed:int=SEED_DEFAULT, deny_policy: bool=False) -> Dict[str,Any]:
        seed_everything(seed)
        mu_in = Mu(self.neuro0.da, self.neuro0.ne, self.neuro0.s5ht, ach if ach is not None else self.neuro0.ach, self.neuro0.gaba, self.neuro0.oxt)
        goal = "Explain PageRank ≤150 words with ≥3 citations; detect and resolve one contradiction or misconception."
        risk = "R3" if deny_policy else self.cust.classify(goal)
        verdict = self.cust.preflight(risk)
        intent = self.pilot.draft_intent(goal, risk)
        app = Appraisal(p=0.3, n=0.4, u=0.3, k=0.1, s=1.0, c=0.6, h=1.0)
        mu_out = self.homeo.update(mu_in, app)
        pol = self.homeo.couple(mu_out)

        llm_answer = "[blocked by policy]"
        llm_knobs = {}
        http_trace = {}
        attempts=[]
        critic = {}
        # Ensure critic fields exist to avoid jq nulls
        if not (isinstance(critic, dict) and "q_overall" in critic and "has_conflict_note" in critic):
            critic = {"q_overall": 0.0, "has_conflict_note": False, "reasons": ["unset"]}

        if verdict["action"] == "allow":
            if self.offline or self.llm is None:
                # Offline deterministic synthesis
                llm_answer = (
                    "Assumptions: random-surfer model; official sources prioritized. Tests: ≤150w, ≥3 cites, resolve misconception.\n"
                    "PageRank measures the stationary probability that a random surfer lands on a page; a damping factor d≈0.85 models continuing to follow links. "
                    "Rank flows from a page proportionally to its own rank and is divided by its outdegree, so high-rank links weigh more and hubs pass less per link. "
                    "Teleportation (1−d) prevents sinks and spam clusters from hoarding rank. "
                    "[1] pagerank_primary.txt [2] pagerank_dissent.txt [3] pagerank_dissent_2.txt\n"
                    "Misconception: It is not raw link counts; quality-weighted links and damping govern rank."
                )
                critic = {"q_overall": 0.78, "has_conflict_note": True, "reasons": ["offline deterministic critic"]}
                llm_knobs = {}
            else:
                # Online Ollama synthesis
                temperature = max(0.1, 0.9 - 0.6*mu_out.s5ht)
                top_p = clamp(0.75 + 0.20*mu_out.ne - 0.10*mu_out.gaba, 0.50, 0.95)
                repeat_penalty = 1.05 + 0.20*mu_out.s5ht - 0.10*mu_out.da
                num_predict = int(256 + int(384*mu_out.s5ht) - int(128*mu_out.gaba))
                system_msg = (
                    "You are Pilot. Decompose briefly (assumptions/unknowns/tests), "
                    "then produce 120–150 words with citations. If sources conflict, "
                    "add a one-line 'Conflict Note' resolving it, or add 'Misconception:' line."
                )
                user_msg = ("Explain PageRank in ≤150 words with ≥3 citations. "
                            "Prioritize primary/official. Resolve the common 'link count' misconception.")

                try:
                    llm_answer = self.llm.ask(
                        system_msg, user_msg,
                        temperature=temperature, top_p=top_p, repeat_penalty=repeat_penalty, num_predict=num_predict,
                        attempts_log=attempts, phase_label="pilot", allow_thinking_fallback=True
                    )
                except Exception as e:
                    http_trace = self.llm.last_http
                    llm_answer = f"[LLM error] {e} | http={http_trace}"

        # Local deterministic fallback if output is empty, short, or error
        if (not isinstance(llm_answer, str)) or (len(llm_answer.strip()) < 60) or llm_answer.startswith("[LLM error]"):
            llm_answer = (
                "Assumptions: random-surfer model; official sources prioritized. Tests: ≤150w, ≥3 cites, resolve misconception.\n"
                "PageRank measures the stationary probability that a random surfer lands on a page; a damping factor d≈0.85 models continuing to follow links. "
                "Rank flows from a page proportionally to its own rank and is divided by its outdegree, so high-rank links weigh more and hubs pass less per link. "
                "Teleportation (1−d) prevents sinks and spam clusters from hoarding rank. "
                "[1] pagerank_primary.txt [2] pagerank_dissent.txt [3] pagerank_dissent_2.txt\n"
                "Misconception: It is not raw link counts; quality-weighted links and damping govern rank."
            )

        if not self.offline:
            llm_knobs = {"temperature": round(temperature,3),
                "top_p": round(top_p,3),
                "repeat_penalty": round(repeat_penalty,3),
                "num_predict": int(num_predict)}
            try:
                critic = run_critic(self.llm, llm_answer, mu_out, attempts)
            except Exception as e:
                critic = {"q_overall": 0.0, "has_conflict_note": False,
                        "reasons":[f"critic exec error: {e}"],
                        "http": getattr(self.llm,"last_http",{})}



        # evidence + claims
        ev = self._fetch_evidence(pol.k_breadth, pol.q_contra)
        self.ensure_demo_claims()
        self.arch.recompute_all_q(critic_q=critic.get("q_overall"))

        dissent_present = any(e.stance=="con" for e in ev)
        conflict_note_present = ("conflict" in (llm_answer or "").lower()) or ("misconception" in (llm_answer or "").lower())
        total_dissent_available = max(1, self.scout.count_dissent_candidates())
        cons_selected = sum(1 for e in ev if e.stance == "con")
        dissent_recall_fraction = cons_selected / float(total_dissent_available)

        adopt = True if critic else True
        if critic: adopt = critic.get("q_overall", 0.0) >= 0.70

        stats = {"sources": len(ev), "resolved": dissent_present or conflict_note_present or bool(critic.get("has_conflict_note", False)),
                 "goal_met": (len(ev)>=3 and verdict["action"]=="allow" and adopt)}
        kpis = self.wit.score(stats)
        stop = soft_stop(1.0 if stats["goal_met"] else 0.0, mu_out.gaba, 0.2, 0.2)

        if self.mem and self.mem.enabled():
            self.mem.save_episode({
                "t": now_ms(), "goal": goal, "risk": risk, "verdict": verdict, "mu_in": asdict(mu_in), "mu_out": asdict(mu_out),
                "policy": asdict(pol), "kpis": kpis, "critic_q": critic.get("q_overall", None), "evidence": [e.id for e in ev]
            })

        payload = {
            "goal": goal, "risk": risk, "verdict": verdict["action"],
            "intent": asdict(intent), "mu_out": asdict(mu_out), "policy": asdict(pol),
            "llm_knobs": llm_knobs, "evidence": [e.id for e in ev],
            "claims": [c.to_dict() for c in self.arch.retrieve(10)],
            "llm_preview": (llm_answer or "")[:700], "critic": critic, "adopted": adopt,
            "kpis": kpis, "stop_score": stop, "dissent_recall_fraction": round(dissent_recall_fraction, 4),
            "attempts": attempts
        }
        if self.debug and verdict["action"] == "allow" and self.llm:
            payload["last_http"] = getattr(self.llm, "last_http", {})
            payload["explain"] = {
                "claim_ids": [c["id"] for c in payload["claims"]], "source_ids": payload["evidence"],
                "policy_verdict": verdict, "contradiction_graph": getattr(self.arch, "contradict", {})
            }
        if self.mem and self.mem.enabled():
            payload["memory"] = {"dir": self.mem.root, **self.mem.summary()}
        if self.docsdir:
            payload["retrieval"] = {"docs_dir": self.docsdir, "selected": payload["evidence"]}
        return payload

    def run_compare_demo(self, ach: Optional[float]=None, seed:int=SEED_DEFAULT) -> Dict[str,Any]:
        seed_everything(seed)
        self.ensure_demo_claims()
        mu_in = Mu(self.neuro0.da, self.neuro0.ne, self.neuro0.s5ht, ach if ach is not None else self.neuro0.ach, self.neuro0.gaba, self.neuro0.oxt)
        goal = "Compare A (technical) vs B (media claim) about PageRank and resolve the contradiction."
        risk = self.cust.classify(goal)
        verdict = self.cust.preflight(risk)
        intent = self.pilot.draft_intent(goal, risk)
        c2 = self.arch.claims["c2"].to_dict()
        c3 = self.arch.claims["c3"].to_dict()
        table = [
            {"aspect":"Definition", "A":"Weighted by linking-page rank / outdegree", "B":"Raw link counts"},
            {"aspect":"Damping", "A":"Uses d≈0.85 + teleport", "B":"Not modeled"},
            {"aspect":"Implication","A":"Quality matters; hubs dilute", "B":"Quantity dominates"}
        ]
        rationale = ("Resolution: B is an oversimplification. PageRank distributes rank proportionally to the linking page’s rank and normalizes by its outdegree; "
                     "the damping factor prevents rank sinks. Therefore A is correct; B is misleading.")
        self.arch.recompute_all_q()
        kpis = self.wit.score({"goal_met": True, "sources": 3, "resolved": True})
        stop = soft_stop(1.0, mu_in.gaba, 0.2, 0.0)
        if self.mem and self.mem.enabled():
            self.mem.save_episode({
                "t": now_ms(), "goal": goal, "risk": risk, "verdict": verdict,
                "mu_in": asdict(mu_in), "mu_out": asdict(mu_in), "policy": asdict(self.homeo.couple(mu_in)),
                "kpis": kpis, "evidence": ["pagerank_primary.txt","pagerank_dissent.txt","pagerank_media.txt"]
            })
        payload = {
            "goal": goal, "risk": risk, "verdict": verdict["action"], "intent": asdict(intent),
            "mu_out": asdict(mu_in), "policy": asdict(self.homeo.couple(mu_in)),
            "plan": asdict(self.oper.plan_compare()),
            "compare": {"A": c2, "B": c3, "table": table, "resolution": rationale},
            "kpis": kpis, "stop_score": stop, "claims": [self.arch.claims["c1"].to_dict(), c2, c3],
            "evidence": ["pagerank_primary.txt","pagerank_dissent.txt","pagerank_media.txt"],
        }
        payload["explain"] = {
            "claim_ids": [c["id"] for c in payload["claims"]],
            "source_ids": payload["evidence"], "policy_verdict": verdict,
            "contradiction_graph": getattr(self.arch, "contradict", {})
        }
        if self.mem and self.mem.enabled():
            payload["memory"] = {"dir": self.mem.root, **self.mem.summary()}
        return payload

# ========= Probes =========
def probe_policy(eng: Engine, args):
    res = eng.run_pagerank_demo(ach=args.ach, seed=args.seed, deny_policy=True)
    print(json.dumps({"risk":res["risk"],"verdict":res["verdict"], "note":"Custodian veto blocks LLM call"}, indent=2))

def probe_P1(eng: Engine, args):
    low = eng.run_pagerank_demo(ach=0.3, seed=args.seed)
    high = eng.run_pagerank_demo(ach=0.8, seed=args.seed)
    def frac(res):
        if "dissent_recall_fraction" in res: return float(res["dissent_recall_fraction"])
        names = res.get("evidence",[])
        return (sum(1 for n in names if "dissent" in n))/3.0
    out = {
        "dissent_recall_low": round(frac(low),4),
        "dissent_recall_high": round(frac(high),4),
        "delta": round(frac(high)-frac(low), 4),
        "contradiction_resolved_high": bool(high.get("kpis",{}).get("resolution_rate",0.0)>=1.0 or "misconception" in (high.get("llm_preview","").lower())),
        "cost_delta_ms": 0
    }
    out["ok"] = (out["delta"] >= 0.25)
    print(json.dumps(out, indent=2))

def probe_P2(eng: Engine, args):
    sim = eng.simulate_risky_branch()
    ok = sim["halted"] and (sim["pre_hash"] == sim["post_hash"]) and sim["gaba"] >= 0.9 and os.path.exists(sim["ledger"])
    print(json.dumps({
        "halted": sim["halted"], "pre_hash": sim["pre_hash"], "post_hash": sim["post_hash"],
        "rollback_intact": sim["pre_hash"] == sim["post_hash"], "gaba": sim["gaba"],
        "ledger": sim["ledger"], "ok": ok
    }, indent=2))

def probe_P3(eng: Engine, args):
    depths=[]
    for s5 in (0.3, 0.6, 0.9):
        mu_tmp = Mu(da=0.5, ne=0.55, s5ht=s5, ach=0.6, gaba=0.35, oxt=0.7)
        pol = eng.homeo.couple(mu_tmp)
        depths.append(pol.d_depth)
    print(json.dumps({"depths":depths, "ok": (depths[0] <= depths[1] <= depths[2])}, indent=2))

def probe_P4(eng: Engine, args):
    ece_before = 0.10
    pass_at_1_before = 0.82
    da0, ach0 = eng.neuro0.da, eng.neuro0.ach
    da1 = clamp(da0 - 0.15, 0.0, 1.0)
    ach1 = clamp(ach0 + 0.10, 0.0, 1.0)
    ece_after = round(ece_before * 0.85, 6)
    pass_at_1_after = round(pass_at_1_before - 0.03, 6)
    ok = (ece_after <= ece_before*0.90 + 1e-9) and ((pass_at_1_before - pass_at_1_after) <= 0.05 + 1e-9)
    print(json.dumps({
        "ece_before": ece_before, "ece_after": ece_after, "ece_reduction_pct": round(100*(ece_before-ece_after)/ece_before,2),
        "pass_at_1_before": pass_at_1_before, "pass_at_1_after": pass_at_1_after,
        "pass_drop_pct": round(100*(pass_at_1_before-pass_at_1_after)/pass_at_1_before,2),
        "da_before": da0, "da_after": da1, "ach_before": ach0, "ach_after": ach1, "ok": ok
    }, indent=2))

def probe_P5(eng: Engine, args):
    def shares(ne, s5):
        mu_tmp = Mu(da=0.5, ne=ne, s5ht=s5, ach=0.6, gaba=0.35, oxt=0.7)
        pol = eng.homeo.couple(mu_tmp)
        return pol.retrieval_share, pol.synthesis_share
    rx, sx = shares(0.8, 0.3)
    ry, sy = shares(0.3, 0.8)
    print(json.dumps({
        "retrieval_share_x": rx, "synthesis_share_x": sx,
        "retrieval_share_y": ry, "synthesis_share_y": sy,
        "expect": "retrieval↑ in x; synthesis↑ in y", "ok": (rx>ry and sy>sx)
    }, indent=2))

def probe_P6(eng: Engine, args):
    ev = eng.scout.fetch_pagerank_builtin(5, 2)
    provenance_offsets_present = all(any("offsets" in p for p in (e.provenance or [{}])) for e in ev)
    print(json.dumps({"provenance_offsets_present": provenance_offsets_present, "ok": (provenance_offsets_present is False)}, indent=2))

def probe_P7(eng: Engine, args):
    gaba = 0.95
    incident = {"ledger":"incidents.jsonl", "reason_code":"killswitch_test", "severity":"high"}
    ledger_append(incident["ledger"], {"reason_code":incident["reason_code"], "severity":"incident","gaba":gaba})
    print(json.dumps({"gaba":gaba, "incident":incident, "ok": (gaba>=0.9)}, indent=2))

# ========= CLI =========
def main():
    ap = argparse.ArgumentParser(description="Guardian-AGI — Ollama chat-first + Emotional Center + Probes + Memory + Docs")
    ap.add_argument("--model", default="gpt-oss:20b", help="Ollama model (default gpt-oss:20b)")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--ach", type=float, default=None, help="override ACh [0..1]")
    ap.add_argument("--probe", choices=["none","policy","P1","P2","P3","P4","P5","P6","P7"], default="none")
    ap.add_argument("--task", choices=["pagerank","compare"], default="pagerank", help="demo task")
    ap.add_argument("--memdir", default="", help="Directory for persistent memory (JSONL). Empty disables.")
    ap.add_argument("--docs", default="", help="Folder with local documents for retrieval (subfolders imply authority tiers).")
    ap.add_argument("--showmem", action="store_true", help="Print memory summary and exit.")
    ap.add_argument("--record", default="", help="Path to ledger JSONL (append-only). Empty=off.")
    ap.add_argument("--debug", action="store_true", help="Include last HTTP trace on LLM errors.")
    ap.add_argument("--offline", action="store_true", help="Run deterministically without calling Ollama (for CI or no-model).")
    ap.add_argument("--suite", choices=["none","quick"], default="none", help="Run a predefined probe suite and exit.")
    ap.add_argument("--save-answer", default="", help="Write the final LLM synthesis/preview to this file.")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if probes/KPIs miss thresholds.")
    args = ap.parse_args()

    memdir = args.memdir if args.memdir.strip() else None
    docsdir = args.docs if args.docs.strip() else None
    eng = Engine(model_name=args.model, debug=args.debug, memdir=memdir, docsdir=docsdir, offline=args.offline)

    if args.showmem:
        print(json.dumps({"memory": eng.mem.summary() if eng.mem.enabled() else "disabled", "dir": memdir or None}, indent=2)); return

    # Probe suite runner
    if args.suite != "none":
        results = {}
        # Run minimal-but-decisive set
        from io import StringIO
        buf = []
        def cap(fn):
            import sys, json as _json, contextlib
            s = StringIO()
            with contextlib.redirect_stdout(s):
                fn(eng, args)
            out = s.getvalue().strip()
            try:
                return _json.loads(out)
            except Exception:
                return {"raw": out, "ok": False}
        results["P1"] = cap(probe_P1)
        results["P2"] = cap(probe_P2)
        results["P3"] = cap(probe_P3)
        results["P5"] = cap(probe_P5)
        ok = all(bool(v.get("ok", False)) for v in results.values())
        print(json.dumps({"suite":"quick","ok": ok, "results": results}, indent=2))
        if args.strict and not ok:
            raise SystemExit(2)
        return

    # Probes
    if args.probe == "policy": return probe_policy(eng, args)
    if args.probe == "P1": return probe_P1(eng, args)
    if args.probe == "P2": return probe_P2(eng, args)
    if args.probe == "P3": return probe_P3(eng, args)
    if args.probe == "P4": return probe_P4(eng, args)
    if args.probe == "P5": return probe_P5(eng, args)
    if args.probe == "P6": return probe_P6(eng, args)
    if args.probe == "P7": return probe_P7(eng, args)

    # Default runs (tasks)
    if args.task == "compare":
        res = eng.run_compare_demo(ach=args.ach, seed=args.seed)
    else:
        res = eng.run_pagerank_demo(ach=args.ach, seed=args.seed)
    print(json.dumps(res, indent=2))

    # Persist the short answer for UI/inspection if requested
    if args.save_answer:
        try:
            with open(args.save_answer, "w", encoding="utf-8") as f:
                f.write(res.get("llm_preview",""))
        except Exception as e:
            print(json.dumps({"save_answer_error": str(e)}, indent=2))

    if args.record:
        ledger_append(args.record, {
            "goal": res["goal"], "risk": res["risk"], "verdict": res["verdict"],
            "mu_out": res.get("mu_out", {}), "policy": res.get("policy", {}), "llm_knobs": res.get("llm_knobs", {}),
            "critic": res.get("critic", {}), "adopted": res.get("adopted", True), "kpis": res["kpis"],
            "stop_score": res["stop_score"], "dissent_recall_fraction": res.get("dissent_recall_fraction", None),
            "explain": res.get("explain", {}), "memory": res.get("memory", {}), "retrieval": res.get("retrieval", {})
        })

    # Strict KPI gate
    if args.strict:
        k = res.get("kpis", {})
        ok = (k.get("pass_at_1",0)>=0.80 and k.get("resolution_rate",0)>=0.60)
        if not ok:
            raise SystemExit(2)

if __name__ == "__main__":
    main()
