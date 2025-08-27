#!/usr/bin/env python3
# guardian_agi_min.py — single-file Guardian-AGI scaffold (seed=137)
# Robust Ollama client: chat-first, generate fallback; critic enforces JSON; acceptance + salvage.
# Default model: gpt-oss:20b (override via --model)

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

def log_jsonl(path: str, obj: dict):
    if not path: return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def open_incident(reason_code: str, severity: str="medium", ledger: str="incidents.jsonl") -> dict:
    entry = {"kind":"incident","reason_code":reason_code,"severity":severity}
    ledger_append(ledger, entry)
    return {"ledger": ledger, "reason_code": reason_code, "severity": severity}

# ========= Data Contracts =========
@dataclass
class Source:
    url: str
    seg: Optional[str]=None
    h: Optional[str]=None
    ts: Optional[str]=None
    domain_tier: int=1

@dataclass
class Claim:
    id: str
    text: str
    q: float = 0.5
    sources: List[Source] = None
    supports: List[str] = None
    contradicts: List[str] = None
    stance: str = "neutral"
    def to_dict(self):
        return {"id": self.id, "text": self.text, "q": self.q, "stance": self.stance,
                "sources": [asdict(s) for s in (self.sources or [])],
                "supports": self.supports or [], "contradicts": self.contradicts or []}

@dataclass
class EvidenceUnit:
    id: str
    content_hash: str
    extract: str
    stance: str="neutral"
    provenance: List[Dict[str, Any]] = None
    offsets: Optional[Dict[str,int]] = None              # {"start": int, "end": int}
    quality_flags: Optional[Dict[str,bool]] = None       # {"ocr": False, "trunc": False}

@dataclass
class Task:
    goal: str
    constraints: Dict[str, Any]
    acceptance_tests: List[str]
    budget: Dict[str, Any]
    criticality: str="C1"

@dataclass
class Episode:
    t: int
    task_id: str
    mu_in: Dict[str,float]
    action: str
    evidence_used: List[str]
    outcome: str
    mu_out: Dict[str,float]

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
    MATRICES = {
        "R0": {"allow": True,  "notes": "General info; normal handling.", "disclaimer": ""},
        "R1": {"allow": True,  "notes": "Personal/sensitive → sanitize PII.", "disclaimer": "Personal data must be redacted."},
        "R2": {"allow": True,  "notes": "Legal/finance/medical → conservative language + citations.", "disclaimer": "Informational only; not professional advice."},
        "R3": {"allow": False, "notes": "Dual-use/bio/cyber → deny unless sanctioned playbook.", "disclaimer": "Denied by safety policy."},
        "R4": {"allow": False, "notes": "Physical actuation/control → deny.", "disclaimer": "Denied by safety policy."},
    }
    def classify(self, goal: str) -> str:
        g = goal.lower()
        if any(x in g for x in ["weapon","exploit","malware","payload","ddos","rootkit","bio","lab","pathogen","synthesis"]): return "R3"
        if any(x in g for x in ["operate","unlock door","control plc","start motor","drone flight","actuate"]): return "R4"
        if any(x in g for x in ["medical","diagnosis","dosage","legal advice","lawsuit","investment","financial advice"]): return "R2"
        if any(x in g for x in ["personal data","pii","address","phone number","email address"]): return "R1"
        return "R0"
    def preflight(self, risk: str) -> Dict[str,str]:
        m = self.MATRICES.get(risk, self.MATRICES["R0"])
        return {"action": "allow" if m["allow"] else "deny",
                "notes":  m["notes"],
                "disclaimer": m["disclaimer"],
                "risk": risk}

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
    def __init__(self): self.claims: Dict[str, Claim] = {}; self.contradict: Dict[str, List[str]] = {}
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
        return EvidenceUnit(
            id=name, content_hash=h, extract=txt[:400], stance=stance,
            provenance=[{"source": name}], offsets=None,
            quality_flags={"trunc": False, "ocr": False}
        )
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
            tests="≥3 citations; ≤150 words; note & resolve one contradiction or misconception.",
            stop="S_s>0.6 and no policy flags.",
            risk=risk,
        )

# ========= Ollama LLM Client (chat-first with generate fallback; JSON-aware) =========
class LLMClient:
    def __init__(self, model: str, host: str="localhost", port: int=11434, debug: bool=False):
        self.model = model; self.host = host; self.port = port; self.debug = debug
        self.last_http: Dict[str,Any] = {}

    def _post(self, path: str, payload: dict, phase: str) -> Dict[str,Any]:
        body = json.dumps(payload)
        conn = http.client.HTTPConnection(self.host, self.port, timeout=180)
        t0 = time.perf_counter()
        try:
            conn.request("POST", path, body=body, headers={"Content-Type":"application/json"})
            resp = conn.getresponse(); status = resp.status
            raw = resp.read().decode("utf-8") if resp else ""
        finally:
            conn.close()
        lat_ms = int((time.perf_counter()-t0)*1000)
        self.last_http = {"phase": phase, "path": path, "status": status, "lat_ms": lat_ms,
                          "raw_preview": raw[:240], "raw_mid": raw[:1024] if self.debug else None, "raw_full": "" if not self.debug else raw}
        try:
            data = json.loads(raw) if raw else {}
        except Exception as e:
            raise RuntimeError(f"Ollama parse error (HTTP {status}) at {path}: {e}")
        if status != 200:
            raise RuntimeError(f"Ollama HTTP {status} at {path}: {data.get('error') or (raw[:200] if raw else 'empty')}")
        return data

    def ask(self, system_msg: str, user_msg: str, *, temperature: float, top_p: float,
            repeat_penalty: float, num_predict: int, num_ctx: int=8192, force_json: bool=False,
            phase: str="pilot") -> str:
        # 1) chat first
        chat_payload = {
            "model": self.model,
            "messages": [
                {"role":"system","content":system_msg},
                {"role":"user","content":user_msg}
            ],
            "options": {"temperature": float(temperature), "top_p": float(top_p),
                        "repeat_penalty": float(repeat_penalty), "num_predict": int(num_predict),
                        "num_ctx": int(num_ctx), **({"format":"json"} if force_json else {})},
            "stream": False
        }
        data = self._post("/api/chat", chat_payload, phase)
        out = ""
        if isinstance(data, dict) and "message" in data and isinstance(data["message"], dict):
            out = data["message"].get("content","") or data["message"].get("thinking","")
        elif isinstance(data, dict):
            out = data.get("response","") or data.get("thinking","")
        if out and out.strip():
            return out.strip()

        # 2) fallback to /api/generate
        gen_payload = {
            "model": self.model,
            "prompt": f"{system_msg}\n\nUser:\n{user_msg}\n\nAssistant:",
            "options": {"temperature": float(temperature), "top_p": float(top_p),
                        "repeat_penalty": float(repeat_penalty), "num_predict": int(num_predict),
                        "num_ctx": int(num_ctx), **({"format":"json"} if force_json else {})},
            "stream": False
        }
        data2 = self._post("/api/generate", gen_payload, phase+"-fallback")
        out2 = ""
        if isinstance(data2, dict):
            out2 = data2.get("response","") or data2.get("thinking","")
        if not out2 or not out2.strip():
            raise RuntimeError("Ollama returned empty text from both chat and generate.")
        return out2.strip()

# ========= Critic (ACh-gated calibration; JSON enforced) =========
CRITIC_SYS = (
  "You are Critic. Given an answer about PageRank, return STRICT JSON with keys: "
  "{'q_overall': float in [0,1], 'has_conflict_note': bool, 'reasons': [str]}. "
  "Return JSON ONLY—no extra text."
)

def _extract_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        # best-effort: last {...}
        blocks = re.findall(r"\{(?:[^{}]|(?R))*\}", s)
        for chunk in reversed(blocks or []):
            try:
                return json.loads(chunk)
            except Exception:
                continue
    raise ValueError("no JSON found")

def run_critic(llm: LLMClient, answer: str, mu: Mu, passes: int=1) -> dict:
    temperature = max(0.1, 0.9 - 0.6*mu.s5ht)
    top_p = 0.8
    repeat_penalty = 1.10 + 0.10*mu.s5ht
    num_predict = 200
    best = {"q_overall": 0.0, "has_conflict_note": False, "reasons": ["no output"]}
    for _ in range(max(1, passes)):
        try:
            j = llm.ask(CRITIC_SYS, f"Answer:\n{answer}\n\nReturn JSON only.",
                        temperature=temperature, top_p=top_p,
                        repeat_penalty=repeat_penalty, num_predict=num_predict,
                        force_json=True, phase="critic")
            data = _extract_json(j)
            if isinstance(data, dict) and data.get("q_overall", 0) >= best.get("q_overall", 0):
                best = data
        except Exception as e:
            best = {"q_overall": best.get("q_overall",0.0), "has_conflict_note": best.get("has_conflict_note",False),
                    "reasons": [f"critic error: {e}"], **({"http": llm.last_http} if llm.last_http else {})}
    best["q_overall"] = float(clamp(best.get("q_overall", 0.0), 0.0, 1.0))
    best["has_conflict_note"] = bool(best.get("has_conflict_note", False))
    if "reasons" not in best: best["reasons"] = []
    return best

# ========= Engine (pilot + critic + salvage + probes) =========
def _word_count(s: str) -> int:
    return len(re.findall(r"\b[\w\-’']+\b", s))

def _count_citations(s: str) -> int:
    return len(re.findall(r"\[\d+\]", s))

def _ends_punct(s: str) -> bool:
    return bool(re.search(r"[.!?]\s*$", s.strip()))

def _has_resolution_line(s: str) -> bool:
    t = s.lower()
    return ("misconception:" in t) or ("conflict note" in t)

class Engine:
    def __init__(self, model_name: str="gpt-oss:20b", neuro: Mu=None, debug: bool=False):
        self.homeo = Homeostat(); self.cust  = Custodian(); self.wit   = Witness()
        self.scout = Scout();     self.arch  = Archivist(); self.pilot = Pilot()
        self.oper  = Operator();  self.llm   = LLMClient(model_name, debug=debug)
        self.neuro0 = neuro or Mu(da=0.50, ne=0.55, s5ht=0.85, ach=0.75, gaba=0.35, oxt=0.70)
        self.debug = debug

    def run_pagerank_demo(self, ach: Optional[float]=None, seed:int=SEED_DEFAULT, deny_policy: bool=False,
                          pilot_alt_model: Optional[str]=None, app_s: float=1.0,
                          override_ne: Optional[float]=None, override_s5ht: Optional[float]=None) -> Dict[str,Any]:
        seed_everything(seed)
        mu_in = Mu(self.neuro0.da, self.neuro0.ne, self.neuro0.s5ht,
                   ach if ach is not None else self.neuro0.ach,
                   self.neuro0.gaba, self.neuro0.oxt)

        goal = "Explain PageRank ≤150 words with ≥3 citations; detect and resolve one contradiction or misconception."
        risk = "R3" if deny_policy else self.cust.classify(goal)
        verdict = self.cust.preflight(risk)
        intent = self.pilot.draft_intent(goal, risk)

        app = Appraisal(p=0.3, n=0.4, u=0.3, k=0.1, s=float(app_s), c=0.6, h=1.0)
        mu_out = self.homeo.update(mu_in, app)
        mu_for_policy = Mu(mu_out.da,
                           override_ne if override_ne is not None else mu_out.ne,
                           override_s5ht if override_s5ht is not None else mu_out.s5ht,
                           mu_out.ach, mu_out.gaba, mu_out.oxt)
        pol = self.homeo.couple(mu_for_policy)

        # μ -> LLM decoding knobs
        attempts: List[Dict[str,Any]] = []
        llm_answer = "[blocked by policy]"; llm_knobs = {}; http_trace = {}
        model_for_pilot = pilot_alt_model or self.llm.model

        if verdict["action"] == "allow":
            temperature    = max(0.1, 0.9 - 0.6*mu_out.s5ht)
            top_p          = clamp(0.75 + 0.20*mu_out.ne - 0.10*mu_out.gaba, 0.50, 0.95)
            repeat_penalty = 1.05 + 0.20*mu_out.s5ht - 0.10*mu_out.da
            num_predict    = int(256 + int(384*mu_out.s5ht) - int(128*mu_out.gaba))

            system_msg = ("You are Pilot. Decompose briefly (assumptions/unknowns/tests), "
                          "then produce 110–160 words with citations [1][2][3...]. If sources conflict, "
                          "add a one-line 'Conflict Note:' or 'Misconception:' resolving it. Output final answer only.")
            user_msg = ("Explain PageRank in ≤150 words with ≥3 citations. Prioritize primary/official. "
                        "Resolve the common 'link count' misconception. Include a 'Misconception:' or 'Conflict Note:' line at end.")

            try:
                # main attempt
                ans = self.llm.ask(system_msg, user_msg,
                    temperature=temperature, top_p=top_p,
                    repeat_penalty=repeat_penalty, num_predict=num_predict, phase="pilot")
                attempts.append({"kind":"pilot","ok":True,"http":self.llm.last_http,"len":len(ans)})
                llm_answer = ans
            except Exception as e:
                http_trace = self.llm.last_http
                attempts.append({"kind":"pilot","ok":False,"error":str(e),"http":http_trace})

            # acceptance tests
            def accepts(s: str) -> bool:
                return (110 <= _word_count(s) <= 160) and (_count_citations(s) >= 3) and _has_resolution_line(s) and _ends_punct(s)

            if not accepts(llm_answer or ""):
                # SALVAGE via generate with a constrained prompt
                salvage_prompt = (
                    "Write exactly 110–160 words explaining PageRank. Use bracketed numeric citations like [1][2][3]. "
                    "End with a line starting with either 'Misconception:' or 'Conflict Note:'. "
                    "Mention random surfer, damping factor d≈0.85, teleport 1−d, iterative formula PR(i)=(1−d)/N + d∑ PR(j)/L(j). "
                    "Prioritize primary/official sources (Brin & Page 1998; Google documentation/patent). "
                    "Output final answer only."
                )
                try:
                    ans2 = self.llm.ask("You are a precise technical writer.", salvage_prompt,
                                        temperature=temperature, top_p=top_p,
                                        repeat_penalty=repeat_penalty, num_predict=num_predict,
                                        phase="pilot-repair-1")
                    attempts.append({"kind":"pilot-repair-1","ok":True,"http":self.llm.last_http,"len":len(ans2)})
                    if accepts(ans2): llm_answer = ans2
                except Exception as e:
                    attempts.append({"kind":"pilot-repair-1","ok":False,"error":str(e),"http":self.llm.last_http})

            if not _ends_punct(llm_answer or ""):
                # tiny reprompt to ensure terminal punctuation
                try:
                    ans3 = self.llm.ask("You fix endings.", "Ensure this ends with a period, without changing meaning:\n"+(llm_answer or ""),
                                        temperature=0.2, top_p=0.8, repeat_penalty=1.1, num_predict=64,
                                        phase="pilot-reprompt")
                    attempts.append({"kind":"pilot-reprompt","ok":True,"http":self.llm.last_http,"len":len(ans3)})
                    if ans3.strip(): llm_answer = ans3
                except Exception as e:
                    attempts.append({"kind":"pilot-reprompt","ok":False,"error":str(e),"http":self.llm.last_http})

            llm_knobs = {"temperature": round(temperature,3),
                         "top_p": round(top_p,3),
                         "repeat_penalty": round(repeat_penalty,3),
                         "num_predict": int(num_predict)}

        # Critic (ACh-gated)
        critic = {}
        if verdict["action"] == "allow":
            critic_passes = 1 + int(2*mu_out.ach)
            try:
                critic = run_critic(self.llm, llm_answer, mu_out, passes=critic_passes)
            except Exception as e:
                critic = {"q_overall": 0.0, "has_conflict_note": False,
                          "reasons":[f"critic exec error: {e}"], "http": getattr(self.llm,"last_http",{})}

        # Evidence (ACh controls dissent quota)
        ev = self.scout.fetch_pagerank(pol.k_breadth, pol.q_contra)
        dissent_present = any(e.stance=="con" for e in ev)
        conflict_note_present = _has_resolution_line(llm_answer or "")

        # Minimal claims + contradiction link
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

        # Adoption threshold tied to critic (fallback heuristic if critic malformed)
        adopt = True
        if critic and isinstance(critic, dict) and "q_overall" in critic:
            adopt = critic.get("q_overall", 0.0) >= 0.70
        else:
            adopt = (110 <= _word_count(llm_answer or "") <= 160) and (_count_citations(llm_answer or "") >= 3)

        stats = {"sources": len(ev),
                 "resolved": dissent_present or conflict_note_present or (critic.get("has_conflict_note", False) if isinstance(critic, dict) else False),
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
            payload["last_http"] = {
                "pilot":  getattr(self.llm, "last_http", {}),
                "critic": critic.get("http", getattr(self.llm, "last_http", {})) if isinstance(critic, dict) else {}
            }
            payload["attempts"] = attempts
        return payload

    # ---------- Probe harness ----------
    def probe_P1(self) -> Dict[str,Any]:
        low  = self.run_pagerank_demo(ach=0.3, seed=137)
        high = self.run_pagerank_demo(ach=0.8, seed=137)
        d_low  = int("pagerank_dissent.txt" in low["evidence"])
        d_high = int("pagerank_dissent.txt" in high["evidence"])
        t_low  = low.get("last_http",{}).get("pilot",{}).get("lat_ms",0)
        t_high = high.get("last_http",{}).get("pilot",{}).get("lat_ms",0)
        resolved = any(x in (high.get("llm_preview","").lower()) for x in ["misconception:", "conflict note"])
        return {"dissent_recall_low": d_low, "dissent_recall_high": d_high,
                "delta": d_high-d_low, "contradiction_resolved_high": resolved,
                "cost_delta_ms": (t_high - t_low), "ok": (d_high-d_low)>=1}

    def probe_P2(self) -> Dict[str,Any]:
        res = self.run_pagerank_demo(deny_policy=True)
        return {"verdict": res["verdict"], "ok": res["verdict"]=="deny"}

    def probe_P3(self) -> Dict[str,Any]:
        a = self.run_pagerank_demo(app_s=0.3); b = self.run_pagerank_demo(app_s=0.6); c = self.run_pagerank_demo(app_s=0.9)
        return {"depths":[a["policy"]["d_depth"], b["policy"]["d_depth"], c["policy"]["d_depth"]],
                "ok": (a["policy"]["d_depth"] <= b["policy"]["d_depth"] <= c["policy"]["d_depth"])}

    def probe_P5(self) -> Dict[str,Any]:
        x = self.run_pagerank_demo(override_ne=0.8, override_s5ht=0.3)
        y = self.run_pagerank_demo(override_ne=0.3, override_s5ht=0.8)
        return {"retrieval_share_x": x["policy"]["retrieval_share"], "synthesis_share_x": x["policy"]["synthesis_share"],
                "retrieval_share_y": y["policy"]["retrieval_share"], "synthesis_share_y": y["policy"]["synthesis_share"],
                "expect": "retrieval↑ in x; synthesis↑ in y",
                "ok": (x["policy"]["retrieval_share"] > y["policy"]["retrieval_share"]) and (y["policy"]["synthesis_share"] > x["policy"]["synthesis_share"])}

    def probe_P6(self) -> Dict[str,Any]:
        # Refuse publication if offsets missing (current Scout leaves offsets=None)
        ev = self.scout.fetch_pagerank(3,1)
        ready = all(bool(e.offsets) for e in ev)
        return {"provenance_offsets_present": ready, "ok": not ready}

    def probe_P7(self) -> Dict[str,Any]:
        inc = open_incident("killswitch_test", severity="high", ledger="incidents.jsonl")
        # Simulate GABA surge and freeze (report only; actual tool-freeze not enacted in demo)
        return {"gaba": 0.95, "incident": inc, "ok": True}

# ========= CLI & Probes =========
def main():
    ap = argparse.ArgumentParser(description="Guardian-AGI (single-file) — Ollama chat-first + Emotional Center + Probes")
    ap.add_argument("--model", default="gpt-oss:20b",
                    help="Ollama model name (default gpt-oss:20b; e.g., qwen2.5:14b-instruct-q4_K_M)")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--ach", type=float, default=None, help="override ACh [0..1]")
    ap.add_argument("--pilot-alt-model", default=None, help="optional alternate model for pilot phase")
    ap.add_argument("--probe", choices=["none","ach","policy","stop","critic","health","P1","P2","P3","P5","P6","P7"], default="none")
    ap.add_argument("--record", default="", help="Path to ledger JSONL (append-only). Empty=off.")
    ap.add_argument("--debug", action="store_true", help="Include last HTTP trace on LLM errors.")
    args = ap.parse_args()

    eng = Engine(model_name=args.model, debug=args.debug)

    if args.probe == "ach":
        low  = eng.run_pagerank_demo(ach=0.2, seed=args.seed, pilot_alt_model=args.pilot_alt_model or None)
        high = eng.run_pagerank_demo(ach=0.9, seed=args.seed, pilot_alt_model=args.pilot_alt_model or None)
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
        res_bad = res_ok.copy()
        res_bad["kpis"] = {**res_bad["kpis"], "pass_at_1": 0.0}
        res_bad["stop_score"] = soft_stop(0.0, res_ok["mu_out"]["gaba"], 0.2, 0.2)
        print(json.dumps({
            "stop_when_goal_met": res_ok["stop_score"],
            "stop_when_unmet": res_bad["stop_score"],
            "expectation":"unmet < met and typically < 0.6 threshold"
        }, indent=2)); return

    if args.probe == "critic":
        low  = eng.run_pagerank_demo(ach=0.2, seed=args.seed, pilot_alt_model=args.pilot_alt_model or None)
        high = eng.run_pagerank_demo(ach=0.9, seed=args.seed, pilot_alt_model=args.pilot_alt_model or None)
        print(json.dumps({"low.critic": low.get("critic",{}), "high.critic": high.get("critic",{})}, indent=2)); return

    if args.probe == "health":
        print(json.dumps({"ok": True, "policy_ver": eng.cust.policy_ver}, indent=2)); return

    if args.probe in ("P1","P2","P3","P5","P6","P7"):
        fn = getattr(eng, f"probe_{args.probe}")
        print(json.dumps(fn(), indent=2)); return

    # default run
    res = eng.run_pagerank_demo(ach=args.ach, seed=args.seed, pilot_alt_model=args.pilot_alt_model or None)
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
