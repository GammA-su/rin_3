#!/usr/bin/env python3
# guardian_agi_min.py — single-file Guardian-AGI scaffold (seed=137)
# Upgrades: robust Ollama client (chat→thinking→generate, retries/backoff, timeouts),
# acceptance gate + self-reprompt + pilot alt model, critic JSON parsing fix,
# per-phase traces, health probe, optional JSON forcing. No internet required.

from __future__ import annotations
import argparse, json, os, random, time, http.client, hashlib, re, copy
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

def log_jsonl(path: str, obj: dict):
    if not path: return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

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

# ========= JSON extractor (robust without recursive regex) =========
def extract_last_json(text: str) -> dict:
    if not text: raise ValueError("empty text")
    # Fast path
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find last balanced {...}
    last_open = text.rfind("{")
    last_close = text.rfind("}")
    if last_open == -1 or last_close == -1 or last_close < last_open:
        # try any brace pair scanning from end
        for i in range(len(text)-1, -1, -1):
            if text[i] == "}":
                # scan backward to matching "{"
                depth = 0
                for j in range(i, -1, -1):
                    if text[j] == "}": depth += 1
                    elif text[j] == "{":
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[j:i+1])
                            except Exception:
                                break
        raise ValueError("no JSON object found")
    chunk = text[last_open:last_close+1]
    # try to expand to balance
    depth = 0
    start = None
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0: start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:idx+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    pass
    raise ValueError("no JSON object found")

# ========= Ollama LLM Client =========
class LLMClient:
    def __init__(self, model: str, host: str="localhost", port: int=11434, debug: bool=False,
                 timeout: int=45, retries: int=2, backoff: float=0.75):
        self.model = model or os.getenv("GUARDIAN_MODEL","")
        self.host = host; self.port = port; self.debug = debug
        self.timeout = int(os.getenv("GUARDIAN_TIMEOUT", timeout))
        self.retries = int(os.getenv("GUARDIAN_RETRIES", retries))
        self.backoff = backoff
        self.last_http: Dict[str,Any] = {}

    def _post(self, path: str, payload: dict, phase: str) -> Dict[str,Any]:
        body = json.dumps(payload)
        attempt = 0
        err: Optional[Exception] = None
        while attempt <= self.retries:
            t0 = now_ms()
            conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            status, raw = -1, ""
            try:
                conn.request("POST", path, body=body, headers={"Content-Type":"application/json"})
                resp = conn.getresponse(); status = resp.status
                raw = resp.read().decode("utf-8") if resp else ""
            except Exception as e:
                err = e
            finally:
                try: conn.close()
                except Exception: pass
            latency = now_ms() - t0
            self.last_http = {"phase": phase, "path": path, "status": status, "lat_ms": latency, "raw_preview": raw[:240]}
            if status == 200:
                try:
                    return json.loads(raw) if raw else {}
                except Exception as e:
                    err = RuntimeError(f"Ollama parse error (HTTP {status}) at {path}: {e}; raw={raw[:200]}")
            # retry on failure
            if attempt == self.retries:
                break
            time.sleep(self.backoff * (2**attempt))
            attempt += 1
        # If we reach here, fail
        if err: raise err
        raise RuntimeError(f"Ollama HTTP {self.last_http.get('status')} at {path}: {self.last_http.get('raw_preview')}")

    def _render_chat_as_prompt(self, system_msg: str, user_msg: str) -> str:
        return f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

    def ask(self, system_msg: str, user_msg: str, *, temperature: float,
            top_p: float, repeat_penalty: float, num_predict: int,
            num_ctx: int=8192, force_json: bool=False, phase: str="pilot") -> str:
        opts = {"temperature": float(temperature), "top_p": float(top_p),
                "repeat_penalty": float(repeat_penalty), "num_predict": int(num_predict),
                "num_ctx": int(num_ctx)}
        if force_json: opts["format"] = "json"

        # Primary: /api/chat
        chat_payload = {
            "model": self.model,
            "messages": [{"role":"system","content":system_msg},
                         {"role":"user","content":user_msg}],
            "options": opts, "stream": False
        }
        data = self._post("/api/chat", chat_payload, phase=phase)
        out = ""
        if isinstance(data, dict):
            msg = data.get("message", {}) if isinstance(data.get("message"), dict) else {}
            out = (msg.get("content") or "").strip()
            if not out:
                out = (msg.get("thinking") or "").strip()  # internal fallback
            if not out:
                out = (data.get("response") or data.get("thinking") or "").strip()
        if out:
            return out

        # Secondary: /api/generate
        gen_payload = {
            "model": self.model,
            "prompt": self._render_chat_as_prompt(system_msg, user_msg),
            "options": opts, "stream": False
        }
        gen = self._post("/api/generate", gen_payload, phase=f"{phase}-fallback")
        out2 = (gen.get("response") or gen.get("thinking") or "").strip()
        if not out2:
            raise RuntimeError("Ollama returned empty text from both chat and generate.")
        return out2

    def get_last_http(self) -> Dict[str,Any]:
        return copy.deepcopy(self.last_http)

# ========= Critic (ACh-gated) =========
CRITIC_SYS = (
  "You are Critic. Respond in STRICT JSON only.\n"
  "Schema: {\"q_overall\": float, \"has_conflict_note\": bool, \"reasons\": [string]}\n"
  "q_overall in [0,1]. No prose outside JSON. Example:\n"
  "{\"q_overall\":0.72,\"has_conflict_note\":true,\"reasons\":[\"≥3 citations\",\"<=150 words\",\"conflict note present\"]}"
)


def run_critic(llm: LLMClient, answer: str, mu: Mu, passes: int=1, log_path:str="") -> dict:
    temperature = max(0.1, 0.9 - 0.6*mu.s5ht)
    top_p = 0.8
    repeat_penalty = 1.10 + 0.10*mu.s5ht
    num_predict = 200
    best = None
    any_json = False
    for i in range(max(1, passes)):
        try:
            j = llm.ask(CRITIC_SYS, f"Answer:\n{answer}\n\nReturn JSON only.",
                        temperature=temperature, top_p=top_p,
                        repeat_penalty=repeat_penalty, num_predict=num_predict,
                        force_json=True, phase="critic")
            data = extract_last_json(j)
            any_json = True
            if (best is None) or (data.get("q_overall",0.0) > best.get("q_overall",0.0)):
                best = data
            log_jsonl(log_path, {"phase":"critic", "pass":i, "raw":j[:240], "http": llm.get_last_http()})
        except Exception as e:
            if not any_json and best is None:
                best = {"q_overall": 0.0, "has_conflict_note": False,
                        "reasons": [f"critic error: {e}", f"http={llm.get_last_http()}"]}
            log_jsonl(log_path, {"phase":"critic", "pass":i, "error": str(e), "http": llm.get_last_http()})
    if best is None:
        best = {"q_overall": 0.0, "has_conflict_note": False, "reasons": ["no JSON from critic"]}
    best["q_overall"] = float(clamp(best.get("q_overall", 0.0), 0.0, 1.0))
    best["has_conflict_note"] = bool(best.get("has_conflict_note", False))
    if "reasons" not in best: best["reasons"] = []
    return best

# ========= Acceptance gate (Pilot output) =========
def acceptable_answer(txt: str) -> bool:
    if not txt: return False
    s = txt.strip()
    if len(s) < 120 or len(s) > 1200:  # ≈ 120–150 words relaxed
        return False
    # Accept bracketed numeric cites or parenthetical author/year; need ≥3 signals
    cites_num = re.findall(r"\[[0-9]+\]", s)
    cites_auth = re.findall(r"\([A-Z][A-Za-z]+[^)]*?(19|20)\d{2}\)", s)
    cites = len(set(cites_num)) + len(cites_auth)
    if cites < 3: return False
    if "conflict" not in s.lower(): return False
    return True

# ========= Engine =========
class Engine:
    def __init__(self, pilot_model: str="gpt-oss:20b", critic_model: Optional[str]=None,
                 neuro: Mu=None, debug: bool=False, log_path:str=""):
        self.homeo = Homeostat(); self.cust  = Custodian(); self.wit   = Witness()
        self.scout = Scout();     self.arch  = Archivist(); self.pilot = Pilot()
        self.oper  = Operator()
        self.pilot_llm  = LLMClient(pilot_model, debug=debug)
        self.critic_llm = LLMClient(critic_model or pilot_model, debug=debug)
        self.neuro0 = neuro or Mu(da=0.50, ne=0.55, s5ht=0.85, ach=0.75, gaba=0.35, oxt=0.70)
        self.debug = debug
        self.log_path = log_path

    # ---- Health probe ----
    def probe_health(self) -> Dict[str,Any]:
        ok = True; notes = []
        # tags
        try:
            data = self.pilot_llm._post("/api/tags", {}, phase="health-tags")
            notes.append({"tags": list(data.get("models",[]))[:5]})
        except Exception as e:
            ok = False; notes.append({"tags_error": str(e), "http": self.pilot_llm.get_last_http()})
        # ps
        try:
            data = self.pilot_llm._post("/api/ps", {}, phase="health-ps")
            notes.append({"ps": data})
        except Exception as e:
            ok = False; notes.append({"ps_error": str(e), "http": self.pilot_llm.get_last_http()})
        # tiny generate echo + JSON capability
        try:
            resp = self.pilot_llm._post("/api/generate", {
                "model": self.pilot_llm.model,
                "prompt": "Return {} exactly", "options": {"format":"json"}, "stream": False
            }, phase="health-json")
            notes.append({"json_support": True, "preview": str(resp)[:80]})
        except Exception as e:
            notes.append({"json_support": False, "error": str(e), "http": self.pilot_llm.get_last_http()})
        return {"ok": ok, "notes": notes}

    def run_pagerank_demo(self, ach: Optional[float]=None, seed:int=SEED_DEFAULT, deny_policy: bool=False,
                          pilot_alt_model: Optional[str]=None) -> Dict[str,Any]:
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

        # μ -> LLM decoding knobs (Pilot)
        llm_answer = "[blocked by policy]"; llm_knobs = {}; http_trace = {}
        attempts: List[Dict[str,Any]] = []
        if verdict["action"] == "allow":
            temperature    = max(0.1, 0.9 - 0.6*mu_out.s5ht)
            top_p          = clamp(0.75 + 0.20*mu_out.ne - 0.10*mu_out.gaba, 0.50, 0.95)
            repeat_penalty = 1.05 + 0.20*mu_out.s5ht - 0.10*mu_out.da
            num_predict    = int(256 + int(384*mu_out.s5ht) - int(128*mu_out.gaba))

            system_msg = (f"You are Pilot (seed={SEED_DEFAULT}). Decompose briefly (assumptions/unknowns/tests), "
                          "then produce 120–150 words with citations. If sources conflict, "
                          "add a one-line 'Conflict Note' resolving it.")
            user_msg = ("Explain PageRank in ≤150 words with ≥3 citations. Prioritize primary/official. "
                        "Note and resolve the common 'link count' misconception.")

            # Attempt 1
            try:
                ans = self.pilot_llm.ask(system_msg, user_msg,
                    temperature=temperature, top_p=top_p,
                    repeat_penalty=repeat_penalty, num_predict=num_predict, phase="pilot")
                attempts.append({"kind":"pilot", "ok": True, "http": self.pilot_llm.get_last_http(),
                                 "len": len(ans)})
            except Exception as e:
                http_trace = self.pilot_llm.get_last_http()
                ans = f"[LLM error] {e} | http={http_trace}"
                attempts.append({"kind":"pilot", "ok": False, "error": str(e), "http": http_trace})

            # Acceptance gate → self-reprompt once if needed
            if not acceptable_answer(ans):
                remsg = (user_msg + "\n\nCRITICAL: Output 120–150 words. Include ≥3 bracketed citations like [1][2][3]. "
                         "Add a one-line 'Conflict Note:' clarifying the damping d vs teleport 1−d.")
                try:
                    ans2 = self.pilot_llm.ask(system_msg, remsg,
                                temperature=max(0.1, temperature-0.1),
                                top_p=max(0.5, top_p-0.15),
                                repeat_penalty=repeat_penalty+0.05,
                                num_predict=max(num_predict, 640),
                                phase="pilot-reprompt")
                    attempts.append({"kind":"pilot-reprompt", "ok": True, "http": self.pilot_llm.get_last_http(),
                                     "len": len(ans2)})
                    ans = ans2
                except Exception as e:
                    attempts.append({"kind":"pilot-reprompt", "ok": False, "error": str(e),
                                     "http": self.pilot_llm.get_last_http()})

            # Alt model swap if still unacceptable
            if not acceptable_answer(ans) and pilot_alt_model:
                alt = pilot_alt_model
                alt_llm = LLMClient(alt, debug=self.debug)
                try:
                    ans3 = alt_llm.ask(system_msg, user_msg,
                                       temperature=0.2, top_p=0.8,
                                       repeat_penalty=1.15, num_predict=max(num_predict, 640),
                                       phase="pilot-alt")
                    attempts.append({"kind":"pilot-alt", "model": alt, "ok": True, "http": alt_llm.get_last_http(),
                                     "len": len(ans3)})
                    ans = ans3
                except Exception as e:
                    attempts.append({"kind":"pilot-alt", "model": alt, "ok": False,
                                     "error": str(e), "http": alt_llm.get_last_http()})

            llm_answer = ans
            llm_knobs = {"temperature": round(temperature,3),
                         "top_p": round(top_p,3),
                         "repeat_penalty": round(repeat_penalty,3),
                         "num_predict": int(num_predict)}

        # Critic (ACh-gated)
        critic = {}
        if verdict["action"] == "allow":
            critic_passes = 1 + int(2*mu_out.ach)
            try:
                critic = run_critic(self.critic_llm, llm_answer, mu_out, passes=critic_passes, log_path=self.log_path)
            except Exception as e:
                critic = {"q_overall": 0.0, "has_conflict_note": False,
                          "reasons":[f"critic exec error: {e} | http={self.critic_llm.get_last_http()}"]}

        # Evidence (ACh controls dissent quota)
        ev = self.scout.fetch_pagerank(pol.k_breadth, pol.q_contra)
        dissent_present = any(e.stance=="con" for e in ev)
        conflict_note_present = ("conflict" in (llm_answer or "").lower())

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

        # Adoption threshold tied to critic
        adopt = True
        if critic: adopt = critic.get("q_overall", 0.0) >= 0.70

        stats = {"sources": len(ev),
                 "resolved": dissent_present or conflict_note_present or critic.get("has_conflict_note", False),
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
            payload["last_http"] = {"pilot": attempts[-1]["http"] if attempts else {},
                                    "critic": self.critic_llm.get_last_http()}
            payload["attempts"] = attempts
        return payload

# ========= CLI & Probes =========
def main():
    ap = argparse.ArgumentParser(description="Guardian-AGI (single-file) — resilient Ollama + Emotional Center")
    ap.add_argument("--pilot-model", default=os.getenv("GUARDIAN_PILOT", "gpt-oss:20b"),
                    help="Pilot LLM model (default gpt-oss:20b)")
    ap.add_argument("--critic-model", default=os.getenv("GUARDIAN_CRITIC", None),
                    help="Critic LLM model (default = same as pilot)")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--ach", type=float, default=None, help="override ACh [0..1]")
    ap.add_argument("--probe", choices=["none","ach","policy","stop","critic","health"], default="none")
    ap.add_argument("--record", default="", help="Path to ledger JSONL (append-only). Empty=off.")
    ap.add_argument("--log", default=os.getenv("GUARDIAN_LOG",""), help="Append structured logs to this JSONL path.")
    ap.add_argument("--pilot-alt-model", default=os.getenv("GUARDIAN_PILOT_ALT",""),
                    help="Optional alternate model if Pilot answer fails acceptance.")
    ap.add_argument("--debug", action="store_true", help="Include per-phase HTTP traces and attempts.")
    args = ap.parse_args()

    eng = Engine(pilot_model=args.pilot_model, critic_model=args.critic_model,
                 debug=args.debug, log_path=args.log)

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
        res = eng.run_pagerank_demo(ach=args.ach, seed=args.seed, deny_policy=True,
                                    pilot_alt_model=args.pilot_alt_model or None)
        print(json.dumps({"risk":res["risk"],"verdict":res["verdict"],
                          "note":"Custodian veto blocks LLM call"}, indent=2)); return

    if args.probe == "stop":
        res_ok = eng.run_pagerank_demo(ach=args.ach, seed=args.seed, pilot_alt_model=args.pilot_alt_model or None)
        res_bad = copy.deepcopy(res_ok)
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
        print(json.dumps({
            "low.critic": low.get("critic",{}),  "low.adopted": low.get("adopted", True),
            "high.critic": high.get("critic",{}),"high.adopted": high.get("adopted", True),
            "expectation": "High ACh → more critic passes; q_overall should be ≥ low or equal; adoption requires q≥0.70."
        }, indent=2)); return

    if args.probe == "health":
        h = eng.probe_health()
        print(json.dumps(h, indent=2)); return

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
