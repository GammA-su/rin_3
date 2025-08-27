#!/usr/bin/env python3
# guardian_agi_min.py — Guardian-AGI (seed=137)
# Resilient Pilot + strict Critic with Ollama /api/chat and /api/generate fallbacks.
# - Detects/repairs truncation; enforces 110–160 words, ≥3 bracketed citations, and a resolution line.
# - Critic output is sanitized to a compact dict (no model/message blobs).
# - Debug traces preserved; critic raw kept internal only.

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
PAGERANK_PRIMARY = "PageRank is a link analysis algorithm assigning importance as the stationary probability a random surfer lands on a page. The damping factor (≈0.85) models continuing to click links."
PAGERANK_MEDIA   = "Popular media often say PageRank ranks pages by counting links; more links imply higher rank."
PAGERANK_DISSENT = "Dissent: PageRank is NOT simple counts; it weights by the rank of linking pages and normalizes by their outdegree."

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

# ========= Pilot (Intent) =========
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

# ========= JSON extractor =========
def extract_last_json(text: str) -> dict:
    if not text: raise ValueError("empty text")
    try:
        return json.loads(text)
    except Exception:
        pass
    depth = 0; start = None; last = None
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0: start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                cand = text[start:idx+1]
                try: last = json.loads(cand)
                except Exception: pass
    if last is not None: return last
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
        body = json.dumps(payload); attempt = 0; err: Optional[Exception] = None
        while attempt <= self.retries:
            t0 = now_ms(); status, raw = -1, ""
            conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
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
            keep_full = self.debug and str(phase).startswith("critic")
            self.last_http = {
                "phase": phase, "path": path, "status": status, "lat_ms": latency,
                "raw_preview": raw[:240], "raw_mid": raw[:2000], "raw_full": raw if keep_full else ""
            }
            if status == 200:
                try: return json.loads(raw) if raw else {}
                except Exception as e:
                    err = RuntimeError(f"Ollama parse error (HTTP {status}) at {path}: {e}; raw={raw[:200]}")
            if attempt == self.retries: break
            time.sleep(self.backoff * (2**attempt)); attempt += 1
        if err: raise err
        raise RuntimeError(f"Ollama HTTP {self.last_http.get('status')} at {self.last_http.get('path')}: {self.last_http.get('raw_preview')}")

    def _render_chat_as_prompt(self, system_msg: str, user_msg: str) -> str:
        return f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

    def _generate(self, prompt: str, opts: dict, phase: str, stop: Optional[List[str]]=None) -> str:
        payload = {"model": self.model, "prompt": prompt, "options": opts, "stream": False}
        if stop: payload["stop"] = stop
        gen = self._post("/api/generate", payload, phase)
        return (gen.get("response") or gen.get("thinking") or "").strip()

    def ask(self, system_msg: str, user_msg: str, *, temperature: float, top_p: float,
            repeat_penalty: float, num_predict: int, num_ctx: int=8192,
            force_json: bool=False, phase: str="pilot", allow_thinking: bool=False) -> str:
        opts = {"temperature": float(temperature), "top_p": float(top_p),
                "repeat_penalty": float(repeat_penalty), "num_predict": int(num_predict),
                "num_ctx": int(num_ctx)}
        if force_json: opts["format"] = "json"
        data = self._post("/api/chat", {
            "model": self.model,
            "messages": [{"role":"system","content":system_msg},
                         {"role":"user","content":user_msg}],
            "options": opts, "stream": False
        }, phase=phase)
        content = ""; thinking = ""
        if isinstance(data, dict):
            msg = data.get("message", {}) if isinstance(data.get("message"), dict) else {}
            content = (msg.get("content") or "").strip()
            thinking = (msg.get("thinking") or data.get("thinking") or "").strip()
        if content: return content
        if allow_thinking and thinking: return thinking
        # fallback to generate
        return self._generate(self._render_chat_as_prompt(system_msg, user_msg), opts, f"{phase}-fallback")

    def rewrite_from_notes(self, system_msg: str, user_msg: str, notes: str, *, phase: str,
                           temperature: float, top_p: float, repeat_penalty: float,
                           num_predict: int, num_ctx:int=8192, max_tries:int=3) -> str:
        stop_token = "### END"
        base_prompt = (
            f"{system_msg}\n\nUser:\n{user_msg}\n\n"
            "Assistant task: Use the NOTES to produce the FINAL ANSWER ONLY.\n"
            "Requirements: 110–160 words, ≥3 bracketed citations like [1][2][3], "
            "include either a 'Conflict Note:' or a 'Misconception:' line. "
            "Do NOT include steps or analysis. End with the literal token: ### END\n\n"
            f"NOTES:\n{notes}\n\nFINAL ANSWER:\n"
        )
        opts = {
            "temperature": float(max(0.1, temperature-0.05)),
            "top_p": float(max(0.5, top_p-0.1)),
            "repeat_penalty": float(repeat_penalty+0.05),
            "num_predict": int(max(900, num_predict, 700)),
            "num_ctx": int(max(8192, num_ctx))
        }
        out = ""
        for _ in range(max_tries):
            raw = self._generate(base_prompt, opts, f"{phase}-salvage", stop=[stop_token])
            out = raw.split(stop_token)[0].strip() if raw else ""
            if out: break
            opts["num_predict"] = int(opts["num_predict"] + 120)
            opts["temperature"] = max(0.1, opts["temperature"] - 0.02)
        if not out:
            raise RuntimeError("Salvage generate returned empty text.")
        return out

    def get_last_http(self) -> Dict[str,Any]: return copy.deepcopy(self.last_http)

# ========= Critic =========
CRITIC_SYS = (
  "You are Critic. Respond in STRICT JSON only.\n"
  "Schema: {\"q_overall\": float, \"has_conflict_note\": bool, \"reasons\": [string]}\n"
  "q_overall in [0,1]. No prose outside JSON."
)

def _heuristic_score(answer: str) -> dict:
    s = (answer or "").strip()
    if not s: return {"q_overall": 0.0, "has_conflict_note": False, "reasons": ["empty answer"]}
    words = len(re.findall(r"\b\w+\b", s))
    cites = len(set(re.findall(r"\[[0-9]+\]", s)))
    resolved = any(x in s.lower() for x in ["conflict note", "misconception:"])
    q = 0.0
    q += 0.35 if 110 <= words <= 160 else 0.0
    q += 0.35 if cites >= 3 else 0.0
    q += 0.30 if resolved else 0.0
    return {"q_overall": float(clamp(q,0.0,1.0)), "has_conflict_note": resolved,
            "reasons": [("length ok" if 110<=words<=160 else "length off"),
                        ("≥3 citations" if cites>=3 else "<3 citations"),
                        ("resolution line present" if resolved else "no resolution line")]}

def _sanitize_critic(d: dict) -> dict:
    out = {
        "q_overall": float(clamp(float(d.get("q_overall", 0.0)), 0.0, 1.0)),
        "has_conflict_note": bool(d.get("has_conflict_note", False)),
        "reasons": [str(r)[:160] for r in (d.get("reasons", []) or [])][:6],
    }
    return out

def run_critic(llm: LLMClient, answer: str, passes: int=1, temperature: float=0.2, log_path:str="") -> dict:
    top_p=0.7; repeat_penalty=1.15; num_predict=220
    best=None; any_json=False
    for i in range(max(1,passes)):
        try:
            j = llm.ask(CRITIC_SYS, f"Answer:\n{answer}", temperature=temperature, top_p=top_p,
                        repeat_penalty=repeat_penalty, num_predict=num_predict,
                        force_json=True, phase="critic", allow_thinking=False)
            data = extract_last_json(j); any_json=True
            if (best is None) or (data.get("q_overall",0.0)>best.get("q_overall",0.0)): best=data
            log_jsonl(log_path, {"phase":"critic","pass":i,"raw":str(j)[:240]})
        except Exception as e:
            h = llm.get_last_http()
            blob = h.get("raw_full") or h.get("raw_mid") or h.get("raw_preview","")
            try:
                salvage = extract_last_json(blob); any_json=True
                if (best is None) or (salvage.get("q_overall",0.0) > (best.get("q_overall",0.0) if isinstance(best,dict) else 0.0)):
                    best = salvage
                log_jsonl(log_path, {"phase":"critic","pass":i,"salvaged": True})
            except Exception:
                log_jsonl(log_path, {"phase":"critic","pass":i,"error": str(e)})
    if not any_json:
        best = _heuristic_score(answer)
        best["reasons"].insert(0, "heuristic fallback (no JSON from critic)")
    return _sanitize_critic(best or {})

# ========= Acceptance & Repair =========
def word_count(s: str) -> int: return len(re.findall(r"\b\w+\b", s))

def balanced_brackets(s: str) -> bool:
    # require equal counts for both [] and ()
    return (s.count("[") == s.count("]")) and (s.count("(") == s.count(")"))

def ends_with_sentence(s: str) -> bool:
    s = s.rstrip()
    return bool(re.search(r"[.!?](?:['\")\]]+)?$", s))

def seems_truncated(s: str) -> bool:
    if not s: return True
    wc = word_count(s)
    broken = not ends_with_sentence(s)
    unbalanced = not balanced_brackets(s)
    return (wc < 90) or broken or unbalanced

def acceptable_answer(txt: str) -> bool:
    if not txt: return False
    s = txt.strip()
    wc = word_count(s)
    if not (110 <= wc <= 160): return False
    cites = len(set(re.findall(r"\[[0-9]+\]", s)))
    if cites < 3: return False
    if ("conflict note" not in s.lower()) and ("misconception:" not in s.lower()): return False
    if re.search(r"\bwe need to\b|\blet'?s\b|step by step|first, we|i will\b", s.lower()): return False
    return ends_with_sentence(s) and balanced_brackets(s)

# ========= Engine =========
@dataclass
class IntentObj:  # alias (not used elsewhere but kept for clarity)
    assumptions: str; unknowns: str; tests: str; stop: str; risk: str

class Engine:
    def __init__(self, pilot_model: str="gpt-oss:20b", critic_model: Optional[str]=None,
                 critic_temp: float=0.2, neuro: Mu=None, debug: bool=False, log_path:str=""):
        self.homeo = Homeostat(); self.cust  = Custodian(); self.wit   = Witness()
        self.scout = Scout();     self.arch  = Archivist(); self.pilot = Pilot()
        self.pilot_llm  = LLMClient(pilot_model, debug=debug)
        self.critic_llm = LLMClient(critic_model or pilot_model, debug=debug)
        self.critic_temp = critic_temp
        self.neuro0 = neuro or Mu(da=0.50, ne=0.55, s5ht=0.85, ach=0.75, gaba=0.35, oxt=0.70)
        self.debug = debug; self.log_path = log_path

    def probe_health(self) -> Dict[str,Any]:
        ok = True; notes = []
        try:
            data = self.pilot_llm._post("/api/tags", {}, phase="health-tags")
            notes.append({"tags": list(data.get("models",[]))[:5]})
        except Exception as e:
            ok = False; notes.append({"tags_error": str(e), "http": self.pilot_llm.get_last_http()})
        try:
            data = self.pilot_llm._post("/api/ps", {}, phase="health-ps")
            notes.append({"ps": data})
        except Exception as e:
            ok = False; notes.append({"ps_error": str(e), "http": self.pilot_llm.get_last_http()})
        try:
            resp = self.pilot_llm._post("/api/generate", {
                "model": self.pilot_llm.model, "prompt": "{}", "options": {"format":"json"}, "stream": False
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

        llm_answer = "[blocked by policy]"; llm_knobs = {}; attempts: List[Dict[str,Any]] = []

        if verdict["action"] == "allow":
            temperature    = max(0.1, 0.9 - 0.6*mu_out.s5ht)
            top_p          = clamp(0.75 + 0.20*mu_out.ne - 0.10*mu_out.gaba, 0.50, 0.95)
            repeat_penalty = 1.05 + 0.20*mu_out.s5ht - 0.10*mu_out.da
            num_predict    = int(256 + int(384*mu_out.s5ht) - int(128*mu_out.gaba))

            system_msg = (f"You are Pilot (seed={SEED_DEFAULT}). Think privately but DO NOT reveal chain-of-thought. "
                          "Final answer only: 110–160 words, ≥3 bracketed citations. Include either 'Conflict Note:' or 'Misconception:' line.")
            user_msg = ("Explain PageRank in ≤150 words with ≥3 citations. Prioritize primary/official. "
                        "Resolve the link-count misconception.")

            # Attempt 1 (chat)
            try:
                ans = self.pilot_llm.ask(system_msg, user_msg,
                                         temperature=temperature, top_p=top_p,
                                         repeat_penalty=repeat_penalty, num_predict=num_predict,
                                         phase="pilot", allow_thinking=False)
                attempts.append({"kind":"pilot","ok":True,"http":self.pilot_llm.get_last_http(),"len":len(ans)})
            except Exception as e:
                attempts.append({"kind":"pilot","ok":False,"error":str(e),"http":self.pilot_llm.get_last_http()})
                ans = ""

            # Salvage from notes if we got no content
            if (not ans):
                h = self.pilot_llm.get_last_http(); notes = ""
                for k in ("raw_full", "raw_mid", "raw_preview"):
                    notes = notes or h.get(k,"")
                if notes:
                    try:
                        ans = self.pilot_llm.rewrite_from_notes(system_msg, user_msg, notes,
                            phase="pilot-repair", temperature=temperature, top_p=top_p,
                            repeat_penalty=repeat_penalty, num_predict=num_predict)
                        attempts.append({"kind":"pilot-repair-salvage","ok":True,"http":self.pilot_llm.get_last_http(),"len":len(ans)})
                    except Exception as e:
                        attempts.append({"kind":"pilot-repair-salvage","ok":False,"error":str(e),"http":self.pilot_llm.get_last_http()})

            # Repair if truncated/off-spec (pass 1)
            if ans and (seems_truncated(ans) or not acceptable_answer(ans)):
                try:
                    ans = self.pilot_llm.rewrite_from_notes(
                        system_msg, user_msg,
                        f"DRAFT (incomplete or off-spec):\n{ans}\n\nREPAIR: polish to 110–160 words, ensure ≥3 bracketed citations and include a resolution line.",
                        phase="pilot-repair-1",
                        temperature=temperature, top_p=top_p,
                        repeat_penalty=repeat_penalty, num_predict=num_predict
                    )
                    attempts.append({"kind":"pilot-repair-1","ok":True,"http":self.pilot_llm.get_last_http(),"len":len(ans)})
                except Exception as e:
                    attempts.append({"kind":"pilot-repair-1","ok":False,"error":str(e),"http":self.pilot_llm.get_last_http()})

            # Acceptance loop (up to two retries with rewrite-based repair)
            tries = 0
            while not acceptable_answer(ans) and tries < 2:
                tries += 1
                remsg = (user_msg + "\n\nCRITICAL: Output 110–160 words, ≥3 bracketed citations [1][2][3], "
                         "include exactly one 'Misconception:' or 'Conflict Note:' line. Final answer only.")
                try:
                    ans2 = self.pilot_llm.rewrite_from_notes(
                        system_msg, remsg,
                        f"NOTES (why previous failed):\n- {'too short' if word_count(ans)<110 else 'too long' if word_count(ans)>160 else 'bad ending or brackets'}\n- ensure [1][2][3]\n- include resolution line\n\nDRAFT:\n{ans}",
                        phase=f"pilot-reprompt-repair-{tries}",
                        temperature=max(0.1, temperature-0.1*tries),
                        top_p=max(0.5, top_p-0.1*tries),
                        repeat_penalty=repeat_penalty+0.05*tries,
                        num_predict=max(num_predict, 900+100*tries)
                    )
                    attempts.append({"kind":f"pilot-reprompt-repair-{tries}","ok":True,"http":self.pilot_llm.get_last_http(),"len":len(ans2)})
                    ans = ans2
                except Exception as e:
                    attempts.append({"kind":f"pilot-reprompt-repair-{tries}","ok":False,"error":str(e),"http":self.pilot_llm.get_last_http()})
                    break

            # Optional alt model path
            if not acceptable_answer(ans) and pilot_alt_model:
                alt_llm = LLMClient(pilot_alt_model, debug=self.debug)
                try:
                    ans3 = alt_llm.ask(system_msg, user_msg, temperature=0.2, top_p=0.8,
                                       repeat_penalty=1.15, num_predict=max(num_predict,640),
                                       phase="pilot-alt", allow_thinking=False)
                    if seems_truncated(ans3) or not acceptable_answer(ans3):
                        ans3 = alt_llm.rewrite_from_notes(system_msg, user_msg,
                                f"DRAFT:\n{ans3}\n\nREPAIR to spec (110–160 words, ≥3 bracketed citations, include resolution line).",
                                phase="pilot-alt-repair",
                                temperature=0.2, top_p=0.7, repeat_penalty=1.15, num_predict=900)
                    attempts.append({"kind":"pilot-alt","model":pilot_alt_model,"ok":True,"http":alt_llm.get_last_http(),"len":len(ans3)})
                    ans = ans3
                except Exception as e:
                    attempts.append({"kind":"pilot-alt","model":pilot_alt_model,"ok":False,"error":str(e),"http":alt_llm.get_last_http()})

            llm_answer = ans
            llm_knobs = {"temperature": round(temperature,3),
                         "top_p": round(top_p,3),
                         "repeat_penalty": round(repeat_penalty,3),
                         "num_predict": int(num_predict)}

        # Critic (compact dict only)
        critic = {}
        if verdict["action"] == "allow":
            critic_passes = 1 + int(2*mu_out.ach)
            try:
                critic = run_critic(self.critic_llm, llm_answer, passes=critic_passes,
                                    temperature=self.critic_temp, log_path=self.log_path)
            except Exception as e:
                critic = {"q_overall": 0.0, "has_conflict_note": False,
                          "reasons":[f"critic exec error: {e}"]}

        # Evidence & claims
        ev = self.scout.fetch_pagerank(pol.k_breadth, pol.q_contra)
        resolution_line_present = any(x in (llm_answer or "").lower() for x in ["conflict note", "misconception:"])

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

        adopt = critic.get("q_overall", 0.0) >= 0.70 if critic else False
        stats = {"sources": len(ev),
                 "resolved": resolution_line_present or critic.get("has_conflict_note", False),
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
                "pilot": (attempts[-1]["http"] if attempts else self.pilot_llm.get_last_http()),
                "critic": self.critic_llm.get_last_http()
            }
            payload["attempts"] = attempts
        return payload

# ========= CLI =========
def main():
    ap = argparse.ArgumentParser(description="Guardian-AGI — resilient Ollama + Emotional Center")
    ap.add_argument("--pilot-model", default=os.getenv("GUARDIAN_PILOT", "gpt-oss:20b"))
    ap.add_argument("--critic-model", default=os.getenv("GUARDIAN_CRITIC", None))
    ap.add_argument("--critic-temp", type=float, default=float(os.getenv("GUARDIAN_CRITIC_TEMP", "0.2")))
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--ach", type=float, default=None)
    ap.add_argument("--probe", choices=["none","ach","policy","stop","critic","health"], default="none")
    ap.add_argument("--record", default="", help="Path to ledger JSONL (append-only). Empty=off.")
    ap.add_argument("--log", default=os.getenv("GUARDIAN_LOG",""))
    ap.add_argument("--pilot-alt-model", default=os.getenv("GUARDIAN_PILOT_ALT",""))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    eng = Engine(pilot_model=args.pilot_model, critic_model=args.critic_model,
                 critic_temp=args.critic_temp, debug=args.debug, log_path=args.log)

    if args.probe == "health":
        print(json.dumps(eng.probe_health(), indent=2)); return
    if args.probe == "ach":
        low  = eng.run_pagerank_demo(ach=0.2, seed=args.seed, pilot_alt_model=args.pilot_alt_model or None)
        high = eng.run_pagerank_demo(ach=0.9, seed=args.seed, pilot_alt_model=args.pilot_alt_model or None)
        out = {"low.policy":low["policy"],"high.policy":high["policy"],
               "delta_k_breadth":high["policy"]["k_breadth"]-low["policy"]["k_breadth"],
               "delta_q_contra":high["policy"]["q_contra"]-low["policy"]["q_contra"]}
        print(json.dumps(out, indent=2)); return
    if args.probe == "policy":
        res = eng.run_pagerank_demo(ach=args.ach, seed=args.seed, deny_policy=True, pilot_alt_model=args.pilot_alt_model or None)
        print(json.dumps({"risk":res["risk"],"verdict":res["verdict"],"note":"Custodian veto blocks LLM call"}, indent=2)); return
    if args.probe == "stop":
        res_ok = eng.run_pagerank_demo(ach=args.ach, seed=args.seed, pilot_alt_model=args.pilot_alt_model or None)
        res_bad = copy.deepcopy(res_ok); res_bad["kpis"] = {**res_bad["kpis"], "pass_at_1": 0.0}
        res_bad["stop_score"] = soft_stop(0.0, res_ok["mu_out"]["gaba"], 0.2, 0.2)
        print(json.dumps({"stop_when_goal_met": res_ok["stop_score"], "stop_when_unmet": res_bad["stop_score"],
                          "expectation":"unmet < met and typically < 0.6 threshold"}, indent=2)); return
    if args.probe == "critic":
        low  = eng.run_pagerank_demo(ach=0.2, seed=args.seed, pilot_alt_model=args.pilot_alt_model or None)
        high = eng.run_pagerank_demo(ach=0.9, seed=args.seed, pilot_alt_model=args.pilot_alt_model or None)
        print(json.dumps({"low.critic": low.get("critic",{}), "high.critic": high.get("critic",{})}, indent=2)); return

    res = eng.run_pagerank_demo(ach=args.ach, seed=args.seed, pilot_alt_model=args.pilot_alt_model or None)
    print(json.dumps(res, indent=2))
    if args.record:
        ledger_append(args.record, {
            "goal": res["goal"], "risk": res["risk"], "verdict": res["verdict"],
            "mu_out": res["mu_out"], "policy": res["policy"], "llm_knobs": res["llm_knobs"],
            "critic": res.get("critic", {}), "adopted": res.get("adopted", False),
            "kpis": res["kpis"], "stop_score": res["stop_score"]
        })

if __name__ == "__main__":
    main()
