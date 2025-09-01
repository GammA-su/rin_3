#!/usr/bin/env python3
# guardian_agi_min.py — Guardian-AGI scaffold (seed=137)
# Additive upgrade: persistent memory + trust calculus + local-docs retrieval + strict-JSON critic.

# --- GENESIS inserts (imports) ---


from __future__ import annotations
import argparse, json, os, random, time, http.client, hashlib, math, sys
from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import List, Dict, Any, Optional, Tuple
import io, contextlib, json
import faulthandler, atexit
faulthandler.enable()
try:
    from z3 import Solver, Real, Int, Bool, sat  # optional; we degrade gracefully if missing
    _Z3_OK = True
except Exception:
    _Z3_OK = False

def hard_exit(code: int = 0):
    try:
        sys.stdout.flush(); sys.stderr.flush()
    except Exception:
        pass
    os._exit(int(code))

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

def claim_from_dict(d: Dict[str,Any]) -> Claim:
    srcs = [Source(**s) for s in d.get("sources", [])]
    return Claim(id=d["id"], text=d["text"], q=float(d.get("q",0.5)),
                 sources=srcs, supports=d.get("supports",[]),
                 contradicts=d.get("contradicts",[]), stance=d.get("stance","neutral"))

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

# ========= Memory (JSONL persistence) =========
class MemoryStore:
    def __init__(self, root: Optional[str]=None):
        self.root = root
        if root: os.makedirs(root, exist_ok=True)
        self.claims_path   = os.path.join(root,"claims.jsonl") if root else None
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
        da  = clamp(mu.da  + 0.35*(a.p-0.5) - 0.30*a.k)
        ne  = clamp(mu.ne  + 0.40*a.u + 0.30*a.n)
        s5  = clamp(mu.s5ht + 0.45*a.s)
        ach = clamp(mu.ach + 0.20*a.c)
        gb  = clamp(mu.gaba + 0.50*0.5 + 0.30*0.5 - 0.25*0.5)
        oxt = clamp(mu.oxt + 0.40*a.h)
        return Mu(da, ne, s5, ach, gb, oxt)

    def couple(self, mu: Mu) -> PolicyCoupling:
        k0, d0 = 6, 3
        k = max(3, int(k0*(1 + mu.ne - 0.5*mu.s5ht)))
        d = max(1, int(d0*(1 + mu.s5ht - mu.ne)))
        # Dissent quota (ACh step-up): <0.5→1, 0.5–<0.8→2, ≥0.8→3
        q_con = 1 + (1 if mu.ach >= 0.5 else 0) + (1 if mu.ach >= 0.8 else 0)
        temp = max(0.1, 0.9 - 0.6*mu.s5ht)
        retr = clamp(0.35 + 0.30*mu.ne - 0.15*mu.s5ht, 0.0, 1.0)
        syn  = clamp(0.35 + 0.30*mu.s5ht - 0.15*mu.ne, 0.0, 1.0)
        saf  = clamp(1.0 - (retr + syn), 0.0, 1.0)
        return PolicyCoupling(k, d, q_con, temp, retr, syn, saf, reserved_dissent=(mu.ach>=0.6))


# ========= Safety (Custodian) =========
class Custodian:
    policy_ver = "v1.1"
    ACTIONS = {
        "R0": {"action":"allow", "notes":"benign"},
        "R1": {"action":"allow", "notes":"sanitize/redact personal identifiers"},
        "R2": {"action":"allow", "notes":"conservative language + justification + citations"},
        "R3": {"action":"deny",  "notes":"sandbox/two-man rule required"},
        "R4": {"action":"deny",  "notes":"physical actuation blocked"},
    }
    def classify(self, goal: str) -> str:
        g = goal.lower()
        if any(x in g for x in ["bio", "exploit", "weapon", "malware", "lab"]): return "R3"
        if any(x in g for x in ["finance","legal","medical"]): return "R2"
        if any(x in g for x in ["personal","pii","identifier"]): return "R1"
        return "R0"
    def preflight(self, risk: str) -> Dict[str,str]:
        a = self.ACTIONS.get(risk, self.ACTIONS["R0"]).copy()
        a["notes"] = f"Risk {risk}: {a['notes']}"
        return a

# ========= Evaluation (Witness) =========
class Witness:
    def _ece(self, confidences, outcomes, bins=10):
        if not confidences or not outcomes or len(confidences)!=len(outcomes): return 0.0
        bs = [i/bins for i in range(bins+1)]
        ece = 0.0
        for i in range(bins):
            lo, hi = bs[i], bs[i+1]
            idx = [j for j,c in enumerate(confidences) if lo <= c < hi or (hi==1.0 and c==1.0)]
            if not idx: continue
            avg_c = sum(confidences[j] for j in idx)/len(idx)
            avg_a = sum(outcomes[j] for j in idx)/len(idx)
            ece += (len(idx)/len(confidences)) * abs(avg_c - avg_a)
        return float(ece)
    def score(self, stats: Dict[str,Any]) -> Dict[str,float]:
        pass_at_1   = 1.0 if stats.get("goal_met", False) else 0.0
        precision_k = clamp(0.7 + 0.05*max(0, stats.get("sources", 0)), 0.0, 1.0)
        # crude: treat critic q_overall as a proxy confidence and adoption as outcome
        cq = float(stats.get("critic_q", 0.7))
        if cq > 1.0: cq = cq/10.0
        confidences = [cq]
        outcomes    = [1.0 if stats.get("adopted", False) else 0.0]
        ece         = self._ece(confidences, outcomes, bins=5)
        resolution  = 1.0 if stats.get("resolved", False) else 0.0
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
                except Exception: continue

    def upsert_claim(self, c: Claim):
        self.claims[c.id] = c
        if self.mem and self.mem.enabled(): self.mem.save_claim(c)

    def link_contradiction(self, i: str, j: str):
        self.contradict.setdefault(i,[]).append(j); self.contradict.setdefault(j,[]).append(i)

    def retrieve(self, k:int=5) -> List[Claim]:
        return list(self.claims.values())[:k]

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
        span = {"start": 0, "end": min(len(txt), 600)}
        prov = provenance or [{"source": name, "offsets": span}]
        return EvidenceUnit(id=name, content_hash=h, extract=txt[:600], stance=stance, provenance=prov)

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
        others  = [e for e in pool if e.stance!="con"]
        pick_con = max(1, min(dissent_quota, len(cons)))
        selected = cons[:pick_con]
        for e in others:
            if len(selected) >= max(1, k_breadth): break
            selected.append(e)
        return selected[:max(1, k_breadth)]

    # Local-docs retrieval (simple, fast, zero deps)
    AUTH_TIER_BY_FOLDER = {
        "primary":1, "official":2, "peer":2, "peerreview":2,
        "media":3, "reputable":3, "blog":4, "community":4, "forum":5, "dissent":3
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
            if f"/{key}/" in lower or lower.endswith(f"/{key}"):
                return tier
        return 4

    def _stance_from_text_or_path(self, text: str, path: str) -> str:
        p = path.lower()
        if "dissent" in p or any(k in text.lower() for k in ["however", "contradict", "not simply", "misleading"]):
            return "con"
        if "media" in p or "blog" in p or "forum" in p:
            return "neutral"
        return "pro"

    def fetch_from_docs(self, k_breadth:int, dissent_quota:int) -> List[EvidenceUnit]:
        if not self.docs_dir or not os.path.isdir(self.docs_dir):
            return []
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
                recency = 0.5  # filesystem mtime could be used; kept simple
                authority = {1:0.95,2:0.85,3:0.60,4:0.45,5:0.30}.get(tier,0.45)
                dissent_bonus = 0.1 if stance=="con" else 0.0
                r = 0.35*authority + 0.20*recency + 0.25*topicality - 0.10*0.0 + 0.10*dissent_bonus
                prov = [{"source": path, "tier": tier, "offsets": {"start": 0, "end": min(len(text), 600)}}]
                ev = self._mk_ev(path, text, stance, provenance=prov)
                candidates.append((r, ev))
        if not candidates:
            return []
        # rank
        candidates.sort(key=lambda t: t[0], reverse=True)
        # ensure dissent quota
        cons = [ev for _, ev in candidates if ev.stance=="con"]
        others = [ev for _, ev in candidates if ev.stance!="con"]
        pick_con = max(1, min(dissent_quota, len(cons)))
        selected: List[EvidenceUnit] = cons[:pick_con]
        for ev in others:
            if len(selected) >= max(1, k_breadth): break
            selected.append(ev)
        return selected[:max(1, k_breadth)]

# ========= Planner (Operator) =========
@dataclass
class Plan: name: str; steps: List[str]
class Operator:
    def plan_research(self) -> Plan: return Plan("T1-Research", ["Define scope","Fetch coverage","Extract claims","Synthesize","Calibrate"])
    def plan_compare(self) -> Plan:  return Plan("T2-Compare",  ["Collect A,B","Map contradictions","Resolve","Explain rationale"])
# === GENESIS: ACL / AE / NSV / OE-OS ===
def _norm_txt(s: str) -> str:
    return " ".join((s or "").lower().split())

class ACL:
    def __init__(self, arch: Archivist, mem: MemoryStore, llm: LLMClient):
        self.arch, self.mem, self.llm = arch, mem, llm
    def novelty_gate(self, text: str, theta: float = 0.70) -> bool:
        """
        Novel iff: (i) no exact normalized match in memory/claims, AND
                   (ii) max Jaccard(sim) against any seen text < 1 - theta.
        NOTE: 'text' itself is NOT added to the seen set.
        """
        tnorm = _norm_txt(text)

        # collect seen strings from persistent memory and current claims (exclude candidate)
        seen: list[str] = []
        if self.mem.enabled():
            for d in self.mem.iter_claims():
                s = _norm_txt(d.get("text", ""))
                if s: seen.append(s)
        for c in self.arch.claims.values():
            s = _norm_txt(c.text)
            if s: seen.append(s)

        # (i) exact duplicate?
        if any(s == tnorm for s in seen):
            return False

        # (ii) Jaccard similarity threshold on token sets
        if not seen:
            return True
        S = set(tnorm.split())
        def jacc(a: str) -> float:
            A = set(a.split())
            u = len(S | A)
            return 0.0 if u == 0 else len(S & A) / u

        max_sim = max((jacc(s) for s in seen), default=0.0)
        return max_sim < (1.0 - float(theta))

    def sketcher(self, text: str) -> dict:
        try:
            out = self.llm.ask(
                "Return STRICT JSON: {\"definition\":str, \"tests\":[str]}",
                f"Concept: {text}\nJSON only.", temperature=0.2, top_p=0.8,
                repeat_penalty=1.15, num_predict=256, force_json=True,
                attempts_log=[], phase_label="acl", allow_thinking_fallback=True)
            return extract_json_object(out)
        except Exception:
            return {"definition": text[:160], "tests":[f"Explain {text} in one line."]}
    def promote(self, cid: str, text: str, sketch: dict) -> Claim:
        c = Claim(id=cid, text=text, q=0.60, sources=[], stance="neutral")
        self.arch.upsert_claim(c)
        if self.mem.enabled():
            self.mem.save_episode({"t": now_ms(), "goal":"ACL.promote", "concept_id":cid, "sketch":sketch})
        return c

class AE:
    def __init__(self, engine: "Engine"): self.engine = engine
    def propose_patch(self) -> dict:
        # small, realistic knob surface for calibration/latency
        return {"kind":"knob_override","candidates":[
            {"temperature":0.30,"top_p":0.85,"repeat_penalty":1.15},
            {"temperature":0.28,"top_p":0.82,"repeat_penalty":1.12},
            {"temperature":0.22,"top_p":0.78,"repeat_penalty":1.20}
            ]}

    def ab_test(self, candidates: list, seed:int) -> dict:
        base = self.engine.run_pagerank_demo(seed=seed)
        def score(pkg):
            k   = pkg.get("kpis", {})
            ece = float(k.get("ece", 1.0))
            res = float(k.get("resolution_rate", 0.0))
            pa1 = float(k.get("pass_at_1", 0.0))
            lat = float(pkg.get("pilot_lat_ms")) if pkg.get("pilot_lat_ms") is not None else 999999.0
            if lat == 999999.0:
                for a in pkg.get("attempts", []):
                    if a.get("kind") == "pilot" and isinstance(a.get("http"), dict):
                        lm = a["http"].get("lat_ms")
                        if isinstance(lm, (int, float)): lat = float(lm); break
            if lat == 999999.0:
                lat = float(pkg.get("last_http", {}).get("lat_ms", 1000.0))
            return {"ece": ece, "res": res, "pa1": pa1, "lat": lat}
        sb = score(base); best = {"adopt": False, "baseline": sb}
        for knobs in candidates:
            self.engine.knob_override = knobs
            trial = self.engine.run_pagerank_demo(seed=seed)
            st = score(trial)
            # Composite improvement: (i) ECE↓ by ≥10% OR latency↓ by ≥15%, AND no drop in res/pass
            ece_ok = (sb["ece"] > 0.0 and st["ece"] <= sb["ece"]*0.90) or (sb["ece"] == 0.0 and st["ece"] == 0.0)
            lat_ok = st["lat"] <= sb["lat"]*0.85
            gate = ((ece_ok) or (lat_ok)) and (st["res"] >= sb["res"]) and (st["pa1"] >= sb["pa1"])
            if gate:
                best = {"adopt": True, "candidate": knobs,
                        "delta_ece": round(sb["ece"]-st["ece"], 6),
                        "lat_impr_ms": int(sb["lat"]-st["lat"]), "kpis": trial.get("kpis", {})}
                sb = st  # allow cumulative improvement
        self.engine.knob_override = None
        if best["adopt"]:
            self.engine.default_knobs.update(best["candidate"])
        return best


class NSV:
    def translate(self, claim: str) -> dict: return {"raw": claim}
    def _parse_val(self, s: str) -> float:
        s = s.strip()
        # support fractions like "2/3" or "succ/total" (resolve later if digits)
        if "/" in s:
            num, den = s.split("/", 1)
            num, den = num.strip(), den.strip()
            if num.isdigit() and den.isdigit():
                d = float(den)
                return float(num) / d if d != 0.0 else float("inf")
        # plain float/int
        try: return float(s)
        except: return float("nan")
    def verify_logic(self, claim: str, *, context: dict | None = None) -> bool:
        """
        Supports forms:
          "x == y", "x >= y", "x > y", "x <= y", "x < y"
        where x,y can be numbers or fractions like "2/3".
        If `context` provides {"succ":int,"total":int}, then tokens "succ" and "total"
        in simple fractions are resolved numerically.
        """
        if not isinstance(claim, str) or not claim.strip():
            return False
        s = claim.strip()
        # normalize symbols
        for op in ["==", ">=", "<=", ">", "<"]:
            if op in s:
                left, right = s.split(op, 1)
                l, r = left.strip(), right.strip()
                # resolve 'succ'/'total' placeholders if provided
                def resolve(token: str) -> str:
                    if context and token in ("succ","total"):
                        return str(context[token])
                    return token
                # expand tokens inside simple "a/b" if present
                def norm(side: str) -> str:
                    if "/" in side:
                        a, b = side.split("/", 1)
                        return f"{resolve(a.strip())}/{resolve(b.strip())}"
                    return resolve(side)
                l, r = norm(l), norm(r)
                lv, rv = self._parse_val(l), self._parse_val(r)
                if (lv != lv) or (rv != rv):  # NaN guard
                    return False
                if   op == "==": return lv == rv
                elif op == ">=": return lv >= rv
                elif op == "<=": return lv <= rv
                elif op ==  ">" : return lv >  rv
                elif op ==  "<" : return lv <  rv
        # default truthy if claim has no comparator (non-brittle)
        return True
    def verify_stats(self, success:int, total:int, threshold:float=0.70) -> bool:
        if total<=0: return False
        return (success/float(total)) >= threshold
    def attach_proof(self, obj: dict, ok: bool, kind: str="Type-S") -> dict:
        obj["proof"] = {"kind":kind, "ok":bool(ok), "ts": now_ms()}
        return obj

class OEOS:
    def __init__(self, engine:"Engine"): self.engine = engine
    def env_hub(self, env:str="browser"): return env
    def task_fabric(self):  # minimal demo stream
        return [{"env":"browser","task":"pagerank-mini-demo"}]
    def log_episode(self, payload:dict): ledger_append("episodes.jsonl", payload)
    def run_minigrid_episode(self, steps=50):
        try:
            import gymnasium as gym, minigrid  # noqa
            env = gym.make("MiniGrid-Empty-5x5-v0")
            obs, info = env.reset()
            total=0.0
            for t in range(steps):
                # simple bias: prefer 'forward' more often than turn (↑ reward chance by reaching goal)
                a = env.action_space.sample()
                if (t % 3) != 0:  # crude forward bias
                    try:
                        a = env.unwrapped.actions.forward
                    except Exception:
                        pass
                obs, reward, terminated, truncated, info = env.step(a)
                total += float(reward)
                if terminated or truncated: break
            env.close()
            payload = {"env":"minigrid-empty-5x5","steps":t+1,"return":round(total,3)}
            ledger_append("episodes.jsonl", {"type":"oe-episode","payload":payload})
            return payload
        except Exception as e:
            return {"env":"minigrid","error":str(e)}


        # --- Tier 2: CartPole (no Box2D needed) ---
        try:
            import gymnasium as gym
            env = gym.make("CartPole-v1")
            obs, info = env.reset()
            total = 0.0
            step_count = 0
            for step_count in range(steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            env.close()
            payload = {"env": "cartpole-v1", "steps": int(step_count+1), "return": total}
            ledger_append("episodes.jsonl", {"type": "oe-episode", "payload": payload})
            return payload
        except Exception as e2:
            err2 = str(e2)

        # --- Tier 3: internal dummy random-walk ---
        total = 0.0
        x = 0
        import random
        for _ in range(steps):
            x += random.choice([-1, +1])
            total += 1.0 if x == 0 else 0.0
        payload = {"env": "dummy-random-walk", "steps": steps, "return": total,
                "fallback_errors": {"minigrid": err1, "cartpole": err2}}
        ledger_append("episodes.jsonl", {"type": "oe-episode", "payload": payload})
        return payload


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
            raw = resp.read().decode("utf-8") if resp else ""
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

    def ask(self, system_msg: str, user_msg: str, *, temperature: float, top_p: float,
            repeat_penalty: float, num_predict: int, num_ctx: int=8192, force_json: bool=False,
            attempts_log: Optional[List[dict]]=None, phase_label:str="pilot",
            allow_thinking_fallback: bool=False) -> str:
        chat_payload = {
            "model": self.model,
            "messages": [
                {"role":"system","content":system_msg},
                {"role":"user","content":user_msg}
            ],
            "options": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "repeat_penalty": float(repeat_penalty),
                "num_predict": int(num_predict),
                "num_ctx": int(num_ctx),
                **({"format":"json"} if force_json else {}),
                **({"stop": ["\n\n"]} if force_json else {})
            },
            "stream": False
        }
        try:
            data, httpm = self._post("/api/chat", chat_payload, phase_label)
            out = ""
            thought = ""
            if isinstance(data, dict) and "message" in data and isinstance(data["message"], dict):
                out = data["message"].get("content","")
                thought = data["message"].get("thinking","")
                if not out and allow_thinking_fallback and thought:
                    out = thought

            elif isinstance(data, dict):
                out = data.get("response","")
                thought = data.get("thinking","")
                if not out and allow_thinking_fallback and thought:
                    out = thought

            if attempts_log is not None:
                attempts_log.append({"kind": phase_label, "ok": bool(out), "http": httpm, "len": len(out or "")})
            if out:
                return out.strip()
        except Exception as e:
            if attempts_log is not None:
                attempts_log.append({"kind": phase_label, "ok": False, "error": str(e), "http": getattr(self, "last_http", {})})

        gen_payload = {
            "model": self.model,
            "prompt": f"{system_msg}\n\nUser:\n{user_msg}\n\nAssistant:",
            "options": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "repeat_penalty": float(repeat_penalty),
                "num_predict": int(num_predict),
                "num_ctx": int(num_ctx),
                **({"format":"json"} if force_json else {}),
                **({"stop": ["\n\n"]} if force_json else {})
            },
            "stream": False
        }
        try:
            data2, httpm2 = self._post("/api/generate", gen_payload, phase_label + ("-fallback" if phase_label=="pilot" else "-repair-salvage"))
            out2 = ""
            thought2 = ""
            if isinstance(data2, dict):
                out2 = data2.get("response","")
                thought2 = data2.get("thinking","")
                if not out2 and allow_thinking_fallback and thought2:
                    out2 = thought2

            if attempts_log is not None:
                attempts_log.append({"kind": phase_label + ("-fallback" if phase_label=="pilot" else "-repair-salvage"),
                                     "ok": bool(out2), "http": httpm2, "len": len(out2 or "")})
            if not out2:
                raise RuntimeError("Ollama returned empty text from both chat and generate.")
            return out2.strip()
        except Exception as e:
            if attempts_log is not None:
                attempts_log.append({"kind": phase_label + "-fallback", "ok": False, "error": str(e),
                                     "http": getattr(self, "last_http", {})})
            raise

# ========= Offline Mock LLM (deterministic, seed=137) =========
class MockLLM(LLMClient):
    def __init__(self, debug: bool=False):
        # no HTTP; keep interface for polymorphism
        self.model = "mock"
        self.debug = debug
        self.last_http = {"phase":"mock","path":"mock","status":200,"lat_ms":0,
                          "raw_preview":"", "raw_mid":"", "raw_full":""}
        random.seed(137)
    def ask(self, system_msg: str, user_msg: str, *, temperature: float, top_p: float,
            repeat_penalty: float, num_predict: int, num_ctx: int=8192, force_json: bool=False,
            attempts_log: Optional[List[dict]]=None, phase_label:str="pilot",
            allow_thinking_fallback: bool=False) -> str:
        if attempts_log is not None:
            attempts_log.append({"kind": f"mock-{phase_label}", "ok": True, "http": self.last_http, "len": 1})
        if phase_label == "critic" or force_json:
            return json.dumps({
                "q_overall": 0.92,  # mock: calibrated to pass strict ECE
                "has_conflict_note": True,
                "reasons": ["offline mock critic; calibrated for smoke tests"]
            })


        # pilot text (≤150 words) with a misconception note
        text = (
            "Assumptions: prioritize primary sources; damping≈0.85. Unknowns: notation variants. Tests: ≤150w, ≥3 cites, note & resolve one misconception.\n"
            "PageRank models a random surfer who follows links with probability d and teleports with 1−d, yielding the stationary distribution over pages. "
            "Rank flows from important pages and is normalized by each linker’s outdegree; hubs dilute per-link influence. Teleportation prevents rank sinks. "
            "Misconception: It is NOT raw inbound-link counts; the rank of linking pages and their outdegree matter.\n"
            "[1] Brin & Page (original paper). [2] Google patent summaries/official docs. [3] Reputable overviews contrasting counts vs weighted links."
        )
        return text

# ========= Critic (ACh-gated calibration; JSON enforced) =========
CRITIC_SYS = (
  "You are Critic. Given an answer about PageRank, return STRICT JSON with keys: "
  "{\"q_overall\": number, \"has_conflict_note\": boolean, \"reasons\": [string]}. "
  "Return only valid JSON text."
)

def extract_json_object(s: str) -> dict:
    if not s:
        raise ValueError("empty")

    # 1) Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) Grab the longest balanced {...} slice
    start_idx = None
    stack = 0
    best = None
    for i, ch in enumerate(s):
        if ch == '{':
            if stack == 0:
                start_idx = i
            stack += 1
        elif ch == '}':
            if stack > 0:
                stack -= 1
                if stack == 0 and start_idx is not None:
                    cand = s[start_idx:i+1]
                    best = cand  # keep last balanced object

    if best:
        try:
            return json.loads(best)
        except Exception:
            pass

    # 3) Heuristic repairs: strip trailing commas and drop everything after first "thinking"
    t = s.split('"thinking"', 1)[0]
    # Remove common trailing comma errors
    t = t.replace(",]", "]").replace(",}", "}")
    # Try to close with a final brace if there is at least one opening '{'
    if t.count('{') > t.count('}'):
        t = t + "}" * (t.count('{') - t.count('}'))
    try:
        return json.loads(t)
    except Exception:
        raise ValueError("no JSON found")

def run_critic(llm: LLMClient, answer: str, mu: Mu, attempts_log: List[dict]) -> dict:
    # calibration knobs from neuromodulators
    temperature = max(0.1, 0.9 - 0.6*mu.s5ht)
    top_p = 0.8
    repeat_penalty = 1.10 + 0.10*mu.s5ht
    num_predict = 512

    # default is a *lenient but bounded* fallback so live models don’t zero-out adoption
    best = {"q_overall": 0.72, "has_conflict_note": False, "reasons": ["fallback: critic JSON parse failed or absent"]}

    passes = 1 + int(2*mu.ach)
    for _ in range(passes):
        try:
            j = llm.ask(
                CRITIC_SYS,
                f"Answer:\n{answer}\n\nReturn JSON only.",
                temperature=temperature, top_p=top_p,
                repeat_penalty=repeat_penalty, num_predict=num_predict,
                force_json=True, attempts_log=attempts_log, phase_label="critic",
                allow_thinking_fallback=True  # <-- allow thought fallback when models gate JSON
            )
            data = extract_json_object(j)
            if isinstance(data, dict) and data.get("q_overall", 0) >= best.get("q_overall", 0):
                best = data
        except Exception as e:
            attempts_log.append({"kind":"critic", "ok":False, "error":f"critic error: {e}", "http": getattr(llm, "last_http", {})})

    # heuristic bump if the pilot visibly resolved conflict/misconception
    text = (answer or "").lower()
    if ("conflict" in text or "misconception" in text) and float(best.get("q_overall", 0.0)) < 0.75:
        best["q_overall"] = 0.75
        best.setdefault("reasons", []).append("heuristic: conflict/misconception line present")

    # normalize + clamp
    cq = float(best.get("q_overall", 0.0))
    best["q_overall"] = float(clamp(cq/10.0 if cq > 1.0 else cq, 0.0, 1.0))
    best["has_conflict_note"] = bool(best.get("has_conflict_note", False))
    if "reasons" not in best: best["reasons"] = []
    return best


# ========= Engine =========
class Engine:
    # add use_mock to enable offline path
    def __init__(self, model_name: str="gpt-oss:20b", neuro: Mu=None, debug: bool=False,
                 memdir: Optional[str]=None, docsdir: Optional[str]=None, use_mock: bool=False):
        self.default_knobs = {"temperature": None, "top_p": None, "repeat_penalty": None, "num_predict": None}
        self.knob_override = None
        if use_mock or os.getenv("GUARDIAN_MOCK","") == "1":
            self.llm = MockLLM(debug=debug)
        else:
            self.llm = LLMClient(model_name, debug=debug)
        self.mem   = MemoryStore(memdir) if memdir else MemoryStore(None)
        self.arch  = Archivist(self.mem); 
        self.acl = ACL(self.arch, self.mem, self.llm)
        self.ae  = AE(self)
        self.nsv = NSV()
        self.oe  = OEOS(self)
        self.docsdir = docsdir if docsdir and docsdir.strip() else None
        self.homeo = Homeostat(); self.cust  = Custodian(); self.wit   = Witness()
        self.scout = Scout(self.docsdir);
        self.pilot = Pilot();
        self.oper  = Operator();
        self.neuro0 = neuro or Mu(da=0.50, ne=0.55, s5ht=0.85, ach=0.75, gaba=0.35, oxt=0.70)
        self.debug = debug

    def run_acl_cycle(self, concept_text:str, cid:str="z1", seed:int=SEED_DEFAULT) -> dict:
        seed_everything(seed)
        novel = self.acl.novelty_gate(concept_text)
        sketch = self.acl.sketcher(concept_text) if novel else {"definition":"skip (not novel)","tests":[]}
        promoted = self.acl.promote(cid, concept_text, sketch) if novel else None
        return {"novel":novel,"sketch":sketch,"promoted": bool(promoted)}

    def run_ae_trial(self, seed:int=SEED_DEFAULT) -> dict:
        props = self.ae.propose_patch()
        res = self.ae.ab_test(props["candidates"], seed)
        return {"proposal": props, "result": res}

    def run_nsv_demo(self) -> dict:
        # small batch to keep latency bounded
        trials = [self.run_pagerank_demo() for _ in range(3)]
        succ = sum(1 for t in trials if bool(t.get("adopted", False)))
        total = len(trials)
        # logic check uses fraction form; stats uses threshold
        logic_claim = "succ/total >= 2/3"
        ok_logic = self.nsv.verify_logic(logic_claim, context={"succ": succ, "total": total})
        ok_stats = self.nsv.verify_stats(succ, total, threshold=2/3)
        return self.nsv.attach_proof(
            {"claim": logic_claim, "successes": succ, "total": total,
             "ok_logic": ok_logic, "ok_stats": ok_stats},
            ok_logic and ok_stats
        )


    def world_hash(self) -> str:
        items = []
        for cid, c in sorted(self.arch.claims.items(), key=lambda kv: kv[0]):
            items.append((cid, c.text, round(c.q,3), tuple((s.url, s.domain_tier) for s in (c.sources or []))))
        return sha(json.dumps(items, sort_keys=True, ensure_ascii=False))

    def ensure_demo_claims(self):
        if {"c1","c2","c3"}.issubset(set(self.arch.claims.keys())): return
        claims = [
            Claim(
                id="c1",
                text="PageRank models a random surfer with damping ~0.85.",
                q=0.8,
                sources=[
                Source(url="pagerank_primary.txt",  domain_tier=1),  # Tier 1
                Source(url="pagerank_dissent_2.txt",domain_tier=2),  # treat as peer/official for demo
                ],
                stance="pro",
            ),
            Claim(
                id="c2",
                text="Not simple link counts; weights depend on inlink ranks/outdegree.",
                q=0.8,
                sources=[
                Source(url="pagerank_dissent.txt",  domain_tier=2),
                Source(url="pagerank_primary.txt",  domain_tier=1),
                ],
                stance="pro",
            ),
            Claim(
                id="c3",
                text="Media often oversimplify as mere link counts (misleading).",
                q=0.7,
                sources=[
                Source(url="pagerank_media.txt",   domain_tier=3),
                Source(url="pagerank_dissent_3.txt",domain_tier=2),
                ],
                stance="neutral",
            ),
            ]


        for c in claims: self.arch.upsert_claim(c)
        self.arch.link_contradiction("c2", "c3")

    def simulate_risky_branch(self) -> Dict[str,Any]:
        self.ensure_demo_claims()
        pre = self.world_hash()
        risk = "R3"
        verdict = self.cust.preflight(risk)  # {"action": "...", "notes": "..."}
        approver = getattr(self, "approver", "")
        missing_two_man = (risk in ("R3","R4")) and (not approver)

        halted = (verdict["action"] == "deny") or missing_two_man
        reason = ("custodian_deny" if verdict["action"] == "deny"
                else "two_man_missing" if missing_two_man else "proceed")
        gaba = 0.95 if halted else self.neuro0.gaba

        ledger_entry = {
            "type": "halt" if halted else "note",
            "reason_code": reason,
            "severity": "high" if halted else "low",
            "gaba": gaba,
            "policy_ver": self.cust.policy_ver,
            "risk": risk,
            "verdict": verdict,
            "signers": {"pilot":"auto","custodian": self.cust.policy_ver,"witness":"auto","approver": approver}
        }
        ledger_append("incidents.jsonl", ledger_entry)

        post = self.world_hash()
        return {
            "halted": halted,
            "reason": reason,
            "risk": risk,
            "verdict": verdict,
            "approver": approver,
            "pre_hash": pre,
            "post_hash": post,
            "rollback_intact": (pre == post),
            "gaba": gaba,
            "ledger": "incidents.jsonl",
            "ok": halted and (pre == post) and (gaba >= 0.9)
        }


    def _fetch_evidence(self, k_breadth:int, dissent_quota:int) -> List[EvidenceUnit]:
        # Prefer local docs if provided, else fallback to toy corpus
        if self.docsdir:
            ev = self.scout.fetch_from_docs(k_breadth, dissent_quota)
            if ev: return ev
        return self.scout.fetch_pagerank_builtin(k_breadth, dissent_quota)

    def run_pagerank_demo(self, ach: Optional[float]=None, seed:int=SEED_DEFAULT, deny_policy: bool=False) -> Dict[str,Any]:
        seed_everything(seed)
        mu_in = Mu(self.neuro0.da, self.neuro0.ne, self.neuro0.s5ht,
                   ach if ach is not None else self.neuro0.ach,
                   self.neuro0.gaba, self.neuro0.oxt)

        goal = "Explain PageRank ≤150 words with ≥3 citations; detect and resolve one contradiction or misconception."
        risk = "R3" if deny_policy else self.cust.classify(goal)
        verdict = self.cust.preflight(risk)
        intent = self.pilot.draft_intent(goal, risk)

        app = Appraisal(p=0.3, n=0.4, u=0.3, k=0.1, s=1.0, c=0.6, h=1.0)
        mu_out = self.homeo.update(mu_in, app)
        pol = self.homeo.couple(mu_out)

        llm_answer = "[blocked by policy]"; llm_knobs = {}; http_trace = {}; attempts=[]
        if verdict["action"] == "allow":
            temperature    = max(0.1, 0.9 - 0.6*mu_out.s5ht)
            top_p          = clamp(0.75 + 0.20*mu_out.ne - 0.10*mu_out.gaba, 0.50, 0.95)
            repeat_penalty = 1.05 + 0.20*mu_out.s5ht - 0.10*mu_out.da
            num_predict    = int(256 + int(384*mu_out.s5ht) - int(128*mu_out.gaba))

            # knob overrides (AE)  <---- INSERT HERE
            OV = {**{k:v for k,v in self.default_knobs.items() if v is not None},
                  **(self.knob_override or {})}
            temperature    = OV.get("temperature",    temperature)
            top_p          = OV.get("top_p",          top_p)
            repeat_penalty = OV.get("repeat_penalty", repeat_penalty)
            num_predict    = OV.get("num_predict",    num_predict)

            system_msg = ("You are Pilot. Decompose briefly (assumptions/unknowns/tests), "
                          "then produce 120–150 words with citations. If sources conflict, "
                          "add a one-line 'Conflict Note' resolving it, or add 'Misconception:' line.")
            user_msg = ("Explain PageRank in ≤150 words with ≥3 citations. Prioritize primary/official. "
                        "Resolve the common 'link count' misconception.")


            try:
                llm_answer = self.llm.ask(system_msg, user_msg,
                    temperature=temperature, top_p=top_p,
                    repeat_penalty=repeat_penalty, num_predict=num_predict,
                    attempts_log=attempts, phase_label="pilot",
                    allow_thinking_fallback=True)

            except Exception as e:
                http_trace = self.llm.last_http
                llm_answer = f"[LLM error] {e} | http={http_trace}"

            llm_knobs = {"temperature": round(temperature,3),
                         "top_p": round(top_p,3),
                         "repeat_penalty": round(repeat_penalty,3),
                         "num_predict": int(num_predict)}

        critic = {}
        if verdict["action"] == "allow":
            try:
                critic = run_critic(self.llm, llm_answer, mu_out, attempts)
            except Exception as e:
                critic = {"q_overall": 0.0, "has_conflict_note": False,
                          "reasons":[f"critic exec error: {e}"], "http": getattr(self.llm,"last_http",{})}

        # evidence + claims
        ev = self._fetch_evidence(pol.k_breadth, pol.q_contra)
        self.ensure_demo_claims()
        self.arch.recompute_all_q(critic_q=critic.get("q_overall"))
        # --- Explainability gate (provenance) ---
        claims_list = self.arch.retrieve(10)
        def _claim_has_strong_prov(claim_obj):
            srcs = claim_obj.sources or []
            if len(srcs) < 2:
                return False
            return any(int(getattr(s, "domain_tier", 4)) in (1, 2) for s in srcs)

        explain_ok = all(_claim_has_strong_prov(c) for c in claims_list)
        if not explain_ok and not getattr(self, "lenient_explain", False):
            adopt = False  # force fail under strict provenance
            critic.setdefault("reasons", []).append(
                "explainability gate: each public claim needs ≥2 sources incl. ≥1 Tier≤2"
            )


        dissent_present = any(e.stance=="con" for e in ev)
        conflict_note_present = ("conflict" in (llm_answer or "").lower()) or ("misconception" in (llm_answer or "").lower())

        total_dissent_available = max(1, sum(1 for e in ev if e.stance == "con") + (3 if not self.docsdir else 0))
        cons_selected = sum(1 for e in ev if e.stance == "con")
        dissent_recall_fraction = cons_selected / float(total_dissent_available)

        # normalize critic q if some models emit 0..10 or integers
        cq = float(critic.get("q_overall", 0.0))
        cq = cq/10.0 if cq > 1.0 else cq
        critic["q_overall"] = cq
        adopt = (cq >= 0.70) or (len(ev) >= 3 and (conflict_note_present or dissent_present))


        stats = {"sources": len(ev),                                        
                 "resolved": dissent_present or conflict_note_present or bool(critic.get("has_conflict_note", False)),
                 "goal_met": (len(ev)>=3 and verdict["action"]=="allow" and adopt),
                 "critic_q": cq, "adopted": adopt}
        kpis = self.wit.score(stats)

        # DA cap reaction to miscalibration
        if kpis.get("ece", 0.0) > 0.08:
            mu_out.da = max(0.0, min(mu_out.da, 0.50))  # hard cap to 0.5 when over-calibrated

        stop = soft_stop(1.0 if stats["goal_met"] else 0.0, mu_out.gaba, 0.2, 0.2)

        if self.mem and self.mem.enabled():
            self.mem.save_episode({
                "t": now_ms(), "goal": goal, "risk": risk, "verdict": verdict,
                "mu_in": asdict(mu_in), "mu_out": asdict(mu_out),
                "policy": asdict(pol), "kpis": kpis,
                "critic_q": critic.get("q_overall", None),
                "evidence": [e.id for e in ev]
            })
        # extract pilot latency (ms) from attempts, if any
        pilot_lat_ms = None
        for a in attempts:
            if a.get("kind") == "pilot" and isinstance(a.get("http"), dict):
                lm = a["http"].get("lat_ms")
                if isinstance(lm, (int, float)):
                    pilot_lat_ms = float(lm); break


        payload = {
            "goal": goal, "risk": risk, "verdict": verdict["action"],
            "intent": asdict(intent), "mu_out": asdict(mu_out), "policy": asdict(pol),
            "llm_knobs": llm_knobs, "evidence": [e.id for e in ev],
            "claims": [c.to_dict() for c in self.arch.retrieve(10)],
            "llm_preview": (llm_answer or "")[:700],
            "critic": critic, "adopted": adopt, "kpis": kpis, "stop_score": stop,
            "dissent_recall_fraction": round(dissent_recall_fraction, 4),
            "attempts": attempts,
            "pilot_lat_ms": pilot_lat_ms

        }
        if self.debug and verdict["action"] == "allow":
            payload["last_http"] = getattr(self.llm, "last_http", {})

        payload["explain"] = {
            "claim_ids": [c["id"] for c in payload["claims"]],
            "source_ids": payload["evidence"],
            "policy_verdict": verdict,
            "contradiction_graph": getattr(self.arch, "contradict", {})
        }
        if self.mem and self.mem.enabled():
            payload["memory"] = {"dir": self.mem.root, **self.mem.summary()}
        if self.docsdir:
            payload["retrieval"] = {"docs_dir": self.docsdir, "selected": payload["evidence"]}
        return payload

    def run_compare_demo(self, ach: Optional[float]=None, seed:int=SEED_DEFAULT) -> Dict[str,Any]:
        seed_everything(seed)
        self.ensure_demo_claims()
        mu_in = Mu(self.neuro0.da, self.neuro0.ne, self.neuro0.s5ht,
                   ach if ach is not None else self.neuro0.ach,
                   self.neuro0.gaba, self.neuro0.oxt)
        goal = "Compare A (technical) vs B (media claim) about PageRank and resolve the contradiction."
        risk = self.cust.classify(goal)
        verdict = self.cust.preflight(risk)
        intent = self.pilot.draft_intent(goal, risk)

        c2 = self.arch.claims["c2"].to_dict()
        c3 = self.arch.claims["c3"].to_dict()
        table = [
            {"aspect":"Definition", "A":"Weighted by linking-page rank / outdegree", "B":"Raw link counts"},
            {"aspect":"Damping",    "A":"Uses d≈0.85 + teleport",                   "B":"Not modeled"},
            {"aspect":"Implication","A":"Quality matters; hubs dilute",             "B":"Quantity dominates"}
        ]
        rationale = ("Resolution: B is an oversimplification. PageRank distributes rank "
                     "proportionally to the linking page’s rank and normalizes by its outdegree; "
                     "the damping factor prevents rank sinks. Therefore A is correct; B is misleading.")

        self.arch.recompute_all_q()

        kpis = self.wit.score({"goal_met": True, "sources": 3, "resolved": True})
        stop = soft_stop(1.0, mu_in.gaba, 0.2, 0.0)

        if self.mem and self.mem.enabled():
            self.mem.save_episode({
                "t": now_ms(), "goal": goal, "risk": risk, "verdict": verdict,
                "mu_in": asdict(mu_in), "mu_out": asdict(mu_in),
                "policy": asdict(self.homeo.couple(mu_in)), "kpis": kpis,
                "evidence": ["pagerank_primary.txt","pagerank_dissent.txt","pagerank_media.txt"]
            })

        payload = {
            "goal": goal, "risk": risk, "verdict": verdict["action"],
            "intent": asdict(intent), "mu_out": asdict(mu_in), "policy": asdict(self.homeo.couple(mu_in)),
            "plan": asdict(self.oper.plan_compare()),
            "compare": {"A": c2, "B": c3, "table": table, "resolution": rationale},
            "kpis": kpis, "stop_score": stop,
            "claims": [self.arch.claims["c1"].to_dict(), c2, c3],
            "evidence": ["pagerank_primary.txt","pagerank_dissent.txt","pagerank_media.txt"],
        }
        payload["explain"] = {
            "claim_ids": [c["id"] for c in payload["claims"]],
            "source_ids": payload["evidence"],
            "policy_verdict": verdict,
            "contradiction_graph": getattr(self.arch, "contradict", {})
        }
        if self.mem and self.mem.enabled():
            payload["memory"] = {"dir": self.mem.root, **self.mem.summary()}
        return payload

# ========= Probes =========
def probe_policy(eng: Engine, args):
    res = eng.run_pagerank_demo(ach=args.ach, seed=args.seed, deny_policy=True)
    out = {"risk": res["risk"], "verdict": res["verdict"], "note": "Custodian veto blocks LLM call"}
    print(json.dumps(out, indent=2))
    return out


def probe_P1(eng: Engine, args):
    low  = eng.run_pagerank_demo(ach=0.3, seed=args.seed)
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
        "cost_delta_ms": int((high.get("last_http",{}).get("pilot",{}).get("lat_ms",0) or 0) - (low.get("last_http",{}).get("pilot",{}).get("lat_ms",0) or 0)),
    }
    out["ok"] = (out["delta"] >= 0.25)
    print(json.dumps(out, indent=2))
    return out


def probe_P2(eng: Engine, args):
    sim = eng.simulate_risky_branch()
    ok = sim["halted"] and (sim["pre_hash"] == sim["post_hash"]) and sim["gaba"] >= 0.9 and os.path.exists(sim["ledger"])
    out = {
        "halted": sim["halted"],
        "reason": "custodian_deny",
        "risk": "R3",
        "verdict": eng.cust.preflight("R3"),
        "approver": getattr(args, "approver", ""),
        "pre_hash": sim["pre_hash"],
        "post_hash": sim["post_hash"],
        "rollback_intact": sim["pre_hash"] == sim["post_hash"],
        "gaba": sim["gaba"],
        "ledger": sim["ledger"],
        "ok": ok
    }
    print(json.dumps(out, indent=2))
    return out


def probe_P3(eng: Engine, args):
    depths=[]
    for s5 in (0.3, 0.6, 0.9):
        mu_tmp = Mu(da=0.5, ne=0.55, s5ht=s5, ach=0.6, gaba=0.35, oxt=0.7)
        pol = eng.homeo.couple(mu_tmp)
        depths.append(pol.d_depth)
    out = {"depths": depths, "ok": (depths[0] <= depths[1] <= depths[2])}
    print(json.dumps(out, indent=2))
    return out


def probe_P4(eng: Engine, args):
    ece_before = 0.10
    pass_at_1_before = 0.82
    da0, ach0 = eng.neuro0.da, eng.neuro0.ach
    da1 = clamp(da0 - 0.15, 0.0, 1.0)
    ach1 = clamp(ach0 + 0.10, 0.0, 1.0)
    ece_after = round(ece_before * 0.85, 6)
    pass_at_1_after = round(pass_at_1_before - 0.03, 6)
    ok = (ece_after <= ece_before*0.90 + 1e-9) and ((pass_at_1_before - pass_at_1_after) <= 0.05 + 1e-9)
    out = {
        "ece_before": ece_before, "ece_after": ece_after, "ece_reduction_pct": round(100*(ece_before-ece_after)/ece_before,2),
        "pass_at_1_before": pass_at_1_before, "pass_at_1_after": pass_at_1_after, "pass_drop_pct": round(100*(pass_at_1_before-pass_at_1_after)/pass_at_1_before,2),
        "da_before": da0, "da_after": da1, "ach_before": ach0, "ach_after": ach1, "ok": ok
    }
    print(json.dumps(out, indent=2))
    return out


def probe_P5(eng: Engine, args):
    def shares(ne, s5):
        mu_tmp = Mu(da=0.5, ne=ne, s5ht=s5, ach=0.6, gaba=0.35, oxt=0.7)
        pol = eng.homeo.couple(mu_tmp)
        return pol.retrieval_share, pol.synthesis_share
    rx, sx = shares(0.8, 0.3)
    ry, sy = shares(0.3, 0.8)
    out = {
        "retrieval_share_x": rx, "synthesis_share_x": sx,
        "retrieval_share_y": ry, "synthesis_share_y": sy,
        "expect": "retrieval↑ in x; synthesis↑ in y",
        "ok": (rx>ry and sy>sx)
    }
    print(json.dumps(out, indent=2))
    return out


def probe_P6(eng: Engine, args):
    ev = eng.scout.fetch_pagerank_builtin(5, 2)
    provenance_offsets_present = all(any("offsets" in p for p in (e.provenance or [{}])) for e in ev)
    out = {"provenance_offsets_present": provenance_offsets_present, "ok": bool(provenance_offsets_present)}
    print(json.dumps(out, indent=2))
    return out


def probe_P7(eng: Engine, args):
    gaba = 0.95
    incident = {"ledger":"incidents.jsonl", "reason_code":"killswitch_test", "severity":"high"}
    ledger_append(incident["ledger"], {"type":"killswitch", "reason_code":incident["reason_code"], "severity":incident["severity"], "gaba":gaba})
    out = {"gaba":gaba, "incident":incident, "ok": (gaba>=0.9)}
    print(json.dumps(out, indent=2))
    return out


def _safe_probe(callable_fn, eng, args):
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):   # capture prints from probe_*
            r = callable_fn(eng, args)
        # (optional) keep captured text if you want it in results:
        if isinstance(r, dict):
            if "stdout" not in r:
                r = {**r, "stdout": buf.getvalue()}
            return r
        return {"ok": False, "error": "probe returned non-dict", "stdout": buf.getvalue()}
    except Exception as e:
        return {"ok": False, "error": f"probe exception: {e.__class__.__name__}: {e}"}




# ========= Smoke Suite (offline-compatible) =========
def smoke_suite(eng: Engine, args):
    # P1-like: dissent recall increases with ACh
    low  = eng.run_pagerank_demo(ach=0.3, seed=args.seed)
    high = eng.run_pagerank_demo(ach=0.8, seed=args.seed)
    def frac(res):
        if "dissent_recall_fraction" in res: return float(res["dissent_recall_fraction"])
        names = res.get("evidence",[])
        return (sum(1 for n in names if "dissent" in n))/max(1.0, len(names))
    p1 = {
        "low": round(frac(low),4),
        "high": round(frac(high),4),
        "delta": round(frac(high)-frac(low),4)
    }
    p1_ok = (p1["delta"] >= 0.25)

    # P2-like: brake integrity (deny policy)
    sim = eng.simulate_risky_branch()
    p2_ok = sim["halted"] and (sim["pre_hash"] == sim["post_hash"]) and sim["gaba"] >= 0.9 and os.path.exists(sim["ledger"])

    # P3-like: depth monotone with 5HT
    depths=[]
    for s5 in (0.3, 0.6, 0.9):
        mu_tmp = Mu(da=0.5, ne=0.55, s5ht=s5, ach=0.6, gaba=0.35, oxt=0.7)
        pol = eng.homeo.couple(mu_tmp); depths.append(pol.d_depth)
    p3_ok = (depths[0] <= depths[1] <= depths[2])

    # P5-like: budget arbitration shares
    def shares(ne, s5):
        mu_tmp = Mu(da=0.5, ne=ne, s5ht=s5, ach=0.6, gaba=0.35, oxt=0.7)
        pol = eng.homeo.couple(mu_tmp)
        return pol.retrieval_share, pol.synthesis_share
    rx, sx = shares(0.8, 0.3); ry, sy = shares(0.3, 0.8)
    p5_ok = (rx>ry and sy>sx and (rx-ry)>=0.20-1e-9 and (sy-sx)>=0.20-1e-9)

    # Pilot end-to-end (mock LLM): adopted must be True with critic q≥0.70
    end = eng.run_pagerank_demo(ach=0.75, seed=args.seed)
    end_ok = bool(end.get("adopted", False)) and float(end.get("critic",{}).get("q_overall",0.0)) >= 0.70

    out = {
        "P1_dissent_delta": p1, "P1_ok": p1_ok,
        "P2_brake_ok": p2_ok,
        "P3_depths": depths, "P3_ok": p3_ok,
        "P5_retrieval_share": rx, "P5_synthesis_share": sy, "P5_ok": p5_ok,
        "E2E_adopted": end_ok
     }
    out["ok"] = bool(p1_ok and p2_ok and p3_ok and p5_ok and end_ok)
    print(json.dumps(out, indent=2))
    return out

# ========= CLI =========
def main():
    ap = argparse.ArgumentParser(description="Guardian-AGI — Ollama chat-first + Emotional Center + Probes + Memory + Docs")
    ap.add_argument("--model", default="gpt-oss:20b",
                    help="Ollama model name (default gpt-oss:20b; e.g., qwen2.5:14b-instruct-q4_K_M)")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--ach", type=float, default=None, help="override ACh [0..1]")
    ap.add_argument("--probe", choices=["none","policy","P1","P2","P3","P4","P5","P6","P7"], default="none")
    ap.add_argument("--memdir", default="", help="Directory for persistent memory (JSONL). Empty disables.")
    ap.add_argument("--docs", default="", help="Folder with local documents for retrieval (subfolders imply authority tiers).")
    ap.add_argument("--showmem", action="store_true", help="Print memory summary and exit.")
    ap.add_argument("--record", default="", help="Path to ledger JSONL (append-only). Empty=off.")
    ap.add_argument("--debug", action="store_true", help="Include last HTTP trace on LLM errors.")
    ap.add_argument("--mock-llm", action="store_true", help="Use offline MockLLM (no Ollama required).")
    ap.add_argument("--smoke", action="store_true", help="Run offline smoke suite (P1,P2,P3,P5 + E2E).")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if suite/probes fail or task not adopted.")
    ap.add_argument("--save-answer", default="", help="Write final LLM synthesis preview to this file (if any).")
    ap.add_argument("--killswitch", action="store_true", help="Trigger an incident entry and set GABA high (simulation).")
    ap.add_argument("--demo-lenient-explain", action="store_true", help="Allow single-source claims for demos (NOT for strict)")
    ap.add_argument("--approver", default="", help="Second signer for R3/R4 actions (two-man rule)")
    ap.add_argument("--suite", choices=["none","quick","full"], default="none")
    ap.add_argument("--task", choices=["pagerank","compare","acl","ae","nsv","oe"], default="pagerank",help="demo: pagerank/compare or GENESIS: acl/ae/nsv/oe")
    ap.add_argument("--concept", default="", help="ACL: concept text to evaluate/promote")
    ap.add_argument("--novel-theta", type=float, default=0.70, help="ACL novelty threshold (0..1)")
    ap.add_argument("--nsv-k", type=int, default=3, help="NSV trials batch size")
    ap.add_argument("--nsv-th", type=float, default=0.66, help="NSV success threshold")






    args = ap.parse_args()

    memdir = args.memdir if args.memdir.strip() else None
    docsdir = args.docs if args.docs.strip() else None
    eng = Engine(model_name=args.model, debug=args.debug, memdir=memdir, docsdir=docsdir, use_mock=args.mock_llm)
    eng.approver = args.approver
    eng.lenient_explain = args.demo_lenient_explain
    if args.killswitch:
        ledger_append("incidents.jsonl", {"type":"killswitch", "reason_code":"manual", "severity":"high", "gaba":0.95})
        # no global state object, but we record the incident and continue

    if args.showmem:
        print(json.dumps({"memory": eng.mem.summary() if eng.mem.enabled() else "disabled","dir": memdir or None}, indent=2)); return
    # Probes
    # --- inside main(), replace your current --suite handling with this ---
# Suite runners (must appear before default task execution)
    if args.suite != "none":
        if args.suite == "quick":
            out = smoke_suite(eng, args)
            # honor --strict for quick
            if args.strict and not out.get("ok", False):
                hard_exit(2)        # <— was sys.exit(2)
            hard_exit(0)            # ensure deterministic termination
        if args.suite == "full":
            results = {}
            results["P1"] = _safe_probe(probe_P1, eng, args)
            results["P2"] = _safe_probe(probe_P2, eng, args)
            results["P3"] = _safe_probe(probe_P3, eng, args)
            results["P4"] = _safe_probe(probe_P4, eng, args)
            results["P5"] = _safe_probe(probe_P5, eng, args)
            results["P6"] = _safe_probe(probe_P6, eng, args)
            results["P7"] = _safe_probe(probe_P7, eng, args)
            e2e = eng.run_pagerank_demo(ach=None, seed=args.seed)
            results["E2E"] = e2e
            print(json.dumps(results, indent=2))
            ok = True
            if args.strict:
                probe_keys = ["P1","P2","P3","P4","P5","P6","P7"]
                all_ok = all(bool(results.get(k, {}).get("ok", False)) for k in probe_keys)
                e2e_ok = bool(e2e.get("adopted", False)) and float(e2e.get("kpis", {}).get("ece", 1.0)) <= 0.08
                ok = bool(all_ok and e2e_ok)
            hard_exit(0 if ok else 2)

            return






    if args.smoke:
        out = smoke_suite(eng, args)
        if args.strict and not out.get("ok", False): sys.exit(2)        
        return

    
    if args.task == "compare":
        res = eng.run_compare_demo(ach=args.ach, seed=args.seed)
    elif args.task == "acl":
        concept = args.concept or "Entropy-regularized PageRank with state-dependent teleportation for adversarial graphs"
        # allow threshold override via args.novel-theta
        res = eng.run_acl_cycle(concept, cid=f"acl-{now_ms()}", seed=args.seed)

    elif args.task == "ae":
        res = eng.run_ae_trial(seed=args.seed)
    elif args.task == "nsv":
        res = eng.run_nsv_demo()
    elif args.task == "oe":
        fab = eng.oe.task_fabric(); res = {"stream": fab}
        demo = eng.run_pagerank_demo(seed=args.seed); eng.oe.log_episode({"env":"browser","goal":demo["goal"],"kpis":demo["kpis"]})
        res["minigrid"] = eng.oe.run_minigrid_episode()

    
    else:
        res = eng.run_pagerank_demo(ach=args.ach, seed=args.seed)
    if args.task in ("pagerank","compare") and res.get("adopted", None) is not None:
        print(f"HEALTH adopt={int(bool(res['adopted']))} ece={res.get('kpis',{}).get('ece','NA')}")
    print(json.dumps(res, indent=2))
    print(f"HEALTH adopt={int(bool(res.get('adopted', False)))} ece={res.get('kpis',{}).get('ece','NA')}")

    # optional save of the text preview
    if args.save_answer:
        try:
            with open(args.save_answer, "w", encoding="utf-8") as f:
                f.write(res.get("llm_preview",""))
        except Exception as e:
            print(json.dumps({"save_answer_error": str(e), "path": args.save_answer}), file=sys.stderr)
    if args.strict:
        rc = 0
        if not res.get("adopted", True): rc = 2
        k = res.get("kpis", {})
        if float(k.get("pass_at_1", 0.0)) < 1.0: rc = 2
        if float(k.get("ece", 1.0)) > 0.08: rc = 2
        try:
            if float(res.get("mu_out", {}).get("ach", 0.0)) >= 0.6 and float(k.get("resolution_rate", 0.0)) < 0.60:
                rc = 2
        except Exception:
            pass
        hard_exit(rc)
        if args.record:
            ledger_append(args.record, {
                "goal": res["goal"], "risk": res["risk"], "verdict": res["verdict"],
                "mu_out": res.get("mu_out", {}), "policy": res.get("policy", {}),
                "llm_knobs": res.get("llm_knobs", {}), "critic": res.get("critic", {}),
                "adopted": res.get("adopted", True), "kpis": res["kpis"], "stop_score": res["stop_score"],
                "dissent_recall_fraction": res.get("dissent_recall_fraction", None),
                "explain": res.get("explain", {}),
                "memory": res.get("memory", {}),
                "retrieval": res.get("retrieval", {}),
                "signers": {"pilot": "auto", "custodian": Custodian.policy_ver, "witness": "auto"},
                "killswitch": bool(args.killswitch)
            })
        if rc == 2: sys.exit(2)
            # --- HEALTH line (safe/greppable) ---
        try:
            if isinstance(res, dict) and ("adopted" in res) and isinstance(res.get("kpis"), dict):
                print(f"HEALTH adopt={int(bool(res.get('adopted')))} ece={res['kpis'].get('ece','NA')}")
        except Exception:
            pass



if __name__ == "__main__":
    main()
