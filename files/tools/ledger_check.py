import json, sys, hashlib, os

if len(sys.argv) < 2:
    print("usage: ledger_check.py <rekor-local.jsonl>")
    sys.exit(2)

path = sys.argv[1]
prev_ts = -1
prev_chain = "0"*64
n = 0
if not os.path.exists(path):
    # Empty, but considered ok as append-only new log
    out = {"entries": 0, "head_chain": prev_chain, "ok": True}
    os.makedirs("out", exist_ok=True)
    open("out/ledger_check.json","w").write(json.dumps(out))
    print("ledger-ok", out)
    sys.exit(0)

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        ts  = int(e.get("timestamp", 0))
        ph  = e.get("prev_chain", "0"*64)
        pay = e.get("payload_hash", "")
        if ts <= prev_ts:
            print("NON-MONOTONIC-TIMESTAMP"); sys.exit(1)
        if ph != prev_chain:
            print("PREV-CHAIN-MISMATCH"); sys.exit(1)
        h = hashlib.sha256((prev_chain + pay + str(ts)).encode()).hexdigest()
        prev_chain = h
        prev_ts = ts
        n += 1
out = {"entries": n, "head_chain": prev_chain, "ok": True}
os.makedirs("out", exist_ok=True)
open("out/ledger_check.json","w").write(json.dumps(out))
print("ledger-ok", out)

