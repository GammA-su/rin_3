import json, sys, os
import numpy as np
try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

doms = ["MATH","CODE","LANG","VISION","PLAN","TOOL","RETRIEVAL","LOGIC"]

if len(sys.argv) < 2:
    print("usage: b1_b5_report.py <logs/metrics.daily.jsonl>")
    sys.exit(2)

path = sys.argv[1]
if not os.path.exists(path):
    print("no metrics found; empty report")
    report = {"oecg_gain_abs_pct": 0.0, "drops_abs_pct_max": 0.0, "per_domain": {d:{"gain_abs_pct":0.0,"max_drop_abs_pct":0.0} for d in doms}}
    os.makedirs("reports", exist_ok=True)
    open("reports/b1_b5_report.json","w").write(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(0)

daily = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        daily.append(json.loads(line))

by_d = {d:[] for d in doms}
for row in daily:
    for d in doms:
        by_d[d].append((row["day"], row["domain_acc"][d]))

report = {"oecg_gain_abs_pct":None,"drops_abs_pct_max":None,"per_domain":{}}
gains, drops = [], []
def _pav_isotonic(y: list[float]) -> list[float]:
    # Pool Adjacent Violators (nondecreasing) – simple implementation
    n = len(y)
    if n == 0:
        return []
    # initialize blocks
    blocks = [[i, i, float(y[i]), 1] for i in range(n)]  # [start, end, mean, weight]
    k = 0
    while k < len(blocks)-1:
        if blocks[k][2] <= blocks[k+1][2]:
            k += 1
            continue
        # merge k and k+1
        a = blocks[k]; b = blocks[k+1]
        tot_w = a[3] + b[3]
        mean = (a[2]*a[3] + b[2]*b[3]) / tot_w
        new = [a[0], b[1], mean, tot_w]
        blocks[k:k+2] = [new]
        k = max(0, k-1)
    # expand back to yhat
    yhat = [0.0]*n
    for s, e, m, _ in blocks:
        for i in range(int(s), int(e)+1):
            yhat[i] = float(m)
    return yhat

for d,xs in by_d.items():
    xs.sort(); days=[t for t,_ in xs]; ys=[v for _,v in xs]
    if len(days) < 2:
        gain = 0.0; drop = 0.0
    else:
        if _HAVE_SK:
            ir = IsotonicRegression().fit(days, ys)
            yhat = ir.predict(days)
        else:
            yhat = _pav_isotonic(ys)
        gain = 100.0*(yhat[-1]-yhat[0])
        drop = 100.0*max(0.0, max((yhat[i]-yhat[i+1]) for i in range(len(yhat)-1)))
    report["per_domain"][d] = {"gain_abs_pct": float(gain), "max_drop_abs_pct": float(drop)}
    gains.append(gain); drops.append(drop)
report["oecg_gain_abs_pct"] = float(np.mean(gains) if gains else 0.0)
report["drops_abs_pct_max"] = float(np.max(drops) if drops else 0.0)
os.makedirs("reports", exist_ok=True)
open("reports/b1_b5_report.json","w").write(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
