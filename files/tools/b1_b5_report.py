import json, sys, os
import numpy as np
try:
    from sklearn.isotonic import IsotonicRegression
except Exception as e:
    print("scikit-learn not installed:", e, file=sys.stderr)
    sys.exit(1)

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
for d,xs in by_d.items():
    xs.sort(); days=[t for t,_ in xs]; ys=[v for _,v in xs]
    if len(days) < 2:
        gain = 0.0; drop = 0.0
    else:
        ir = IsotonicRegression().fit(days, ys)
        yhat = ir.predict(days)
        gain = 100.0*(yhat[-1]-yhat[0])
        drop = 100.0*max(0.0, max((yhat[i]-yhat[i+1]) for i in range(len(yhat)-1)))
    report["per_domain"][d] = {"gain_abs_pct": float(gain), "max_drop_abs_pct": float(drop)}
    gains.append(gain); drops.append(drop)
report["oecg_gain_abs_pct"] = float(np.mean(gains) if gains else 0.0)
report["drops_abs_pct_max"] = float(np.max(drops) if drops else 0.0)
os.makedirs("reports", exist_ok=True)
open("reports/b1_b5_report.json","w").write(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

