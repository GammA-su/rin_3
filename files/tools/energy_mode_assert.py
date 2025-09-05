import json, sys, os

if len(sys.argv) < 2:
    print("usage: energy_mode_assert.py <calibration.json> <probe.json> [<probe.json> ...]")
    sys.exit(2)

cal = json.load(open(sys.argv[1]))
want = cal.get("energy_mode")
obs = []
for p in sys.argv[2:]:
    try:
        d = json.load(open(p))
        obs.append(d.get("energy_mode"))
    except Exception:
        obs.append(None)
ok = (want is not None) and all(m == want for m in obs)
out = {"required": want, "observed": obs, "ok": ok}
os.makedirs("out", exist_ok=True)
open("out/energy_mode_check.json","w").write(json.dumps(out))
if not ok:
    print("ENERGY-MODE-MISMATCH", out)
    sys.exit(1)
print("energy-mode-ok")

