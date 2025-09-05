import sys, json, pathlib

if len(sys.argv) < 2:
    print("usage: assert_caps_equal.py <UCBxTOT-gold.json>")
    sys.exit(2)

CFG = json.load(open(sys.argv[1]))

def need_files(paths):
    missing = [p for p in paths if not (pathlib.Path(p).exists() and pathlib.Path(p).stat().st_size > 0)]
    if missing:
        print("MISSING PINS:", ", ".join(missing)); sys.exit(1)

def get(path):
    cur = CFG
    for k in path.split("."): cur = cur[k]
    return cur

need_files([
  "pins/embed.sha256","pins/train.idx.sha256","pins/rag.sha256",
  "pins/adv.sha256","pins/harm.sha256","pins/energy.cal.json"
])

a = float(get("gate_v2.global_caps.lora_adapter_params_m_max"))
b = float(get("gate_v2.micro.caps.params_active_m"))
if a != b:
    print(f"ASSERT-FAIL: lora_adapter cap mismatch global={a} micro={b}")
    sys.exit(1)

c = float(get("bars.B2.time_budget_hours"))
d = float(get("atf.tst.wall_clock_hours_max"))
if c != d:
    print(f"ASSERT-FAIL: B2 time mismatch bars={c} atf.tst={d}")
    sys.exit(1)

print("assertions-ok")

