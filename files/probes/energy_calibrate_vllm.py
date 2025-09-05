import os, json, time, math
import numpy as np
import pynvml as nv
import requests

SERVER = os.getenv("SERVER", "http://localhost:8000/v1")
MODEL  = os.getenv("HF_MODEL", os.getenv("MODEL", "openai/gpt-oss-20b"))
CTX    = int(os.getenv("CTX", "8192"))
NEW    = int(os.getenv("NEW", "64"))
N      = int(os.getenv("N", "20"))

def qtile(a,q):
    a=np.sort(np.asarray(a)); i=max(0, math.ceil(q*len(a))-1); return float(a[i])

nv.nvmlInit(); h = nv.nvmlDeviceGetHandleByIndex(0)
def has_total():
    try:
        nv.nvmlDeviceGetTotalEnergyConsumption(h); return True
    except nv.NVMLError:
        return False
energy_mode = "nvml_total" if has_total() else "power_integral"

def read_power_w():
    try: return nv.nvmlDeviceGetPowerUsage(h)/1000.0
    except nv.NVMLError: return 0.0

lat, J = [], []
prompt = ("A ") * CTX
url = f"{SERVER.rstrip('/')}/completions"
headers = {"Content-Type": "application/json"}
payload = {"model": MODEL, "prompt": prompt, "max_tokens": NEW, "temperature": 0.0}

for _ in range(N):
    if energy_mode == "nvml_total":
        e0 = nv.nvmlDeviceGetTotalEnergyConsumption(h)/1000.0; t0=time.time()
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
        t1=time.time(); r.raise_for_status()
        e1 = nv.nvmlDeviceGetTotalEnergyConsumption(h)/1000.0
        J.append(max(0.0, e1-e0))
    else:
        t0=time.time()
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
        t1=time.time(); r.raise_for_status()
        steps=32; dt=(t1-t0)/steps if steps>0 else 0.0; acc=0.0
        for _ in range(steps):
            time.sleep(dt)
            acc += read_power_w() * dt
        J.append(max(0.0, acc))
    lat.append(t1-t0)

arr = np.array(J)
res8 = {"Jmean": float(arr.mean()), "Jcv_pct": float(100.0*arr.std(ddof=1)/max(1e-9, arr.mean()))}
out = {"energy_mode": energy_mode, "measured_ctx8k": res8, "dev": {}, "promo": {}}
out["dev"]["Jmean"]   = res8["Jmean"]
out["promo"]["Jmean"] = res8["Jmean"] * 0.96
os.makedirs("pins", exist_ok=True)
open("pins/energy.cal.json","w").write(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
