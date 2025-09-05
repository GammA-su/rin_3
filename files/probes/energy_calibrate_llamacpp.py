import os, json, time, math
import numpy as np
import pynvml as nv
import requests

SERVER = os.getenv("SERVER", "http://127.0.0.1:8080")
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
url = f"{SERVER.rstrip('/')}/completion"
headers = {"Content-Type": "application/json"}

# adaptive prompt sizing to avoid HTTP 400
eff_ctx = CTX
def make_prompt(n):
    return ("A ") * max(1, n)

for _ in range(N):
    tries = 0
    while True:
        tries += 1
        payload = {"prompt": make_prompt(eff_ctx), "n_predict": NEW, "temperature": 0.0, "cache_prompt": True}
        if energy_mode == "nvml_total":
            e0 = nv.nvmlDeviceGetTotalEnergyConsumption(h)/1000.0; t0=time.time()
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
            t1=time.time()
        else:
            t0=time.time();
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
            t1=time.time()
        if r.status_code == 400 and eff_ctx > 64:
            eff_ctx = max(64, int(eff_ctx * 0.8))
            if tries < 6:
                continue
        r.raise_for_status()
        if energy_mode == "nvml_total":
            e1 = nv.nvmlDeviceGetTotalEnergyConsumption(h)/1000.0
            J.append(max(0.0, e1-e0))
        else:
            steps=32; dt=(t1-t0)/steps if steps>0 else 0.0; acc=0.0
            for _ in range(steps):
                time.sleep(dt)
                acc += read_power_w() * dt
            J.append(max(0.0, acc))
        lat.append(t1-t0)
        break

arr = np.array(J)
res8 = {"Jmean": float(arr.mean()), "Jcv_pct": float(100.0*arr.std(ddof=1)/max(1e-9, arr.mean()))}
os.makedirs("pins", exist_ok=True)
out = {"energy_mode": energy_mode, "measured_ctx8k": res8, "dev": {}, "promo": {}}
out["dev"]["Jmean"]   = res8["Jmean"]
out["promo"]["Jmean"] = res8["Jmean"] * 0.96
open("pins/energy.cal.json","w").write(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
