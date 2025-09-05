import os, time, math, json
import numpy as np
import pynvml as nv
import requests

SERVER = os.getenv("SERVER", "http://localhost:8000/v1")
MODEL  = os.getenv("HF_MODEL", os.getenv("MODEL", "openai/gpt-oss-20b"))
CTX    = int(os.getenv("CTX", "8192"))
NEW    = int(os.getenv("NEW", "64"))
N      = int(os.getenv("N", "10"))

def qtile(a,q):
    a=np.sort(np.asarray(a)); i=max(0, math.ceil(q*len(a))-1); return float(a[i])

nv.nvmlInit(); h = nv.nvmlDeviceGetHandleByIndex(0)
def has_total():
    try:
        nv.nvmlDeviceGetTotalEnergyConsumption(h); return True
    except nv.NVMLError:
        return False
use_total = has_total()
energy_mode = "nvml_total" if use_total else "power_integral"

def read_power_w():
    try: return nv.nvmlDeviceGetPowerUsage(h)/1000.0
    except nv.NVMLError: return 0.0

lat, vram, energy = [], [], []
sample_period_ms = 0.0

# Build a large prompt approximating CTX tokens
prompt = ("A ") * CTX

url = f"{SERVER.rstrip('/')}/completions"
headers = {"Content-Type": "application/json"}
payload = {"model": MODEL, "prompt": prompt, "max_tokens": NEW, "temperature": 0.0}

for _ in range(N):
    if use_total:
        e0 = nv.nvmlDeviceGetTotalEnergyConsumption(h)/1000.0; t0=time.time()
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
        t1=time.time(); r.raise_for_status()
        e1 = nv.nvmlDeviceGetTotalEnergyConsumption(h)/1000.0
        joules = max(0.0, e1 - e0)
    else:
        t0=time.time();
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
        t1=time.time(); r.raise_for_status()
        steps = 32
        dt = (t1 - t0)/steps if steps>0 else 0.0
        sample_period_ms = dt*1000.0
        acc = 0.0
        for _ in range(steps):
            time.sleep(dt)
            acc += read_power_w() * dt
        joules = max(0.0, acc)
    lat.append(t1 - t0)
    try:
        mem = nv.nvmlDeviceGetMemoryInfo(h).used/1e9
    except nv.NVMLError:
        mem = 0.0
    vram.append(mem)
    energy.append(joules)

out = {
  "ctx": CTX,
  "energy_mode": energy_mode,
  "samples": N,
  "sample_period_ms": sample_period_ms if not use_total else 0.0,
  "p50_s": float(np.median(lat)),
  "p95_s": qtile(lat,0.95),
  "p99_s": qtile(lat,0.99),
  "vram_gb": float(np.mean(vram)),
  "j_per_inf": float(np.mean(energy))
}
print(json.dumps(out))
