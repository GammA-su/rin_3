import os, time, math, json
import numpy as np
import pynvml as nv
import requests

# llama.cpp server defaults
SERVER = os.getenv("SERVER", "http://127.0.0.1:8080")
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

url = f"{SERVER.rstrip('/')}/completion"
headers = {"Content-Type": "application/json"}

# adaptively back off prompt length on HTTP 400 (too long / bad request)
eff_ctx = CTX
def make_prompt(n):
    # keep it simple and deterministic; chars ~= tokens order
    return ("A ") * max(1, n)

for _ in range(N):
    tries = 0
    while True:
        tries += 1
        payload = {"prompt": make_prompt(eff_ctx), "n_predict": NEW, "temperature": 0.0, "cache_prompt": True}
        if use_total:
            e0 = nv.nvmlDeviceGetTotalEnergyConsumption(h)/1000.0; t0=time.time()
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
            t1=time.time()
        else:
            t0=time.time();
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
            t1=time.time()
        if r.status_code == 400 and eff_ctx > 64:
            # back off prompt length and retry quickly
            eff_ctx = max(64, int(eff_ctx * 0.8))
            if tries < 6:
                continue
        r.raise_for_status()
        if use_total:
            e1 = nv.nvmlDeviceGetTotalEnergyConsumption(h)/1000.0
            joules = max(0.0, e1 - e0)
        else:
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
        break

out = {
  "ctx": eff_ctx,
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
