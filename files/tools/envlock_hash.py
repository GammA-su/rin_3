import hashlib, json, os, platform, subprocess, sys

def sh(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True).strip()
        return out
    except Exception:
        return ""

env = {
    "os": platform.platform(),
    "python": sys.version.split()[0],
    "cuda": sh("nvcc --version | tail -1 | awk '{print $5}'"),
    "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
    "pytorch": sh("python -c 'import torch,sys;print(torch.__version__)'"),
}

blob = json.dumps(env, sort_keys=True).encode()
h = hashlib.sha256(blob).hexdigest()
out = {"env": env, "hash": h}
os.makedirs("files/configs", exist_ok=True)
open("files/configs/env.hash", "w").write(json.dumps(out, indent=2))
print(json.dumps(out))

