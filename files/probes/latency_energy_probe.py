import os, sys
# Set allocator knobs and make one GPU visible before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import time, math, json
import numpy as np
import torch
import pynvml as nv
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StaticCache, BitsAndBytesConfig

MODEL = os.getenv("HF_MODEL", "openai/gpt-oss-20b")
SUBFOLDER = os.getenv("HF_SUBFOLDER", "")
CTX = int(os.getenv("CTX", "8192"))
NEW = int(os.getenv("NEW", "64"))
N = int(os.getenv("N", "20"))

# Early GPU check to avoid HF warmup crash when no CUDA
if not torch.cuda.is_available():
    print("NO-CUDA: Torch cannot see a CUDA GPU. Check nvidia-smi, driver, and CUDA_VISIBLE_DEVICES.", file=sys.stderr)
    sys.exit(2)

tok = AutoTokenizer.from_pretrained(
    MODEL,
    use_fast=True,
    subfolder=SUBFOLDER if SUBFOLDER else None,
    trust_remote_code=True,
)
max_memory = {0: f"{int(os.getenv('GPU_MAX_GB','23'))}GiB", "cpu": f"{int(os.getenv('CPU_MAX_GB','64'))}GiB"}
attn_impl = os.getenv("ATTN_IMPL", "eager")
load_kwargs = dict(
    device_map='auto',
    low_cpu_mem_usage=True,
    max_memory=max_memory,
    attn_implementation=attn_impl,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
if SUBFOLDER:
    load_kwargs["subfolder"] = SUBFOLDER
# Detect MXFP4 quantization in model config and avoid BitsAndBytes if present
is_mxfp4 = False
try:
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True, **({"subfolder": SUBFOLDER} if SUBFOLDER else {}))
    qc = getattr(cfg, "quantization_config", None)
    is_mxfp4 = qc is not None and ("mxfp4" in str(qc).lower())
except Exception:
    is_mxfp4 = False

if not is_mxfp4:
    # Quantization config: prefer 4-bit NF4 if requested, else 8-bit
    if os.getenv("LOAD_IN_4BIT", "0") not in ("", "0", "false", "False"):
        q = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    else:
        q = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    load_kwargs["quantization_config"] = q
else:
    # Let Transformers load MXFP4 weights via its native quantizer
    load_kwargs.pop("quantization_config", None)
    load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
mdl = AutoModelForCausalLM.from_pretrained(MODEL, **load_kwargs)

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
    try:
        return nv.nvmlDeviceGetPowerUsage(h)/1000.0
    except nv.NVMLError:
        return 0.0

pad = tok.pad_token_id or tok.eos_token_id or 0
lat, vram, energy = [], [], []
sample_period_ms = 0.0

ctx_try = CTX
new_try = NEW

def attempt_once(ctx_len, new_tokens):
    global sample_period_ms
    with torch.inference_mode():
        ids = torch.full((1, ctx_len), pad, dtype=torch.long, device=mdl.device)
        if tok.bos_token_id is not None:
            ids[0,0] = tok.bos_token_id
        torch.cuda.reset_peak_memory_stats()
        cache = StaticCache(config=mdl.config, max_cache_len=ctx_len + new_tokens, offloading=True)
        if use_total:
            e0 = nv.nvmlDeviceGetTotalEnergyConsumption(h)/1000.0; t0=time.time()
            _ = mdl.generate(input_ids=ids, max_new_tokens=new_tokens, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize(); t1=time.time()
            e1 = nv.nvmlDeviceGetTotalEnergyConsumption(h)/1000.0
            joules = max(0.0, e1 - e0)
        else:
            t0=time.time(); _ = mdl.generate(input_ids=ids, max_new_tokens=new_tokens, past_key_values=cache, use_cache=True); torch.cuda.synchronize(); t1=time.time()
            steps = 16
            dt = (t1 - t0)/steps if steps>0 else 0.0
            sample_period_ms = dt*1000.0
            acc = 0.0
            for _ in range(steps):
                time.sleep(dt)
                acc += read_power_w() * dt
            joules = max(0.0, acc)
        return (t1 - t0), torch.cuda.max_memory_allocated()/1e9, joules

for _ in range(N):
    try:
        t, mem, j = attempt_once(ctx_try, new_try)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        # Downshift context/new tokens until it fits
        ctx_try = max(64, ctx_try // 2)
        new_try = max(1, new_try // 2)
        t, mem, j = attempt_once(ctx_try, new_try)
    lat.append(t)
    vram.append(mem)
    energy.append(j)
    if os.getenv("EMPTY_CACHE", "0") not in ("", "0", "false", "False"):
        torch.cuda.empty_cache()

out = {
  "ctx": ctx_try,
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
