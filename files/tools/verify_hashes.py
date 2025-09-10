import argparse, json, hashlib, os, sys

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_pin(path: str) -> str:
    try:
        data = open(path, 'r', encoding='utf-8').read().strip()
        # accept either raw hex or lines like "<hex>  <filename>"
        token = data.split()[0]
        return token
    except Exception:
        return ''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, help='Path to UCBxTOT-gold.json')
    args = ap.parse_args()

    spec = json.load(open(args.manifest))
    fail_closed = bool(spec.get('reproducibility', {}).get('ci_fail_closed_on_unpinned', False))
    results = {"policy": [], "bench_pins_present": [], "novelty": {}, "rag_eval": {}}
    ok_all = True

    # Policy suite files: compute sha256 of file and compare to pin value
    ps = spec.get('policy_suite', {}).get('sets', [])
    for entry in ps:
        path = entry.get('path')
        pin_file = entry.get('sha256_from_file')
        item = {"name": entry.get('name'), "path": path, "pin_file": pin_file, "ok": False}
        try:
            if not (path and os.path.exists(path)):
                item["reason"] = "missing-policy-file"
            elif not (pin_file and os.path.exists(pin_file) and os.path.getsize(pin_file) > 0):
                item["reason"] = "missing-pin"
            else:
                want = read_pin(pin_file)
                got = sha256_file(path)
                item["want"] = want
                item["got"] = got
                item["ok"] = (want == got)
                if not item["ok"]:
                    item["reason"] = "sha256-mismatch"
        except Exception as e:
            item["reason"] = f"error: {e}"
        results["policy"].append(item)
        ok_all = ok_all and item.get("ok", False)

    # Bench pins presence: we don't have local tar paths here; just check pins exist
    suites = spec.get('reproducibility', {}).get('bench_lock', {}).get('suites', {})
    for name, cfg in suites.items():
        pf = cfg.get('sha256_from_file')
        cf = cfg.get('commit_from_file')
        pres = {
            "suite": name,
            "sha256_pin": pf,
            "commit_pin": cf,
            "sha256_pin_present": bool(pf and os.path.exists(pf) and os.path.getsize(pf) > 0),
            "commit_pin_present": bool(cf and os.path.exists(cf) and os.path.getsize(cf) > 0),
        }
        results["bench_pins_present"].append(pres)
        ok_all = ok_all and pres["sha256_pin_present"] and pres["commit_pin_present"]

    # Novelty: train corpus index hash vs pin; embed pin presence
    try:
        nov = spec.get('novelty', {})
        train_path = nov.get('train_corpus_index')
        train_pin = nov.get('train_corpus_index_sha256_from_file')
        emb_pin = nov.get('embed_model_sha256_from_file')
        nov_res = {
            "train_index": train_path,
            "train_pin": train_pin,
            "train_ok": False,
            "embed_pin": emb_pin,
            "embed_pin_present": False,
        }
        # Train index hash check
        if train_path and os.path.exists(train_path) and train_pin and os.path.exists(train_pin):
            want = read_pin(train_pin)
            got = sha256_file(train_path)
            nov_res.update({"train_want": want, "train_got": got})
            if want == got and want:
                nov_res["train_ok"] = True
        # Embed pin presence (cannot verify remote model; ensure non-empty and not all-zero)
        if emb_pin and os.path.exists(emb_pin) and os.path.getsize(emb_pin) > 0:
            val = read_pin(emb_pin)
            nov_res["embed_pin_present"] = bool(val and val.strip('0') != '')
        results["novelty"] = nov_res
        ok_all = ok_all and nov_res.get("train_ok", False) and nov_res.get("embed_pin_present", False)
    except Exception:
        ok_all = False

    # RAG eval set hash vs pin
    try:
        rag = spec.get('rag', {}).get('eval_set', {})
        rag_path = rag.get('path')
        rag_pin = rag.get('sha256_from_file')
        rag_res = {"path": rag_path, "pin": rag_pin, "ok": False}
        if rag_path and os.path.exists(rag_path) and rag_pin and os.path.exists(rag_pin):
            want = read_pin(rag_pin)
            got = sha256_file(rag_path)
            rag_res.update({"want": want, "got": got})
            rag_res["ok"] = (want == got and bool(want))
        results["rag_eval"] = rag_res
        ok_all = ok_all and rag_res.get("ok", False)
    except Exception:
        ok_all = False

    out = {"ok": ok_all, **results}
    print(json.dumps(out))
    # Allow dev override via ALLOW_UNPINNED=1
    if not ok_all and fail_closed and not os.environ.get('ALLOW_UNPINNED'):
        sys.exit(1)

if __name__ == '__main__':
    main()
