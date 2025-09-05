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
    results = {"policy": [], "bench_pins_present": []}
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

    print(json.dumps({"ok": ok_all, **results}))
    if not ok_all:
        # Non-fatal: exit 0 so your pipeline can still gather outputs; adjust to 1 to fail hard.
        sys.exit(0)

if __name__ == '__main__':
    main()

