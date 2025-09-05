import sys, json, os
try:
    import jsonschema
except Exception:
    print("jsonschema not installed", file=sys.stderr)
    sys.exit(1)

if len(sys.argv) != 3:
    print("usage: json_validate.py <schema.json> <data.json|jsonl>")
    sys.exit(2)

schema = json.load(open(sys.argv[1]))
data_path = sys.argv[2]

if data_path.endswith('.jsonl'):
    ok = True
    if not os.path.exists(data_path):
        print("json-validate-ok (missing file treated as empty)")
        sys.exit(0)
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                jsonschema.validate(instance=json.loads(line), schema=schema)
            except Exception as e:
                print(f"line {i}: {e}")
                ok = False
    if not ok:
        sys.exit(1)
else:
    jsonschema.validate(instance=json.load(open(data_path)), schema=schema)
print("json-validate-ok")

