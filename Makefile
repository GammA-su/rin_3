.PHONY: smoke probes demo clean mem

PY?=python3
SCRIPT?=guardian_agi_min.py

smoke:
	$(PY) $(SCRIPT) --mock-llm --smoke

probes:
	for p in P1 P2 P3 P4 P5 P6 P7 ; do \
		$(PY) $(SCRIPT) --mock-llm --probe $$p || exit 1 ; \
	done

demo:
	$(PY) $(SCRIPT) --mock-llm --task pagerank --memdir .guardian_mem
	$(PY) $(SCRIPT) --mock-llm --task compare --memdir .guardian_mem

mem:
	$(PY) $(SCRIPT) --mock-llm --showmem --memdir .guardian_mem

clean:
	rm -rf .guardian_mem
	rm -f incidents.jsonl
suite:
	$(PY) $(SCRIPT) --mock-llm --smoke | tee artifacts/smoke.json
	@make probes | tee artifacts/probes.log
	@grep -E '"ok": (true|false)' artifacts/probes.log | awk '{print NR": "$$0}'
	@if grep -q '"ok": false' artifacts/probes.log; then echo "[FAIL] probe failed"; exit 2; fi
	@echo "[OK] P1–P7 all passed"
e2e:
	$(PY) $(SCRIPT) --mock-llm --task pagerank --strict | tee artifacts/e2e.json

suite-full:
	$(PY) $(SCRIPT) --mock-llm --suite full --strict > artifacts/suite_full.json
	jq '.P1.ok,.P2.ok,.P3.ok,.P4.ok,.P5.ok,.P6.ok,.P7.ok,.E2E.kpis.ece,.E2E.adopted' artifacts/suite_full.json
suite-full-tee:
	set -o pipefail; \
	$(PY) $(SCRIPT) --mock-llm --suite full --strict | tee artifacts/suite_full.json
full:
	$(PY) $(SCRIPT) --mock-llm --suite full --strict > artifacts/suite_full.json
