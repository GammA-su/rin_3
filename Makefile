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
