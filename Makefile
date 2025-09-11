suite:      ## Run quick suite in strict mode with memory
	python3 guardian_agi_min.py --memdir .guardian_mem --suite quick --strict
acl:        ## Promote a new concept (edit CONCEPT=)
	python3 guardian_agi_min.py --task acl --memdir .guardian_mem --concept "$(CONCEPT)"
ae:         ## A/B adopt knobs if better
	python3 guardian_agi_min.py --task ae --strict
nsv:        ## Run NSV batch proof
	python3 guardian_agi_min.py --task nsv
oe:         ## Log one episode
	python3 guardian_agi_min.py --task oe

.PHONY: smoke-energy
smoke-energy: ## OOM-safe energy smoke (BACKEND=hf|ollama|vllm)
	bash scripts/smoke_energy.sh

.PHONY: gate-v2
gate-v2: ## Run Gate v2 orchestrator on sample candidates
	python3 files/tools/gate_v2.py --spec UCBxTOT-gold.json --candidates files/configs/gatev2_candidates.json --pvals-out out/pvals.json --out out/gate_v2.json

.PHONY: council
council: ## Run Council vote on sample inputs
	python3 files/tools/council_vote.py --spec UCBxTOT-gold.json --input files/configs/council_input.json --out out/council.json

.PHONY: promote
promote: ## Run Gate v2 + Council and append to transparency log
	bash scripts/run_promotion.sh

.PHONY: promote-auto
promote-auto: ## Auto-candidate from probes (+ optional PVALS_IN) then Gate v2 + Council
	bash scripts/promote_auto.sh

.PHONY: atf-auto
atf-auto: ## Append ATF admission using current artifacts (cv_tost + tails + novelty)
	python3 files/tools/atf_auto.py --spec UCBxTOT-gold.json --probe8 out/out_ctx8k.json --probe16 out/out_ctx16k.json --cv out/cv_tost.json --nov out/novelty.json --tool-id auto.tool --out logs/atf.daily.jsonl
	python3 files/tools/json_validate.py files/schemas/atf_result.schema.json logs/atf.daily.jsonl || true

.PHONY: atf-from-artifact
atf-from-artifact: ## Append ATF admission using eval artifact + probes + novelty
	python3 files/tools/atf_from_artifact.py --spec UCBxTOT-gold.json --artifact $${IN:-artifacts/suite_full.json} --probe8 out/out_ctx8k.json --probe16 out/out_ctx16k.json --nov out/novelty.json --tool-id e2e.run --out logs/atf.daily.jsonl
	python3 files/tools/json_validate.py files/schemas/atf_result.schema.json logs/atf.daily.jsonl || true

.PHONY: collect-council
collect-council: ## Build council input from current outputs
	python3 files/tools/collect_council_input.py --spec UCBxTOT-gold.json --gate out/gate_v2.json --cv out/cv_tost.json --policy out/policy_scan.json --energy out/energy_mode_check.json --out files/configs/council_input.auto.json
	python3 files/tools/council_vote.py --spec UCBxTOT-gold.json --input files/configs/council_input.auto.json --out out/council.json

.PHONY: ci
ci: ## Run light CI pipeline (envlock, probes stubs, checks, bundle)
	bash scripts/run_ucbxtot_ci.sh

.PHONY: suite-full
suite-full: ## Alias for CI pipeline used by GitHub Actions
	$(MAKE) ci

.PHONY: bundle
bundle: ## Create proof bundle from current outputs
	bash files/ci/make_artifact_bundle.sh

.PHONY: status
status: ## Print current proof status summary
	python3 files/tools/proof_status.py

.PHONY: pvals-from-csv
pvals-from-csv: ## Build files/configs/pvals.input.json from a tidy CSV (suite,seed,base,cand)
	python3 files/tools/pvals_from_csv.py --csv $${CSV:-evals.csv} --out files/configs/pvals.input.json

.PHONY: pvals-aggregate
pvals-aggregate: ## Merge multiple CSVs/pvals docs into files/configs/pvals.input.json
	python3 files/tools/pvals_aggregate.py $${CSVS:+--csv $${CSVS}} $${PVALS:+--pvals $${PVALS}} --out $${OUT:-files/configs/pvals.input.json}

.PHONY: pvals-from-manifest
pvals-from-manifest: ## Build pvals.input.json from files/configs/eval_sources.json (or your manifest)
	python3 files/tools/pvals_from_manifest.py --manifest $${MANIFEST:-files/configs/eval_sources.json} --out $${OUT:-files/configs/pvals.input.json}

.PHONY: evals-to-csv
evals-to-csv: ## Convert eval artifact (CSV/JSON/JSONL/Parquet) to tidy CSV (suite,seed,base,cand)
	python3 files/tools/evals_to_csv.py --in $${IN:-artifacts/suite_full.json} --out $${OUT:-evals.csv} $${SUITE:+--suite $${SUITE}} $${SUITE_COL:+--suite-col $${SUITE_COL}} $${SEED:+--seed $${SEED}} $${SEED_COL:+--seed-col $${SEED_COL}} $${BASE_COL:+--base-col $${BASE_COL}} $${CAND_COL:+--cand-col $${CAND_COL}}

.PHONY: pvals-from-artifact
pvals-from-artifact: ## Convert artifact to CSV then build pvals.input.json
	$(MAKE) evals-to-csv IN=$${IN:-artifacts/suite_full.json} OUT=$${OUT:-evals.csv} SUITE=$${SUITE:-my-suite} SEED=$${SEED:-7}
	$(MAKE) pvals-from-csv CSV=$${OUT:-evals.csv}

.PHONY: proof-cycle
proof-cycle: ## Stable probes -> pvals (from artifact) -> promote -> atf -> bundle
	SERVER=$${SERVER:-http://127.0.0.1:8090} $(MAKE) probes-llamacpp-stable
	$(MAKE) pvals-from-artifact IN=$${IN:-artifacts/suite_full.json} OUT=$${OUT:-evals.csv} SUITE=$${SUITE:-my-suite} SEED=$${SEED:-7} $${BASE_COL:+BASE_COL=$${BASE_COL}} $${CAND_COL:+CAND_COL=$${CAND_COL}}
	SERVER=$${SERVER:-http://127.0.0.1:8090} $(MAKE) promote-auto PVALS_IN=files/configs/pvals.input.json
	$(MAKE) atf-auto
	$(MAKE) bundle

.PHONY: proof-cycle-multi
proof-cycle-multi: ## Aggregate pvals from multiple CSVs/pvals, then promote + atf + bundle
	# Aggregate p-values (set CSVS and/or PVALS env vars)
	$(MAKE) pvals-aggregate $${CSVS:+CSVS="$${CSVS}"} $${PVALS:+PVALS="$${PVALS}"} OUT=files/configs/pvals.input.json
	# Ensure probes exist (skip if you already ran stable probes)
	SERVER=$${SERVER:-http://127.0.0.1:8090} $(MAKE) probes-llamacpp-stable || true
	# Promote using aggregated p-values
	SERVER=$${SERVER:-http://127.0.0.1:8090} $(MAKE) promote-auto PVALS_IN=files/configs/pvals.input.json
	# Append ATF from artifact if available, else artifact-less ATF
	if [ -s artifacts/suite_full.json ]; then \
	  python3 files/tools/atf_from_artifact.py --spec UCBxTOT-gold.json --artifact artifacts/suite_full.json --probe8 out/out_ctx8k.json --probe16 out/out_ctx16k.json --nov out/novelty.json --tool-id e2e.run --out logs/atf.daily.jsonl || true ; \
	else \
	  python3 files/tools/atf_auto.py --spec UCBxTOT-gold.json --probe8 out/out_ctx8k.json --probe16 out/out_ctx16k.json --cv out/cv_tost.json --nov out/novelty.json --tool-id auto.tool --out logs/atf.daily.jsonl || true ; \
	fi
	# Bundle proof
	$(MAKE) bundle

.PHONY: metrics-from-artifact
metrics-from-artifact: ## Append a daily B1–B5 line using artifact (and optional domain map)
	python3 files/tools/metrics_from_artifact.py --spec UCBxTOT-gold.json --artifact $${IN:-artifacts/suite_full.json} $${MAP:+--map $${MAP}} --out logs/metrics.daily.jsonl
	python3 files/tools/json_validate.py files/schemas/metrics.daily.schema.json logs/metrics.daily.jsonl || true
	python3 files/tools/b1_b5_report.py logs/metrics.daily.jsonl > reports/b1_b5_report.json || true
	python3 files/tools/json_validate.py files/schemas/b1_b5_report.schema.json reports/b1_b5_report.json || true

.PHONY: novelty-setup
novelty-setup: ## Install novelty deps in venv and pre-download embed model from spec
	python3 files/tools/novelty_setup.py --spec UCBxTOT-gold.json

.PHONY: domain-map-wizard
domain-map-wizard: ## List numeric metric paths and optionally write files/configs/domain_map.json
	python3 files/tools/domain_map_wizard.py --artifact $${IN:-artifacts/suite_full.json} $${SET_ALL:+--set-all $${SET_ALL}} $${SET:+--set $${SET}} $${OUT:+--out $${OUT}}

.PHONY: gpu-free
gpu-free: ## Stop/mask Ollama and kill LM Studio; show VRAM
	bash scripts/gpu_free.sh

.PHONY: probes-llamacpp
probes-llamacpp: ## Run llama.cpp probes (SERVER=http://127.0.0.1:8091)
	BACKEND=llamacpp N=3 NEW=8 CTX=2048 SERVER=$${SERVER:-http://127.0.0.1:8090} bash scripts/smoke_energy.sh

.PHONY: probes-llamacpp-stable
probes-llamacpp-stable: ## Warmup + stable probes (defaults: WARMUP=20, N=200, NEW=24, CTX8=6144, CTX16=8192)
	SERVER=$${SERVER:-http://127.0.0.1:8090} \
	CTX8=$${CTX8:-6144} CTX16=$${CTX16:-8192} N=$${N:-200} NEW=$${NEW:-24} WARMUP=$${WARMUP:-20} \
	bash scripts/probes_llamacpp_stable.sh

.PHONY: acl-sweep
acl-sweep: ## Sweep a list of concepts through ACL (topics file: FILE=files/configs/topics.txt)
	bash scripts/acl_sweep.sh $${FILE:-files/configs/topics.txt} $${MEMDIR:-.guardian_mem} $${NOVEL_THETA:-0.7}

.PHONY: autopilot
autopilot: ## Continuous discovery+promotion loop (env: HOURS, INTERVAL=0 for nonstop, SERVER)
	python3 files/tools/autopilot.py $${MEMDIR:+--memdir $${MEMDIR}} $${DOCS:+--docs $${DOCS}} $${TOPICS:+--topics $${TOPICS}} $${NOVEL_THETA:+--novel-theta $${NOVEL_THETA}} $${MANIFEST:+--manifest $${MANIFEST}} $${SERVER:+--server $${SERVER}} $${INTERVAL:+--interval-min $${INTERVAL}} $${CYCLES:+--cycles $${CYCLES}} $${HOURS:+--hours $${HOURS}}

.PHONY: ground-claims
ground-claims: ## Populate sources/stance for ACL claims from docs (env: MEMDIR, DOCS, K)
	python3 files/tools/ground_claims.py $${MEMDIR:+--memdir $${MEMDIR}} $${DOCS:+--docs $${DOCS}} $${K:+-k $${K}}

.PHONY: web-enrich-claims
web-enrich-claims: ## Fetch web evidence for claims (ALLOW_NET=1) and append sources/stance
	python3 files/tools/web_enrich_claims.py $${MEMDIR:+--memdir $${MEMDIR}} $${DOCS:+--docs $${DOCS}} $${K:+-k $${K}} $${ONLY_MISSING:+--only-missing}

.PHONY: test-learned
test-learned: ## Ask the LLM to explain learned claims using attached sources (env: MEMDIR, INCLUDE, EXCLUDE, LIMIT, MODEL, HOST, PORT, MOCK)
	python3 files/tools/test_learned_claims.py $${MEMDIR:+--memdir $${MEMDIR}} $${INCLUDE:+--include $${INCLUDE}} $${EXCLUDE:+--exclude $${EXCLUDE}} $${LIMIT:+--limit $${LIMIT}} $${MODEL:+--model $${MODEL}} $${HOST:+--host $${HOST}} $${PORT:+--port $${PORT}} $${MOCK:+--mock}

.PHONY: test-learned-llama
test-learned-llama: ## Use llama.cpp server to explain learned claims (env: MEMDIR, INCLUDE, EXCLUDE, LIMIT, LLAMA=http://127.0.0.1:11434)
	python3 files/tools/test_learned_claims.py $${MEMDIR:+--memdir $${MEMDIR}} $${INCLUDE:+--include $${INCLUDE}} $${EXCLUDE:+--exclude $${EXCLUDE}} $${LIMIT:+--limit $${LIMIT}} $${LLAMA:+--llama-server $${LLAMA}}

.PHONY: console
console: ## Interactive AGI console (env: LLAMA=http://127.0.0.1:11435, DOCS=docs, GBNF=files/grammars/json_text_cites.gbnf)
	python3 files/tools/agi_console.py $${LLAMA:+--llama-server $${LLAMA}} $${DOCS:+--docs $${DOCS}} $${GBNF:+--gbnf $${GBNF}}

.PHONY: claims-similarities
claims-similarities: ## Find cross-domain similarities between brain and AGI claims (env: MEMDIR, K=3, MIN_SIM=0.25)
	python3 files/tools/claims_similarity.py $${MEMDIR:+--memdir $${MEMDIR}} $${INCB:+--include-brain $${INCB}} $${INCA:+--include-agi $${INCA}} $${K:+--k $${K}} $${MIN_SIM:+--min-sim $${MIN_SIM}}

.PHONY: memory-prune
memory-prune: ## Prune memory: dedupe by text, drop slugs, require sources+tier (env: IN, OUT, MAX=100000, MIN_SOURCES=1, TIER_MAX=3, DROP=^(en-wikipedia-org|arxiv-org-abs), DOMAINS, INCLUDE)
	python3 files/tools/memory_prune.py $${IN:+--in $${IN}} $${OUT:+--out $${OUT}} $${MAX:+--max $${MAX}} $${MIN_SOURCES:+--min-sources $${MIN_SOURCES}} $${TIER_MAX:+--tier-max $${TIER_MAX}} $${DROP:+--drop-slugs $${DROP}} $${DOMAINS:+--domains $${DOMAINS}} $${INCLUDE:+--include $${INCLUDE}}

.PHONY: harvest-topics
harvest-topics: ## Fetch many wiki/arXiv intros per topic (env: TOPICS=files/configs/topics.brain.txt, DOCS=docs/brain, WIKI_K=12, RELATED_K=6, ARXIV_K=6, ALLOW_NET=1)
	python3 files/tools/harvest_topics.py $${TOPICS:+--topics $${TOPICS}} $${DOCS:+--docs $${DOCS}} $${WIKI_K:+--wiki-k $${WIKI_K}} $${RELATED_K:+--related-k $${RELATED_K}} $${ARXIV_K:+--arxiv-k $${ARXIV_K}}

.PHONY: build-cards
build-cards: ## Create knowledge cards (3–5 sentences + cites) (env: TOPICS, DOCS, MEMDIR, LLAMA, K=8, MAX_EACH=600)
	python3 files/tools/build_cards.py $${TOPICS:+--topics $${TOPICS}} $${DOCS:+--docs $${DOCS}} $${MEMDIR:+--memdir $${MEMDIR}} $${LLAMA:+--llama $${LLAMA}} $${K:+--k $${K}} $${MAX_EACH:+--max_each $${MAX_EACH}}

.PHONY: build-notes-tests
build-notes-tests: ## Build 400–800 word notes + 10 Q/A per topic (env: TOPICS, DOCS, MEMDIR, LLAMA, K=12, MAX_EACH=1200)
	python3 files/tools/build_notes_tests.py $${TOPICS:+--topics $${TOPICS}} $${DOCS:+--docs $${DOCS}} $${MEMDIR:+--memdir $${MEMDIR}} $${LLAMA:+--llama $${LLAMA}} $${K:+--k $${K}} $${MAX_EACH:+--max_each $${MAX_EACH}}

.PHONY: build-graph
build-graph: ## Build concept graph from cards/notes (env: CARDS, NOTES, LLAMA, OUT)
	python3 files/tools/build_concept_graph.py $${CARDS:+--cards $${CARDS}} $${NOTES:+--notes $${NOTES}} $${LLAMA:+--llama $${LLAMA}} $${OUT:+--out $${OUT}}

.PHONY: learn-brain
learn-brain: ## One-shot brain loop: harvest -> (review keep.txt) -> prune -> cards -> notes/tests -> graph -> prune memory -> autopilot (set CONFIRM=1 to proceed past review)
	# Defaults (shell variables)
	BRAIN_TOPICS=$${TOPICS:-files/configs/topics.brain.txt}; \
	BRAIN_DOCS=$${DOCS:-docs/brain}; \
	BRAIN_MEM=$${MEMDIR:-.guardian_mem_brain}; \
	LLAMA_SRV=$${LLAMA:-http://127.0.0.1:11435}; \
	SRV=$${SERVER:-http://127.0.0.1:8090}; \
	mkdir -p "$$BRAIN_DOCS"; \
	# 1) Harvest many brain-specific sources (wiki search + related + arXiv)
	ALLOW_NET=1 UA=$${UA:-Triforce-Prophet/1.0 (+local)} $(MAKE) harvest-topics TOPICS="$$BRAIN_TOPICS" DOCS="$$BRAIN_DOCS" $${WIKI_K:+WIKI_K=$${WIKI_K}} $${RELATED_K:+RELATED_K=$${RELATED_K}} $${ARXIV_K:+ARXIV_K=$${ARXIV_K}}; \
	# 2) Propose keep list (review before deletion)
	find "$$BRAIN_DOCS" -type f | LC_ALL=C grep -E -i -f files/configs/brain_keep.patterns > keep.txt || true; \
	if [ -z "$$CONFIRM" ]; then \
	  echo; echo 'Review keep.txt (brain-only files). When ready, rerun with CONFIRM=1 to prune and continue:'; \
	  echo '    make learn-brain CONFIRM=1 [TOPICS=… DOCS=… MEMDIR=… LLAMA=… SERVER=…]'; echo; exit 0; \
	fi; \
	# 3) Prune non-brain files based on keep.txt
	comm -23 <(find "$$BRAIN_DOCS" -type f | sort) <(sort keep.txt) | xargs -r rm -f; \
	# 4) Build knowledge cards (3–5 sentences + cites)
	$(MAKE) build-cards TOPICS="$$BRAIN_TOPICS" DOCS="$$BRAIN_DOCS" MEMDIR="$$BRAIN_MEM" LLAMA="$$LLAMA_SRV" $${K:+K=$${K}} $${MAX_EACH:+MAX_EACH=$${MAX_EACH}}; \
	# 5) Build notes (400–800 words) + 10 Q/A per topic
	$(MAKE) build-notes-tests TOPICS="$$BRAIN_TOPICS" DOCS="$$BRAIN_DOCS" MEMDIR="$$BRAIN_MEM" LLAMA="$$LLAMA_SRV" $${K:+K=$${K}} $${MAX_EACH:+MAX_EACH=$${MAX_EACH}}; \
	# 6) Build concept graph from cards/notes
	$(MAKE) build-graph CARDS="$$BRAIN_MEM/knowledge.cards.jsonl" NOTES="$$BRAIN_MEM/knowledge.notes.jsonl" LLAMA="$$LLAMA_SRV" OUT="$$BRAIN_MEM/concepts.graph.json"; \
	# 7) Prune memory to compact brain-only, deduped claims
	$(MAKE) memory-prune IN="$$BRAIN_MEM/claims.jsonl" OUT="$$BRAIN_MEM/claims.brain.clean.jsonl" MAX=$${MAX_CLAIMS:-5000} MIN_SOURCES=1 TIER_MAX=2 DOMAINS='BRAIN' INCLUDE='(cortex|hippocamp|synaps|dopamin|basal|prefrontal|thalam|cerebell|neuron|neuromodul|predictive|workspace|oscillat|grid|place|vta|snc)'; \
	mv "$$BRAIN_MEM/claims.jsonl" "$$BRAIN_MEM/claims.jsonl.bak" 2>/dev/null || true; mv "$$BRAIN_MEM/claims.brain.clean.jsonl" "$$BRAIN_MEM/claims.jsonl"; \
	# 8) Lean autopilot (no web in-loop)
	SERVER="$$SRV" MEMDIR="$$BRAIN_MEM" NOVEL_THETA=$${NOVEL_THETA:-0.85} TOPICS_ONLY=1 TOPICS="$$BRAIN_TOPICS" DOCS="$$BRAIN_DOCS" $(MAKE) autopilot INTERVAL=$${INTERVAL:-0}

.PHONY: probes-hf
probes-hf: ## Run HF probes with safe defaults (TinyLlama 4-bit)
	BACKEND=hf N=3 NEW=8 CTX=2048 HF_MODEL=$${HF_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0} LOAD_IN_4BIT=1 bash scripts/smoke_energy.sh
