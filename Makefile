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

.PHONY: collect-council
collect-council: ## Build council input from current outputs
	python3 files/tools/collect_council_input.py --spec UCBxTOT-gold.json --gate out/gate_v2.json --cv out/cv_tost.json --policy out/policy_scan.json --energy out/energy_mode_check.json --out files/configs/council_input.auto.json
	python3 files/tools/council_vote.py --spec UCBxTOT-gold.json --input files/configs/council_input.auto.json --out out/council.json

.PHONY: ci
ci: ## Run light CI pipeline (envlock, probes stubs, checks, bundle)
	bash scripts/run_ucbxtot_ci.sh

.PHONY: bundle
bundle: ## Create proof bundle from current outputs
	bash files/ci/make_artifact_bundle.sh

.PHONY: gpu-free
gpu-free: ## Stop/mask Ollama and kill LM Studio; show VRAM
	bash scripts/gpu_free.sh

.PHONY: probes-llamacpp
probes-llamacpp: ## Run llama.cpp probes (SERVER=http://127.0.0.1:8091)
	BACKEND=llamacpp N=3 NEW=8 CTX=2048 SERVER=$${SERVER:-http://127.0.0.1:8090} bash scripts/smoke_energy.sh

.PHONY: probes-hf
probes-hf: ## Run HF probes with safe defaults (TinyLlama 4-bit)
	BACKEND=hf N=3 NEW=8 CTX=2048 HF_MODEL=$${HF_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0} LOAD_IN_4BIT=1 bash scripts/smoke_energy.sh
