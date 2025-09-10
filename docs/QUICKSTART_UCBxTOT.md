UCB×TOT Quickstart (single 4090)

1) Start llama.cpp server

- Example:
  - `LLAMA_SERVER_BIN=/path/to/llama-server \\
     GGUF=/path/to/gpt-oss-20b-Q4_K_M.gguf PORT=8090 CTX=8192 NGL=99 FLASH_ATTN=auto \\
     ./scripts/run_llamacpp_server.sh`

2) Collect stable probes

- `SERVER=http://127.0.0.1:8090 make probes-llamacpp-stable`
- This produces `out/out_ctx8k.json`, `out/out_ctx16k.json`, and checks energy mode.

3) Produce p-values from your evals

- From a CSV/JSON/JSONL/Parquet artifact to a tidy CSV:
  - `make evals-to-csv IN=artifacts/suite_full.json OUT=evals.csv SUITE=my-suite SEED=7 \\
     BASE_COL=E2E.kpis.base_acc CAND_COL=E2E.kpis.cand_acc`
- Build p-values JSON:
  - `make pvals-from-csv CSV=evals.csv`

4) Promote (Gate v2 + Council) and append to ledger

- `SERVER=http://127.0.0.1:8090 make promote-auto PVALS_IN=files/configs/pvals.input.json`

5) Seed ATF admission from current artifacts (optional)

- `make atf-auto`

6) Bundle proof

- `make bundle`

All outputs are placed under `out/` (probes, stats, promotion summary), `logs/` (transparent log, ATF), and `reports/`.

7) Status + Nightly Evidence

- Quick status snapshot:
  - `make status` (writes `out/proof_status.json` and prints a summary)
- Nightly run (scheduled; can also dispatch manually from Actions):
  - Appends daily metrics/ATF from `artifacts/suite_full.json`, bundles proof, and uploads artifact.
  - Commits updated `logs/metrics.daily.jsonl`, `logs/atf.daily.jsonl`, and `reports/b1_b5_report.json` to the repo.

8) Domain Metrics Mapping (optional)

- Inspect metric paths in your artifact and set a domain map:
  - `make domain-map-wizard IN=artifacts/suite_full.json`
  - Set one metric for all domains:
    - `make domain-map-wizard IN=artifacts/suite_full.json SET_ALL=E2E.kpis.pass_at_1 OUT=files/configs/domain_map.json`
  - Or set per-domain:
    - `make domain-map-wizard IN=artifacts/suite_full.json SET='MATH=path1,CODE=path2,...'`

9) Multi‑suite promotion (optional)

- Aggregate p‑values from multiple sources, then promote:
  - `make pvals-aggregate CSVS="evals_seed5.csv evals_seed7.csv evals_seed11.csv" OUT=files/configs/pvals.input.json`
  - Or merge existing pvals docs: `make pvals-aggregate PVALS="pvals_a.json pvals_b.json"`
  - Run the multi‑suite cycle: `SERVER=http://127.0.0.1:8090 make proof-cycle-multi CSVS="evals_seed5.csv evals_seed7.csv evals_seed11.csv"`
