# Ops SOPs v1.1

## Pre-flight (global)
- policy_ver pinned
- signer keys valid
- ledger space ≥ 2× daily volume
- ACh floor set by task criticality (C2 ≥ 0.5)
- probe datasets loaded

## Per-task
- Create Task object
- Pilot fills I(A,U,T_a,Z)+Goal-Fit
- Custodian classifies risk, issues notes/disclaimer
- Witness sets KPI targets
- Homeostat initializes μ baseline (C2: 5HT≥0.85)

## Post-task
- Witness computes KPIs & ECE
- Archivist writes Episode, updates q
- Custodian attaches policy notes
- If unresolved_conflicts>0 and ACh<0.6 → schedule T2 Compare-Resolve

## Audit
- Weekly: 5% sample recomputation
- Monthly: policy diff + key rotation
- Quarterly: external evaluation
