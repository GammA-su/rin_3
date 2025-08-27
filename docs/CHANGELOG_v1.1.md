# Guardian-AGI Change Log

## v1.1 (Build-ready pack)
- Finalized appraisal math + clamp ranges.
- Homeostat → policy couplings table locked; ACh raises contradiction quota (q_contra≥2).
- Custodian: risk matrices, kill-switch, two-man rule; per-episode policy_ver pin.
- Archivist schemas + compaction policy + provenance invariant (≥2 independent sources, mixed tiers).
- Scout: source tiering, dissent targeting, recency windows.
- Pilot/Operator: HTN checklists (T1/T2/T4), budget splits, rollback semantics.
- Witness: KPI targets (C1), ECE coupling, drift monitors.
- Probes P1–P7 harness; P1 metric fixed (recall vs total dissent).
- Critic JSON extractor hardened (brace stack, no `(?R)`).

## v1.0 (Spec baseline)
- Initial single-file scaffold, minimal claims, single-dissent pool, basic KPIs.
