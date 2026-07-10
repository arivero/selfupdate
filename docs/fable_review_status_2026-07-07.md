# Remaining work — FableReviewBy55xh.md + recommendations.md

Audited 2026-07-07 (121 commits after the 2026-07-05 review). The **DONE** and
**SUPERSEDED** items have been edited out to leave only actionable remaining work;
the full done/left/superseded audit is in this file's git history (commit that
first added it). In short, what was removed: the tail/target-law purge, config
audit + tests, corpus index, run_class, MoE routing modes, layer-loss CSV+heatmap
and per-run forget curves are **DONE**; the `report.py` 11-section redesign and the
general-CE forget-curves are **SUPERSEDED** by `cross_report.py` + the retention
battery.

## LEFT — build work

All four build items landed 2026-07-10: `scripts/model_matrix.py`
(runs/model_matrix.{csv,png}), `scripts/conclusion_check.py`,
`runs/conclusions.yaml` (13 claims, validates clean), and
`evaluate.py --layer-residuals` (layer_residuals.{json,csv,png}; first
profile on lw_r_s43_pinned shows the expected shallow-tight /
deep-departing storage signature). Nothing left in this section.

## LEFT — speed / hardware (review §3)

Review §3's speed priorities are CLOSED as of the 2026-07-10 refactor
(GPU-side logging, equivalence-tested batching, explicit optimizer policy
with streamed pinned offload, cache-read question resolved by item
memoization + a measured-negative prefetch, PP2 certification: certs/pp2 vs
certs/pre plus the lw_q_pp2fix science repro at CER 0.011; TP2 probe-only by
policy). See docs/runtime.md and issues.md 2026-07-10 notes. Remaining:

- Large-model batched measurements: 4B LoRA landed 2026-07-10
  (item 5.9 items/s @ 8.1 GB; padded B4 13.8 items/s @ 8.6 GB —
  runs/bench_4b_*.json); 8B points queued the same day
  (runs/bench_8b_*.json when landed). Nothing else left here.

## LEFT — open research gaps (review §4; science, not cleanup)

- ~~Crown seed claim~~ CLOSED 2026-07-10: `lw_r_s43_pinned` replicated the
  crown (CER 0.0076 / 0.991 / intrusion 1.5%).
- ~~Disjoint-window pinned evidence~~ CLOSED 2026-07-10: `lw_r_disj_pinned`
  recalls clean (0.023 / 7% / non-destructive) — C2-35 resolved.
- **Teacher-stream k-windows** not implemented (C3 #1).
- **H100** throughput / memory / PP-TP evidence absent (needs an H100
  allocation; L40S evidence complete).
- NEW: **1.7B cleanliness** — xs spectrum recalls but intrusion stays
  22-40% at 1.7B (vs 1.5-2.5% at 0.6B).

## Count

- Build work: **0**.
- Speed/H100: **0** on L40S; H100 evidence awaits hardware.
- Research gaps: **3** — teacher-stream k-windows, H100 evidence, 1.7B
  cleanliness (plus the crown17 owner decision recorded in
  EXPERIMENTS.md).
