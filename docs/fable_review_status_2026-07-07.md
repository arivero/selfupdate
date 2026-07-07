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

- `scripts/model_matrix.py` — **missing** (cross-model comparison figure; data
  already exists in `runs/corpus.csv`).
- `scripts/conclusion_check.py` — **missing**.
- `runs/conclusions.yaml` — **missing** (machine-readable conclusion ledger; seed
  from `EXPERIMENTS.md`).
- `evaluate.py --layer-residuals` — **absent** (checkpoint-time per-layer residual
  eval: storage quality vs training loss; also C3 item #8).

## LEFT — speed / hardware (review §3)

- Equivalence-tested **padded/bucketed batching path** — unbuilt (C3 #5, "4–6×");
  same bottleneck hit on the 40B q_ch1 run (pipeline bubbles + per-block sync).
- Fused / `foreach` optimizer stepping.
- **PP2/TP2 correctness certification** vs a single-device reference (`pp2fix` / TP chain).

## LEFT — open research gaps (review §4; science, not cleanup)

- Crown **seed claim** still open (`lw_r_s43_pinned`).
- **Disjoint-window** pinned evidence needed (`lw_r_disj_pinned`, C2-35).
- **Teacher-stream k-windows** not implemented (C3 #1).
- **H100** throughput / memory / PP-TP evidence absent.

## Count

- Build work: **4** — `model_matrix.py`, `conclusion_check.py`, `conclusions.yaml`,
  `evaluate --layer-residuals`.
- Speed/H100: **3** — batching path, foreach stepping, PP2/TP2 cert.
- Research gaps: **4** — seed claim, disjoint pinned, teacher-stream k-windows, H100 evidence.
