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

- Equivalence-tested **padded/bucketed batching path** — built for the summed
  schedule (`train.batching: padded|bucketed`; `tests/test_padded_batching.py`).
  Large-model throughput and peak-memory measurements remain to be collected.
- Resident summed training uses one AdamW instance so LoRA parameters can be
  `foreach`-stepped across blocks. Full-FT and `offload_adam` deliberately use
  the lower-peak non-foreach path; a fused full-FT optimizer is not a free win
  because foreach intermediates scale with trainable parameter count.
- **PP2/TP2 correctness certification** vs a single-device reference (`pp2fix` / TP chain).
  `scripts/parallel_bench.py` now emits JSON and checks nonzero loss/update
  metrics against a single-device reference; matched hardware evidence is
  still pending.

## LEFT — open research gaps (review §4; science, not cleanup)

- Crown **seed claim** still open (`lw_r_s43_pinned`).
- **Disjoint-window** pinned evidence needed (`lw_r_disj_pinned`, C2-35).
- **Teacher-stream k-windows** not implemented (C3 #1).
- **H100** throughput / memory / PP-TP evidence absent.

## Count

- Build work: **4** — `model_matrix.py`, `conclusion_check.py`, `conclusions.yaml`,
  `evaluate --layer-residuals`.
- Speed/H100: **2 evidence tasks** — batched large-model measurements and
  PP2/TP2 certification runs.
- Research gaps: **4** — seed claim, disjoint pinned, teacher-stream k-windows, H100 evidence.
