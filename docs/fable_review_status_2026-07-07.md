# Status audit — FableReviewBy55xh.md + recommendations.md

Quantifies the Fable review (`FableReviewBy55xh.md`, 2026-07-05) and its
`recommendations.md` against the current tree on 2026-07-07. **121 commits** landed
since the review; most of the cleanup is done. Each item is scored **DONE** /
**LEFT** / **SUPERSEDED** (made irrelevant by a later pivot) with the evidence used.

Headline: the **branch-hygiene cleanup is essentially complete**; what remains is a
short list of **analysis scripts + the conclusion ledger**, the **speed/H100** work,
and **open research gaps**. The **report.py redesign and general-CE forget-curves**
have been **superseded** by `cross_report.py` + the retention battery.

## DONE (verified in tree)

Tail / target-law purge (the review's #1 priority):
- `tail_step` → `window_step` — layerwise.py: 0 `tail_step`, 10 `window_step`.
- `tail_ce_blocks/_weight/_kind`, `tail_hidden_weight`, `answer_ce_weight`,
  `last_block_ce_weight`, `lens_ce_weight` — **all removed from `config.py`** (0 hits).
  This closes the stale-knob, anchor-coupling, and `tail_ce_kind`-inheritance findings at once.
- `configs/base.yaml` sets no `readout_source`/`tail_ce_kind`.
- `tests/test_tail_ce.py` removed; no active config has `tail_ce_blocks>0`.
- Config-audit enforcement: `scripts/audit_configs.py` rejects `tail_ce_*` keys,
  reference-text `readout_source`, base.yaml setting a source, and unpinned source
  on readout runs. Backed by `tests/test_config_audit.py`, `test_training_target_law.py`,
  `test_conn_window.py`, `test_window_dedup.py`.
- Crown moved to a teacher-sourced sliding arm (`slide8pure`) per `CLAUDE.md` pointer.

Tooling the review/recommendations asked for:
- Implemented & present: `audit_configs.py`, `build_corpus_index.py` (→ `runs/corpus.csv` exists),
  `report.py`, `layer_loss_plots.py` (now emits **CSV + heatmap**, closing the "PNG-only" gap),
  `forget_curves.py` (now **per-run + grouped**, reads `base-general-*`), `speed_check.py`,
  `base_general.py`, `signal_attribution.py`.
- Run classification (`run_class`) live and carried into `runs/corpus.csv`.
- MoE routing modes (`dense_or_black_box` / `teacher_forced` / `router_aligned`) implemented
  (commits: MoE routing modes, router overlap, MXFP4 gpt-oss fix).
- Hot-loop logging keeps loss tensors on-device, flushes at accum/epoch boundaries.

## LEFT (still missing / pending)

Analysis scripts & artifacts:
- `scripts/model_matrix.py` — **missing** (cross-model comparison figure).
- `scripts/conclusion_check.py` — **missing**.
- `runs/conclusions.yaml` — **missing** (the machine-readable conclusion ledger).
- `evaluate.py --layer-residuals` — **absent** (checkpoint-time per-layer residual eval;
  also C3 item #8). This is the review's "storage quality vs training loss" gap.

Speed / hardware (review §3 priorities 2–5):
- Equivalence-tested **padded/bucketed batching path** — unbuilt (C3 #5, "4–6×"); this is the
  same bottleneck we hit today on the 40B q_ch1 run (pipeline bubbles + sync).
- Fused/`foreach` optimizer stepping — not done.
- **PP2/TP2 correctness certification** vs a single-device reference — pending (`pp2fix`/TP chain).

Open research gaps (review §4 — these are science, not cleanup):
- Crown **seed claim** still open (`lw_r_s43_pinned`).
- **Disjoint-window** pinned evidence still needed (`lw_r_disj_pinned`, C2-35).
- **Teacher-stream k-windows** not implemented (C3 #1).
- **H100** throughput/memory/PP-TP evidence absent.

## SUPERSEDED (made irrelevant by a later pivot)

- **`report.py` "Report Redesign" (11-section order)** — the tree pivoted to
  `cross_report.py` (cross-checkout, 375 ln) + `retention_eval.py`/`retention_plots.py`.
  The reporting surface evolved differently than the recommendation specified; treat the
  redesign spec as historical intent, not a live TODO.
- **General-CE forget-curves emphasis** — general-CE was judged noisy and **replaced** by the
  ARC-Easy/WikiText + exact-recall **retention battery** (`retention_eval.py`, cross-checkout
  `retention_index.csv`). `forget_curves.py` still exists but general-CE is no longer the
  headline forgetting metric.
- **Entire "Branch Hygiene First / remove tail_*" section** — completed, therefore moot; keep
  only as the record of why tail material is excluded.
- **Llama-8B / Phi-4-mini coverage** — the review itself dropped these; the live target is the
  2026 ladder (Qwen3.6-27B, Gemma-4-26B-A4B/31B, gpt-oss), partially in progress (Gemma
  teacher-refs queued, gpt-oss MoE landed).

## `CERTIFICATETHING.md` (home-dir top level)

Not audit material: a one-line `export SSL_CERT_FILE=…` reminder, already documented in
`CLAUDE.md`. Duplicate Fable docs also exist in the `selfupdate_multi` sibling checkout.

## Score (major line items)

- Cleanup / config-audit / tooling: **~DONE** (tail purge, audit, corpus index, classification,
  plots CSV+heatmap, MoE modes, base-general refs).
- Remaining build work: **4 items** — `model_matrix.py`, `conclusion_check.py`,
  `conclusions.yaml`, `evaluate --layer-residuals`.
- Speed/H100: **~4 items** — batching path, foreach stepping, PP2/TP2 cert (hot-loop logging done).
- Research gaps: **4 open** — seed claim, disjoint pinned, teacher-stream k-windows, H100 evidence.
- Superseded: **report redesign + general-CE curves** (→ cross_report + retention battery).
