# Python Source Token Report

Generated 2026-07-10 on branch `layerwise`.

- **Tokenizer:** `Qwen/Qwen3-0.6B` (the project's workhorse model), loaded offline
  from the local HF cache, `add_special_tokens=False`.
- **File set:** all `*.py` tracked by git plus untracked-but-not-ignored
  (`git ls-files` ∪ `git ls-files --others --exclude-standard`), so `.venv/`
  and other ignored trees are excluded by construction. 110 files.
- **Method:** each file read as UTF-8 and tokenized whole; lines counted as
  newline-terminated lines.

## Totals

| Metric | Value |
|---|---:|
| Tokens | **215,634** |
| Files | 110 |
| Lines | 21,263 |
| Chars | 850,513 |
| Density | 3.94 chars/token |

## By top-level directory

| Tokens | Lines | Chars | Directory |
|---:|---:|---:|---|
| 109,801 | 10,605 | 426,407 | `scripts/` |
| 61,346 | 5,795 | 252,085 | `src/` |
| 27,500 | 2,758 | 102,770 | `tests/` |
| 11,849 | 1,763 | 53,251 | `docs/` |
| 5,138 | 342 | 16,000 | `paper/` |

## Per file (descending by tokens)

| Tokens | Lines | Chars | File |
|---:|---:|---:|---|
| 14,932 | 1,424 | 58,534 | `scripts/experiment_report_assets.py` |
| 14,030 | 1,275 | 59,683 | `src/selfupdate/train/layerwise.py` |
| 11,849 | 1,763 | 53,251 | `docs/lens_diagnostics.py` |
| 8,037 | 739 | 30,240 | `scripts/report.py` |
| 7,759 | 724 | 30,116 | `scripts/retention_eval.py` |
| 5,138 | 342 | 16,000 | `paper/make_figs.py` |
| 4,978 | 375 | 18,129 | `scripts/cross_report.py` |
| 4,829 | 450 | 18,362 | `src/selfupdate/data/poem.py` |
| 4,412 | 429 | 19,318 | `src/selfupdate/train/runtime.py` |
| 4,220 | 383 | 17,117 | `scripts/train_certify.py` |
| 4,123 | 375 | 16,159 | `src/selfupdate/eval/destruction.py` |
| 3,901 | 370 | 16,240 | `src/selfupdate/train/moe.py` |
| 3,836 | 446 | 15,973 | `scripts/standard_destruction_eval.py` |
| 3,655 | 352 | 13,953 | `scripts/build_corpus_index.py` |
| 3,468 | 337 | 12,852 | `tests/test_moe_modes.py` |
| 3,045 | 286 | 12,859 | `scripts/evaluate.py` |
| 3,031 | 250 | 11,258 | `scripts/analyze.py` |
| 2,988 | 257 | 11,552 | `src/selfupdate/config.py` |
| 2,984 | 239 | 9,755 | `scripts/retention_plots.py` |
| 2,876 | 248 | 10,976 | `scripts/surprise_probe.py` |
| 2,859 | 319 | 11,869 | `src/selfupdate/masking.py` |
| 2,853 | 276 | 11,472 | `scripts/speed_check.py` |
| 2,836 | 327 | 10,592 | `scripts/trajectory_plot.py` |
| 2,779 | 254 | 10,745 | `scripts/tasks_report.py` |
| 2,701 | 308 | 11,447 | `scripts/qualitative_chat_review.py` |
| 2,687 | 287 | 11,112 | `src/selfupdate/data/dataset.py` |
| 2,657 | 259 | 9,703 | `scripts/retention_index.py` |
| 2,609 | 167 | 9,577 | `src/selfupdate/eval/probes.py` |
| 2,546 | 224 | 9,923 | `scripts/memory_plan.py` |
| 2,517 | 268 | 11,019 | `src/selfupdate/eval/recite.py` |
| 2,340 | 186 | 9,258 | `src/selfupdate/train/losses.py` |
| 2,087 | 200 | 8,548 | `scripts/parallel_bench.py` |
| 2,069 | 219 | 8,983 | `src/selfupdate/chatfmt.py` |
| 1,986 | 176 | 7,353 | `src/selfupdate/eval/tasks.py` |
| 1,915 | 182 | 7,875 | `scripts/signal_attribution.py` |
| 1,895 | 191 | 8,871 | `src/selfupdate/train/blocks.py` |
| 1,882 | 176 | 7,168 | `scripts/smoke_family.py` |
| 1,819 | 163 | 5,846 | `tests/test_losses.py` |
| 1,782 | 174 | 6,163 | `scripts/fetch_quijote.py` |
| 1,736 | 145 | 5,943 | `tests/test_destruction.py` |
| 1,727 | 207 | 7,564 | `scripts/train_batch_bench.py` |
| 1,625 | 168 | 6,559 | `scripts/layer_loss_plots.py` |
| 1,607 | 156 | 5,563 | `scripts/forget_curves.py` |
| 1,597 | 132 | 5,738 | `scripts/attention_probe.py` |
| 1,521 | 142 | 6,587 | `scripts/destruct_eval.py` |
| 1,483 | 137 | 5,376 | `scripts/moe_router_probe.py` |
| 1,443 | 152 | 5,609 | `tests/test_training_target_law.py` |
| 1,436 | 156 | 6,429 | `tests/test_alignment.py` |
| 1,376 | 144 | 5,400 | `tests/test_layerwise_locality.py` |
| 1,372 | 145 | 5,128 | `tests/test_padded_batching.py` |
| 1,313 | 122 | 5,472 | `scripts/teacher_ceiling.py` |
| 1,287 | 150 | 5,220 | `scripts/layer_delta_timeline.py` |
| 1,277 | 110 | 4,574 | `tests/test_thinking_selective.py` |
| 1,257 | 112 | 4,417 | `tests/test_conn_window.py` |
| 1,220 | 111 | 4,271 | `scripts/build_coverage_queue.py` |
| 1,206 | 133 | 4,339 | `tests/test_window_dedup.py` |
| 1,201 | 116 | 4,737 | `tests/test_online_teacher.py` |
| 1,182 | 121 | 4,688 | `src/selfupdate/teacher/cache.py` |
| 1,174 | 122 | 4,244 | `scripts/model_matrix.py` |
| 1,149 | 110 | 4,465 | `src/selfupdate/train/tuned_lens.py` |
| 1,121 | 138 | 4,413 | `scripts/audit_configs.py` |
| 1,113 | 116 | 4,419 | `scripts/train_tuned_lens.py` |
| 1,073 | 121 | 4,363 | `tests/test_chatfmt.py` |
| 1,059 | 108 | 4,086 | `scripts/recite_long.py` |
| 1,031 | 96 | 4,338 | `src/selfupdate/experimental/frontier_recall.py` |
| 999 | 107 | 4,200 | `scripts/build_dataset.py` |
| 988 | 86 | 3,227 | `tests/test_analysis.py` |
| 976 | 109 | 3,558 | `scripts/summarize_standard_destruction.py` |
| 964 | 101 | 3,596 | `tests/test_mixed_schedule.py` |
| 933 | 103 | 3,954 | `scripts/build_teacher_cache.py` |
| 932 | 90 | 3,618 | `tests/test_tuned_lens.py` |
| 918 | 102 | 3,621 | `scripts/logit_lens.py` |
| 898 | 92 | 3,471 | `tests/test_cache_roundtrip.py` |
| 831 | 93 | 3,543 | `scripts/sanity_chat.py` |
| 817 | 75 | 2,998 | `tests/test_anchor.py` |
| 755 | 84 | 3,086 | `tests/test_offload_adam.py` |
| 737 | 81 | 2,732 | `src/selfupdate/eval/weight_deltas.py` |
| 711 | 81 | 3,310 | `src/selfupdate/teacher/generate.py` |
| 693 | 79 | 2,897 | `scripts/conclusion_check.py` |
| 651 | 79 | 2,436 | `scripts/fetch_poem.py` |
| 649 | 72 | 2,620 | `scripts/delta_profiles.py` |
| 613 | 67 | 2,356 | `src/selfupdate/eval/layer_swap.py` |
| 600 | 69 | 2,445 | `scripts/layer_swap.py` |
| 577 | 67 | 2,157 | `tests/test_config_audit.py` |
| 576 | 49 | 2,368 | `src/selfupdate/eval/logit_lens.py` |
| 540 | 46 | 2,113 | `src/selfupdate/eval/general.py` |
| 514 | 53 | 2,078 | `scripts/frontier_recall.py` |
| 501 | 56 | 1,891 | `tests/test_corpus_style.py` |
| 463 | 48 | 1,609 | `tests/test_tasks_report.py` |
| 449 | 58 | 1,758 | `tests/test_config_loader.py` |
| 424 | 50 | 1,663 | `scripts/train.py` |
| 408 | 49 | 1,602 | `src/selfupdate/utils/runlog.py` |
| 394 | 51 | 1,740 | `src/selfupdate/train/lora.py` |
| 393 | 41 | 1,579 | `src/selfupdate/eval/convergence.py` |
| 385 | 37 | 1,345 | `tests/test_catechism.py` |
| 347 | 36 | 1,233 | `tests/test_gemma4_blockstack.py` |
| 347 | 38 | 1,400 | `tests/test_position_invariance.py` |
| 326 | 47 | 1,367 | `tests/test_recite.py` |
| 304 | 21 | 1,182 | `src/selfupdate/experimental/__init__.py` |
| 218 | 18 | 730 | `scripts/premise_gate.py` |
| 157 | 16 | 604 | `scripts/base_general.py` |
| 87 | 9 | 377 | `tests/conftest.py` |
| 63 | 13 | 256 | `src/selfupdate/utils/seeding.py` |
| 0 | 0 | 0 | `src/selfupdate/__init__.py` |
| 0 | 0 | 0 | `src/selfupdate/data/__init__.py` |
| 0 | 0 | 0 | `src/selfupdate/eval/__init__.py` |
| 0 | 0 | 0 | `src/selfupdate/teacher/__init__.py` |
| 0 | 0 | 0 | `src/selfupdate/train/__init__.py` |
| 0 | 0 | 0 | `src/selfupdate/utils/__init__.py` |
| 0 | 0 | 0 | `tests/__init__.py` |

## Notes

- ~51% of the token mass is in `scripts/` (reporting/eval/plotting harnesses);
  the science core under `src/` is ~61k tokens, dominated by
  `train/layerwise.py` (14k) and `train/runtime.py` (4.4k).
- 3.94 chars/token — denser than English prose (~4.5–5) because BPE fragments
  identifiers, operators, and indentation into small tokens.
- The whole Python surface (~216k tokens) fits in a single 1M-token context;
  a targeted `src/` + key-scripts pass is ~90k.
