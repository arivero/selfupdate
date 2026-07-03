# Experiment plan & status board

Updated: 2026-07-03 ~07:00. Metrics: `runs/results.md` (auto) · report: `runs/report.pdf` · logs: `runs/pipeline_*.log`.
Base model control: CER 0.932, general-CE 3.278.

## Wave A — method grid, Qwen3-0.6B, v1 data (✅ done)

| run | status | headline |
|---|---|---|
| kd_full (pure KD) | ✅ | KL→0.03 but CER 0.82: distillation alone doesn't recite |
| kd_ce (KD + gold-CE, 20 ep) | ✅ | **best recitation: CER 0.596, 44% lines verbatim**; forget +0.52 |
| lw_summed | ✅ | no recitation (0.90); forget +0.58 |
| lw_seq | ✅ | no recitation (0.94); **3.2 GB = 34% of full backprop**; forget +0.41 |
| kd_lora (lr 1e-5) | ✅ | KL stuck at 2.2 (lr too low); forget only +0.06 |

## Wave B — (a) student-stream vs (b) teacher_censored, LoRA (✅ done)

| run | status | headline |
|---|---|---|
| lw_summed_lora (a) | ✅ | CER 0.945, forget +0.31 |
| lw_tc_lora (b) | ✅ | **(b) dominates: CER 0.878, forget +0.11**; increment peak @ layer 7 |

## Wave C — causal analysis, hybrids, 1.7B first contact (🟡 eval finishing)

| item | status | headline |
|---|---|---|
| graft/ablate on kd_ce | ✅ | graft: early layers (4–8) carry content; ablate: deep (25–28) needed |
| logit lens on kd_ce | ✅ | recall readable only above layer ~21 |
| lw_summed_ce (hybrid last-block CE) | ✅ | failed (0.99) — one block's CE can't coordinate the rest |
| lw_tc_ce | ✅ | 0.867 ≈ no gain over (b) at lr 1e-5 |
| kd_lora_ce (lr 1e-5) | ✅ | 0.943 — lr was the binding constraint |
| kd_lora_ce_1p7b (lr 1e-5) | 🟡 eval running | trained 36 min @ 4.8 GB (full-FT wouldn't fit) |

## Wave D — round 2: proper LoRA lr (⏳ queued, next)

| run | why |
|---|---|
| kd_lora_ce_hi (lr 1e-4) | the bet: recitation ≈ full-FT at ~2 GB, forgetting ≈ 0 |
| lw_tc_ce_hi (lr 1e-4, 30 ep) | can (b)+CE recite with a working lr? |
| kd_ce_long (40 ep, full-FT) | coverage: does more training fix the front-of-poem bias? |

## Wave D2 — round 2 on the second model (⏳ queued)

| run | why |
|---|---|
| kd_lora_ce_hi_1p7b | recipe transfer + is the layer-7 peak positional or proportional? |

## Wave E — v2 extended-recitation data (⏳ queued)

| item | why |
|---|---|
| v2 cache + kd_lora_ce_hi_v2 + kd_ce_v2 (40 ep) | paraphrases, 24/48-verse windows, part chunks (333 tasks) |
| recite_long chained evals (incl. kd_ce v1) | the Pierre Menard metric: whole-poem self-chained recitation |

## Wave F — compaction axis (⏳ queued, last)

| run | why |
|---|---|
| kd_lora_ce_hi_stub | uninformative placeholder vs outright removal |
| kd_lora_ce_hi_stubgap | + position-gap: teacher-identical RoPE geometry |

## Watchdog backlog (fills idle GPU, any time)

| item | status |
|---|---|
| logit lens ×3 (kd_full, lw_summed, lw_seq) | ⏳ |
| graft/ablate ×3 (same runs) — causal cross-method localization | ⏳ |
| lw_seq_bf16 re-measure (expect ~1.9 GB ≈ 20% of full backprop) | ⏳ |
| kd_full re-eval (fills forgetting NaN) | ⏳ |
| 1.7B cache + lw_seq_1p7b (memory-curve point 2) | ⏳ |

## Original milestone plan (for the record; superseded by the waves above)

M1 data+cache+premise ✅ · M2 KD baseline (CER<5% subset) ✅ · M3 layerwise
(locality tests ✅, VRAM<40% ✅ via sequential, recitation ❌ open) · M4
analysis suite ✅ · M5 LoRA axis ✅ · M6 scale-up prep (docs ✅, streaming
mock + FSDP2 stubs pending). Original grid axis not yet run: **thinking-mode**
(trace harvesting implemented; needs a ~30 min generation pass + cache).

## Beyond (not yet scheduled)

- Batched eval generation (task #1; 5–8× eval speedup)
- Thinking-mode arm (trace harvesting implemented, never run)
- Per-block lens-CE layerwise variant (if hybrids keep failing)
- Sequential + online-teacher lockstep cache
- Move to 2×4090: one experiment per GPU (AGENTS.md Tier 1); replicate grid with 2nd seed
