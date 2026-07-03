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

| run | status | headline |
|---|---|---|
| kd_lora_ce_hi (lr 1e-4) | ✅ | learns (full CER 0.774, 22% exact, 2.3 GB) but forgetting +0.96 — WORST; forgetting tracks amount learned, not parameterization |
| kd_lora_ce_mid (lr 3e-5, 40 ep) | ⏳ queued | the middle point of the lr/forgetting trade |
| lw_tc_ce_hi (lr 1e-4, 30 ep) | ✅ | still no recitation (0.840, 0 exact), forget +0.99 — 7th layerwise config, all negative: local losses can't do multi-layer credit assignment |
| kd_ce_long (40 ep, full-FT) | coverage: does more training fix the front-of-poem bias? |

## Wave D2 — round 2 on the second model (⏳ queued)

| run | status | headline |
|---|---|---|
| kd_lora_ce_hi_1p7b | ✅ | recipe transfers: full CER 0.798, 13% exact (subset 0.42/45%) — same shape incl. coverage bias; forgetting delta pending 1.7B base ref |

## Wave E — v2 extended-recitation data (⏳ queued)

| item | status | headline |
|---|---|---|
| kd_lora_ce_hi_v2 | ✅ | **CER 0.600, 41% exact — matches full-FT champion at 1/4 memory** (forget +1.06 at hi lr) |
| recite_long (hi_v2) | ✅ | whole poem (715 verses): anchored CER 0.396 (19/30 rounds correct), self-chained 0.724 — drift, not missing memory, dominates the gap |
| kd_ce_v2 (40 ep full-FT) | ⏳ waits 11GB window | paraphrases + long windows on full-FT |

## Wave F — compaction axis (⏳ queued, last)

| run | status | headline |
|---|---|---|
| kd_lora_ce_hi_stub | ✅ | 0.791/17% — stub buys nothing over removal |
| kd_lora_ce_hi_stubgap | ✅ | 0.788 but 0.4% exact, forget +2.0 — teacher-geometry imitation actively harmful; **remove is the right default** |

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
- Per-block lens-CE layerwise variant (hybrids have kept failing)
- Two-phase: layerwise pre-conditioning -> short KD polish (needs adapter-resume in kd.py)
- Sequential + online-teacher lockstep cache
- Move to 2×4090: one experiment per GPU (AGENTS.md Tier 1); replicate grid with 2nd seed

## Model ladder (premise-check each with teacher_recite.py before committing GPU time)

| tier | model | unique question it answers |
|---|---|---|
| 3060 | Llama-3.2-1B | cross-family replication (different tokenizer/template) |
| 3060 | SmolLM3-3B | open training data → verify the corpus is truly absent |
| 2×4090 | Qwen3-4B / 8B | localization: positional or proportional depth? memory curve |
| 2×4090 | DeepSeek-V2-Lite (16B MoE) | first MoE: per-expert localization, routing agreement, MLA path |
| 2×4090 | R1-Distill-Qwen-1.5B | thinking-hiding arm with long load-bearing traces |
| 4×L40S | Qwen3-14B / 32B / 30B-A3B | recipe at scale; the serious MoE study |
| 4×L40S | Gemma-3-12B (stretch) | third family; needs sliding-window mask support |
| 4×H100 | GLM / DeepSeek flagship class | Pierre Menard: Don Quijote |
