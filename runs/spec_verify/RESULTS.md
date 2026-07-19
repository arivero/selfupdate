# Does the training stack reproduce vLLM? — teacher-forced acceptance table

**SCOPE CORRECTION (owner, 2026-07-19 ~17:30):** the numbers below are the
**torch/transformers-math baseline** — the standalone `run_block` walk, no
trainer machinery. The GOAL metric must come from the **PPP1/2/4/8 trainers
themselves**: `teacher_output_eval` now logs `teacher_argmax_acceptance` /
`student_argmax_acceptance` vs the answer ids (= the vLLM draft under an
index-only cache), so every v4 run measures its own vLLM reproduction with
the full cache/store/relay machinery in the loop (commit e67d7b0).
Teacher-side is adapter-free → valid at any epoch; a 1-epoch run per model ×
PPP level fills the trainer-native table. This baseline remains the control:
any trainer-native acceptance BELOW this baseline indicts the machinery, not
the math.

Owner goal (2026-07-19): "our training stack reproduces vLLM exactly for all
the models in our project; for large models this implies runs on 4 or 8 GPU
cards." Method (owner): NO generation on our side — vLLM's whole greedy answer
is the draft (speculative-decoding framing) and the **training stack** is the
verifier: one teacher-forced full-sequence forward over `[prompt + vLLM
answer]` through `embed → run_block(1..n) → frozen head` (the exact
`_online_teacher_capture` walk), then per-answer-position argmax vs vLLM's
token. Teacher-forced ⇒ every position is scored independently against vLLM's
true prefix (<b>no cascade</b>); only answer tokens are scored, never the
prompt. Instrument: `scripts/verify_vllm_teacher_forced.py`.

Definitions:
- **token-accept** = matched answer positions / all answer tokens (pooled).
  Length-fair; the honest cross-model number.
- **exact-seq** = fraction of answers with EVERY token matched. The literal
  "reproduces exactly" bar; ≈ token-accept ^ answer-length, so it is DOMINATED
  by answer length (0.985^115 ≈ 0.17 but 0.985^16 ≈ 0.79).
- **gap** = our_argmax_logit − vLLM_token_logit at the first divergence
  (≈0 ⇒ bf16 tie; large ⇒ confident disagreement). **top-2** = fraction of
  divergences where vLLM's token was still our #2.

## Results (64 items each, greedy, bf16, 2026-07-19)

| model | attention | cards | mean ans len | token-accept | exact-seq | first-tok | n_div | gap | top-2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | linear 18 + full 6 | 1 | ~115 | 0.9854 | 0.250 | 0.984 | 48 | **5.09** | **0.67** |
| Qwen3.6-27B | linear 48 + full 16 | 1 | ~8.5 | **1.0000** | **1.000** | 1.000 | 0 | — | — |
| Qwen3.6-35B-A3B | linear MoE | 1 | ~16 | 0.9981 | 0.969 | 1.000 | 2 | 0.19 | 1.00 |
| gemma-4-26B-A4B | full/sliding MoE | 1 | ~12 | 0.9960 | 0.953 | 1.000 | 3 | 3.96 | 1.00 |
| gemma-4-31B | full/sliding | 1 | ~10 | 0.9984 | 0.984 | 1.000 | 1 | — | 1.00 |
| Qwen3.5-122B-A10B | linear MoE | **4** (device_map) | ~16 | 0.9981 | 0.969 | 1.000 | 2 | 0.38 | 1.00 |
| Qwen3.5-397B-A17B | linear MoE | needs 8 (2 nodes) | — | **pending** | | | | | |
| DeepSeek-V4-Flash | MLA MoE | — | — | **blocked**: responses file has no prompt_token_ids; regenerate with ids first | | | | | |

Auxiliary findings (0.8B, same instrument):
- **Self-consistency**: teacher-forcing over OUR OWN autoregressive output
  (HF generate, native hybrid cache — one-off diagnostic, script since removed)
  gave token-accept 0.991 with **gap 0.05 and top-2 = 100%** — pure bf16 ties.
  The training-stack forward is faithful to the model's own generation; the
  teacher hidden targets are sound. The vLLM gap is transformers↔vLLM.
- **fp32 lever (our side only)**: gap 5.09 → 2.14, but first-tok 0.984 → 0.922:
  more precision moves us away from vLLM's bf16 rounding. Matching vLLM needs
  matched precision, not more. (fp32 forward also ran 7× faster than bf16 at
  0.8B B=1 — kernel selection worth a look.)

## TRAINER-NATIVE rows (the goal instrument; 0.8B, 64 items, epoch 1)

`teacher_argmax_acceptance` from the v4 trainer's own `teacher_output_eval`
(index-only cache from the vLLM responses; online teacher; evals off):

| level | stages/GPUs | teacher_argmax_acceptance | student (lr 1e-6 live) |
|---|---|---|---|
| torch baseline | standalone walk | 0.9854211663066955 | — |
| **PPP1** | 1 proc, 1 GPU | **0.9854211663066955** | 0.9750269978401728 |
| **PPP2** | 2 procs, GPUs 0-1 | **0.9854211663066955** | 0.9750269978401728 |
| **PPP4** | 4 procs, GPUs 0-3 | **0.9854211663066955** | 0.9750269978401728 |

**Bit-identical to 16 digits across every level** (7,408 answer tokens).
The v4 machinery — cohorts, index-only cache, online teacher capture, staged
processes, relay — contributes EXACTLY ZERO divergence: the entire residual
gap to vLLM (1.46% at 0.8B) is the transformers↔vLLM kernel difference
characterized above, not our pipeline. PPP8 (cross-node) + the other models'
trainer-native rows: next lease (configs in configs/experiments/spec_verify/).

## Reading

1. **It is NOT the linear attention.** 27B and 35B are linear-hybrid and match
   at 100% / 99.8%. The one poor row (0.8B) is the SMALLEST model with ~10×
   LONGER answers: tiny transformers↔vLLM bf16 differences accumulate with
   answer depth and, in a small model's flatter logit landscape, eventually
   flip confidently (gap 5, a third of vLLM tokens outside our top-2).
2. **Length confound**: exact-seq is not comparable across rows (answer lengths
   differ ~10×). By the length-fair metric every model is ≥ 98.5%, and every
   model but 0.8B is ≥ 99.6% with only near-tie divergences.
3. **Goal status**: met exactly for 27B; near (ties only) for 35B/26B/31B;
   NOT met for 0.8B (confident divergences); 122B/397B multi-card pending.
   A length-matched retest (equal answer budgets across models) is required
   before comparing exact-seq rows.

## Multi-card note (why device_map=auto is legitimate here)

The per-block MATH is the training stack (`run_block`). Multi-card placement
for this read-only verify uses accelerate hooks (`--device-map auto`) — the
same mechanism as every subprocess battery (`scripts/v4_battery.py`). The v4
staged shm/NCCL relay exists for TRAINING's constraints (N concurrent
block-local trainers, stage-scoped memory, cross-node); training steps
themselves need no communication. Transport in both schemes is bit-preserving
boundary `.to(device)` moves; M2 certified staged-vs-single bit-identity.

## Next (397B / the 8-card row)

397B bf16 (752 GB) cannot hold a full copy on one node ⇒ the verify itself
must be TWO half-model processes with a mid-model boundary hand-off — i.e. a
2-stage `_relay_segment` walk over vLLM's tokens (machinery exists; needs a
cohort-from-responses builder). This doubles as the "verify through the
literal staged relay transport" test. Blocked this session on lease time
(~750 GB load alone exceeded the remaining window).
