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

## Speed report (owner request): whole-answer-set times, 0.8B, same 64 items

| path | what it does | wall s | note |
|---|---|---:|---|
| v4 trainer epoch (PPP1) | ALL 24 layers x all answer positions, teacher fwd + training writes + eval | **6.4** | 177,792 token-events; teacher-fwd 2.2 s of it |
| v4 trainer epoch (PPP2/PPP4, per stage) | owned layers only, concurrent stages | 5.6 / 5.2 | wall ≈ slowest stage |
| vLLM greedy GENERATION of the set | autoregressive, ~115 tok/item | 10.7 | 695 tok/s in-engine (+52 s load) |
| standalone verify loop, bf16 B=1 | one full-seq fwd/item, unbatched | 49.6 | naive instrument; NOT the trainer |
| standalone verify loop, fp32 B=1 | same | 6.6 | bf16 B=1 kernel anomaly (7x) |

Reading: the TRAINER traverses the whole answer set — including per-layer
training math vLLM doesn't do — faster than vLLM generates it once (6.4 s vs
10.7 s), because cohort batching + teacher-forcing removes autoregression.
The naive B=1 verify loop is 8x slower than the trainer doing MORE work —
batching, not model math, dominates at this scale.

QUEUED (needs GPU): vLLM-side VERIFICATION timing — feed vLLM
prompt+answer[:-1] with SamplingParams(max_tokens=1, prompt_logprobs=1) and
read per-position argmax from the prefill; that is vLLM's native
teacher-forced mode and the symmetric comparator to our trainer epoch.

## OWNER DECISION (2026-07-19 ~18:20): 0.8B DISCARDED from the envelope

Qwen3.5-0.8B is removed from the vLLM-reproduction goal (and the model
envelope for this claim). Basis: the margin measurement — 27B has ZERO
answer positions within 2 logits of a flip (0/541; p05 4.125, median 10.5)
while 0.8B keeps 15.7% of positions within 0.5 logits (p05 0.125, median
3.375), so the transformers<->vLLM linear-attention noise flips ~1.5% of
0.8B tokens and nothing at campaign scale. 0.8B remains a MECHANICS vehicle
only (bit-identity certs PPP1=PPP2=PPP4, which never involve vLLM).
The goal set is now: 27B (trainer-native EXACT, acceptance 1.0), 35B, 26B,
31B, 122B, 397B (genuine reference pending 2-node vLLM), DeepSeek (blocked
on driver-565 fp4).

## TRAINER-NATIVE rows: 35B / 26B / 31B (2026-07-19 18:30, PPP1, 64 items)

| model | trainer teacher_argmax_acceptance | torch baseline | note |
|---|---|---|---|
| gemma-4-26B-A4B | **0.9960159362549801** | 0.9960159362549801 | IDENTICAL to 16 digits |
| gemma-4-31B | **0.998371335504886** | 0.9983713355 | identical |
| Qwen3.6-35B-A3B | **0.99618** | 0.99809 | DIFFERENT 64-item subset (baseline = first-64 usable responses; matrix = dataset-order match) — re-run at matched subset before reading a delta |
| Qwen3.6-27B | **1.0** | 1.0 | exact |

Machinery-exactness generalizes beyond 0.8B: where the item subset matches
(26B/31B/27B), the trainer's own acceptance equals the standalone walk
bit-for-bit. Lesson recorded: the two agpuh0x "wedges" of the depth probe
were the LUSTRE IMPORT CRAWL amplified by concurrent model loads (SIGINT
stack: charset_normalizer _path_stat) — never launch a cold vllm import
while bulk Lustre reads are in flight.
