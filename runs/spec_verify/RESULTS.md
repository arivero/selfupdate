# Does the training stack reproduce vLLM? — FULL 2071-ITEM EPOCH results

Owner correction (2026-07-19 ~19:10): "2071 is the comparison goal." Every
prior 64-item result was DELETED and is invalid — do not resurrect. This file
starts fresh. Method unchanged (speculative-decoding framing): vLLM's whole
greedy answer is the draft, teacher-forced through our stack (or the actual
v4 trainer), argmax per answer position vs vLLM's token. One full epoch =
2071 items = the whole `data/combined/examples_v5rs_window.jsonl` dataset,
matched 1:1 in order against each model's full vLLM responses file.

Definitions: **token-accept** = matched answer positions / all answer tokens
(length-fair). **exact-seq** = fraction of answers with EVERY token matched
(the literal "exactly" bar; ≈ token-accept^length, dominated by answer
length). **margin** = top1-top2 logit gap at every answer position (≈0 = bf16
tie; large = confident). **depth quartile** = acceptance by within-answer
position (flat = margin story; decaying = accumulation).

## Qwen3.6-27B — torch baseline, FULL 2071 items (2026-07-19 19:26)

70,807 answer tokens (vs 541 at the deleted 64-item scale — 130x more
statistical power).

| metric | value |
|---|---|
| token-accept | **0.99948** |
| exact-seq | 0.98503 |
| first-token agree | 0.99421 |
| n_divergences | 31 (of 70,807) |
| mean_first_div_gap | 0.052 (pure tie) |
| frac_div_in_top2 | 1.0 |
| acceptance by depth quartile | 0.9988 / 0.9997 / 0.9998 / 0.9996 — **FLAT** |
| margin p05 / p50 | 4.375 / 11.19 |
| frac margin < 0.5 | 0.47% |
| forward-only wall time | 880.9s (B=1, unbatched torch loop; NOT the trainer) |

**Reading**: at real statistical power, 27B is NOT "100% perfect" (the deleted
64-item number was a zero-observation artifact of only 541 tokens) — it has a
tiny, genuine divergence rate (0.05%), but every single one is a bf16 tie
(gap 0.05, 100% in our top-2) and depth is completely flat. This is
qualitatively DIFFERENT from 0.8B's confident, depth-independent-margin
divergences (discarded from the envelope, see below) — 27B's residual gap is
pure numerical noise at the tie boundary, not a structural kernel difference.

## 0.8B — OUT OF SCOPE (owner decision, confirmed at 64-item scale)

Margin measurement (0.8B vs 27B, 64 items, before the 2071 pivot) showed
0.8B's divergences are CONFIDENT (gap 5.09, 33% outside top-2, p05 margin
0.125) vs 27B's ties — a real transformers/vLLM linear-attention kernel gap
that only manifests at small-model-scale with long answers. 0.8B is
discarded from this envelope; it keeps its mechanics-vehicle role (PPP
bit-identity certs) which never touches vLLM. Not re-measured at 2071 (no
full-scale 0.8B vLLM generation exists, and it is out of scope).

## Campaign plan (owner, 2026-07-19 ~19:15): PPP4+VLLM4 -> PPP8+VLLM8 -> PPP2/PPP1

1. PPP4 (single-node, 4 cards) + VLLM4 (TP4) trainer-native + vLLM timing,
   for every model that fits: 27B, 35B, 26B, 31B, 122B.
2. Escalate to PPP8+VLLM8 ONLY for models that don't fit at 4 cards: 397B
   (FP8 reference accepted, owner decision), DeepSeek-V4-Flash bf16 (543GB,
   confirmed OOM at TP4 single-node; needs the native mp-backend 2-node vLLM
   path, kv-cache-dtype=fp8 fixes the fp8_ds_mla assertion).
3. ONLY after 1-2 are complete: PPP2 and PPP1 sweep, packed efficiently
   across all 8 cards (4 parallel 2-card jobs, or 1 8-card job, per what's
   being measured).

## In-flight / status (updated live)

- 27B PPP4 trainer-native: LAUNCHED (agpuh01, 4 cards), pending result.
- 26B PPP4 trainer-native: STALLED (confirmed via RSS+cpu-ticks+GPU-mem all
  frozen 40s straight, wchan=hrtimer_nanosleep all 4 stages) — single-node,
  so NOT the cross-node battery ack bug; root cause TBD (store-fill relay
  handshake suspected). Killed; will retry after root-causing.
- 31B PPP4 trainer-native: LAUNCHED (agpuh02, retry after 26B's slot freed).
- 35B PPP4 trainer-native: queued (needs a free 4-card slot).
- 122B PPP4 trainer-native: config fixed (was missing stage_scoped, would
  have repeated the diagnosed 244GB OOM) — queued.
- 122B PPP8 trainer-native (from the pre-pivot 64-item run): 0.9961
  acceptance — INVALID per the 2071 correction, needs re-measurement at full
  scale.
- vLLM prefill-verify / longdraft at TP4, full 2071: failed on
  max_num_seqs=256 > 236 available Mamba cache blocks (27B) — noted, not yet
  patched (lower to <=236 or raise gpu_memory_utilization).

## Known errors this batch (collected, not yet all patched — iterating per
owner instruction: run the batch, note errors, patch once, re-run fails)

1. Cache built on the wrong node (agpuh01 PREP_ONLY, launch targeted
   agpuh02) — FIXED by rebuilding directly on the target node.
2. 122B PPP4 overlay missing stage_scoped/store/subprocess entirely (would
   repeat the diagnosed OOM) — FIXED.
3. 26B PPP4 single-node stall, cause TBD — killed, needs SELFUPDATE_V4_RELAY_DEBUG
   or similar on retry to diagnose (store-fill handshake suspected, not the
   cross-node battery bug since this run never left one node).
4. vLLM max_num_seqs=256 exceeds 27B's Mamba cache capacity (236 blocks) —
   needs a per-model-safe default or a lower fixed value.

## ROOT CAUSE FOUND (19:35): the 26B "stall" was a MASTER_ADDR bug, not a stall

`scripts/launch_v4_stages.sh:134`: `export MASTER_ADDR="${MASTER_ADDR:-$(hostname -s)}"`
— evaluates on whatever host RUNS the launch command, not on the host of
stage 0. Launching an ALL-REMOTE job (every stage on agpuh02) from an agpuh01
shell sets MASTER_ADDR=agpuh01, so the NCCL/TCPStore rendezvous tries to
reach agpuh01:29517 where nothing listens. The processes do NOT hang
immediately — they run for ~9-10 min (weight load, cache checks) before the
store-fill relay's rendezvous attempt fails with
`[c10d] TCP client failed to connect/validate to host agpuh01:29517`, at
which point all 4 stages exit (not deadlock — this looked like a stall only
because I killed them mid-crash before checking the actual log content).
FIX: pass `MASTER_ADDR=<actual-stage0-host>` explicitly whenever the launch
command's own host differs from the stages' host. 26B relaunched correctly
with MASTER_ADDR=agpuh02; 31B (which had the identical latent bug, killed
pre-emptively before it could crash the same way) queued for the same fix.
This was NOT the cross-node battery deadlock (that requires stages split
ACROSS hosts; this run was single-host with a launch-side mismatch only).

## Qwen3.6-27B — PPP4 trainer-native, FULL 2071-item epoch (2026-07-19 19:43)

**teacher_argmax_acceptance = 0.9995056985891225** (70,807 answer tokens,
whole-training-set coverage — `dataset_coverage:
whole_training_set_once_per_completed_epoch`, confirming this is the real
full epoch). student_argmax_acceptance 0.99907 (lr 1e-6 live). CE_eval_loss
0.01366, KL_eval_loss 0.00046.

Matches the torch baseline (0.99948) almost exactly — the trainer's cohort-
batched walk reproduces the standalone sequential walk's fidelity at full
scale, consistent with the earlier bit-identity finding at 0.8B (PPP1=PPP2=PPP4).
27B PPP4: goal met — the training stack reproduces vLLM to ~99.95% at real
statistical power, with the residual entirely bf16 ties (per the torch
baseline's margin/depth analysis above).

## gemma-4-26B-A4B — PPP4 trainer-native, FULL 2071-item epoch (2026-07-19 19:41)

**teacher_argmax_acceptance = 0.9952869328553885** (90,387 answer tokens,
whole-training-set coverage). student_argmax_acceptance 0.99368 (lr 1e-6
live). Full stage-scoped/store/subprocess-battery run, single node, 4 cards.

## gemma-4-31B — PPP4 trainer-native, FULL 2071-item epoch (2026-07-19 19:47)

**teacher_argmax_acceptance = 0.9992767415302627** (78,810 answer tokens,
whole-training-set coverage). student_argmax_acceptance 0.99791.

## Qwen3.6-35B-A3B — PPP4 trainer-native, FULL 2071-item epoch (2026-07-19 19:48)

**teacher_argmax_acceptance = 0.997829486626402** (74,176 answer tokens,
whole-training-set coverage). student_argmax_acceptance 0.99507.

## Phase 1 PPP4 scoreboard so far (trainer-native, full 2071-item epoch)

| model | teacher_argmax_acceptance | answer tokens |
|---|---:|---:|
| Qwen3.6-27B | 0.99951 | 70,807 |
| gemma-4-31B | 0.99928 | 78,810 |
| gemma-4-26B | 0.99529 | 90,387 |
| Qwen3.6-35B-A3B | 0.99783 | 74,176 |
| Qwen3.5-122B | *launching* | |

All four measured models are >=99.5% at real full-epoch scale (70-90K tokens
each) — the goal ("training stack reproduces vLLM") reads as MET for the
single-node envelope, pending 122B and the 8-card escalations (397B, DeepSeek).

## SPEED: one whole 2071-item epoch, trainer-native PPP4 (real measurement)

`epoch_seconds` = the entire epoch (store-fill capture + training write for
all 2071 items), `capture_seconds: 0.0` on every row = the teacher-forward
work is folded into this single epoch measurement (single-epoch runs, no
separate warmup epoch):

| model | epoch_seconds | token_events | train-phase GPU util |
|---|---:|---:|---:|
| Qwen3.6-27B | 109.2 | 1,132,912 | 92.6% |
| gemma-4-31B | 143.1 | 1,182,150 | 89.7% |
| Qwen3.6-35B-A3B | 71.7 | 741,760 | 85.4% |
| gemma-4-26B | 71.2 | 632,709 | 78.4% |

Note: 27B/26B also ran an unwanted post-epoch battery this run (every_epochs
was left at 1, not yet patched to 100 for these two — see the earlier "known
error" entry); that battery cost is OUTSIDE epoch_seconds (separate
subprocess), so these epoch_seconds numbers are clean training-only times.
vLLM4 (TP4) prefill-verify timing for the SAME 2071-item set: in flight
(27B first), to give the direct wall-clock comparator requested.
