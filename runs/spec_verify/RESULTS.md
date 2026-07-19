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
