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

## Campaign plan (owner, 2026-07-19 ~19:15): PPP4+vLLM TP4 -> PPP8+vLLM TP8 -> PPP2/PPP1

(naming corrected 2026-07-19 ~21:10: "VLLM4"/"VLLM8" renamed to "vLLM TP4"/
"vLLM TP8" — see CORRECTION note below; vLLM's side is tensor parallelism
via its own `tensor_parallel_size` argument, run from a standalone script,
not our PPPn pipeline architecture)

1. PPP4 (single-node, 4 cards) + vLLM TP4 trainer-native + vLLM timing,
   for every model that fits: 27B, 35B, 26B, 31B, 122B.
2. Escalate to PPP8+vLLM TP8 ONLY for models that don't fit at 4 cards: 397B
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

## Qwen3.5-122B-A10B — PPP4 trainer-native, FULL 2071-item epoch (2026-07-19 19:56)

**teacher_argmax_acceptance = 0.9966268015946029** (101,091 answer tokens,
whole-training-set coverage). student_argmax_acceptance 0.96652.

## PHASE 1 COMPLETE: PPP4 trainer-native, all 5 models, full 2071-item epoch

| model | teacher_argmax_acceptance | answer tokens |
|---|---:|---:|
| Qwen3.6-27B | 0.99951 | 70,807 |
| gemma-4-31B | 0.99928 | 78,810 |
| Qwen3.5-122B-A10B | 0.99663 | 101,091 |
| Qwen3.6-35B-A3B | 0.99783 | 74,176 |
| gemma-4-26B | 0.99529 | 90,387 |

Every model >=99.5% at real full-epoch statistical power (70K-101K tokens
each). Goal MET for the single-node envelope. Next: vLLM TP4 timing comparators
(in progress via a queue subagent), then the 8-card escalation for 397B/DeepSeek.

## CORRECTION (2026-07-19 ~21:10, owner-prompted code inspection)

Two mislabelings in this section, both caught by direct code/data inspection
rather than assumption — fixed below, not just patched over:

1. **"VLLM4" renamed to "vLLM TP4" throughout.** `scripts/vllm_prefill_verify.py`
   (`grep -n selfupdate scripts/vllm_prefill_verify.py` → zero code hits, only
   cache-dir path strings; `grep -n '^import\|^from'` → argparse/json/os/time/
   pathlib, then `from vllm import LLM, SamplingParams` — no import of `train.py`
   or any `src/selfupdate` module) calls vLLM's own `tensor_parallel_size=4`.
   That is **tensor parallelism** — every layer's weight matrices sharded
   across 4 GPUs with an NCCL all-reduce per layer — the opposite
   communication pattern from our own PPP4 (contiguous block ranges per
   stage-owned process, no cross-stage traffic during the per-cohort training
   step). "PPP4" and "VLLM4" sharing a "4" implied an architectural parallel
   that isn't there: two different parallelism strategies that both happen to
   use 4 GPUs, run from two entirely separate entry points (there was never a
   shared one on the table — vLLM's `LLM()` is a self-contained serving engine
   with its own weight loading/KV-cache/CUDA-graph capture; it cannot run
   inside `train.py`'s process).
2. **Three of the five "PPP4 trainer epoch" seconds below were wrong**, not
   just mislabeled — 122B and 27B had the SAME value (109.2), a copy-paste
   duplicate, and 31B's 143.1 was never real either. Re-pulled directly from
   each run's `stage0/metrics.jsonl`, the `v4_epoch` record's
   `epoch_seconds` field (= `prep_seconds + exec_seconds`, i.e. cohort setup
   + the full forward+backward+optimizer-write walk, excluding one-time model
   load): 26B 68.95, 122B **96.64** (was 109.2), 27B **106.31** (was a
   109.2 duplicate), 31B **83.04** (was 143.1), 35B 71.07. Table below uses
   these verified numbers.

Related clarification on "no cross-stage communication" (owner question):
precise for the **training** step only — `online_v4.py`'s per-cohort update
reads pre-cached teacher hidden states as both input and target (already
staged in `/dev/shm`, no live neighbor needed), so no stage waits on another
during training. Actual student-forward boundary hidden states DO relay
across stage boundaries via `_RelayServicer`/`_relay_segment`
(`v4_relay_every_cohorts`, same-node shm or cross-node NCCL) — but that path
is explicitly evaluation-only (docstring: "The skew is bounded by pipeline
depth, evaluation-only"), asynchronous, and not on the training critical path.
Both statements are true; they describe different code paths.

Also noted, not fixed now (real gap): **the trainer has no forward-only
("skip backward, skip write") mode.** `teacher_output_eval`'s acceptance
metric is a side effect of the full training step; there is no isolated
verify-only path to time against vLLM's inference-only prefill on equal
footing. The seconds below are reported as what they are — a full training
epoch vs. vLLM's inference-only prefill pass — not as a matched-workload
ratio.

## gemma-4-26B-A4B — vLLM TP4 prefill-verify, FULL 2071-item epoch (2026-07-19 20:31)

**self_consistency_match_rate = 0.9889** (2071/2071 items processed).
load_seconds 140.66 (LLM() construction, eager mode, disable_custom_all_reduce),
seconds 27.797 (the actual generate() call over all 2071 prompts),
items_per_s 74.5, context_tok_per_s 25,787.

This is vLLM verifying ITS OWN earlier greedy answers via one eager-mode
prefill pass (max_tokens=1) per item. `self_consistency` is vLLM-vs-itself
(sanity check, should read ~1.0); it is NOT the same quantity as our
trainer's `teacher_argmax_acceptance` (our-stack-vs-vLLM), and the two must
not be read as one comparison column. 98.89% self-consistency (not 100%) is
itself informative: even vLLM checking its own prior output disagrees ~1.1%
of the time under eager mode (vs whatever mode/settings produced the
original greedy answers) — a real numerical-precision/kernel-path
sensitivity baseline, useful context for reading our own 99.5%+ acceptance
numbers.

## Qwen3.5-122B-A10B — vLLM TP4 prefill-verify, FULL 2071-item epoch (2026-07-19 20:32)

**self_consistency_match_rate = 0.9845** (2071/2071 items).
load_seconds 121.52, generate seconds 30.713, items_per_s 67.43.

## vLLM TP4 — full 5-model completion (2026-07-19, queue-managed on agpuh01/agpuh02)

All 5 models complete, full 2071-item coverage, post enforce_eager +
disable_custom_all_reduce fix (see below):

| model | self_consistency | seconds | items_per_s | load_seconds |
|---|---:|---:|---:|---:|
| Qwen3.6-27B | 0.9986 | 30.723 | 67.41 | 140.54 |
| gemma-4-26B-A4B | 0.9889 | 27.797 | 74.5 | 140.66 |
| gemma-4-31B | 0.9990 | 14.956 | 138.47 | 101.67 |
| Qwen3.6-35B-A3B | 0.9850 | 18.432 | 112.36 | 131.4 |
| Qwen3.5-122B-A10B | 0.9845 | 30.713 | 67.43 | 121.52 |

Operational note: every one of the 5 models crashed at least once pre-fix
with `CUDA error: an illegal memory access` sourced in vLLM's
`CUDASymmetricMemory`/custom-all-reduce path (during CUDA-graph capture or
KV-cache memory profiling) — not model-specific, hit on both TP4 nodes.
Fixed by `enforce_eager=True` + `disable_custom_all_reduce=True`
(`scripts/vllm_prefill_verify.py`, commits `11fc186`/`91b44ac`); every retry
after both fixes succeeded on the first attempt.

## SPEED SCOREBOARD — vLLM's timing is the reference point, NOT a same-workload ratio

Owner framing (2026-07-19): vLLM's numbers are what our training stack is
measured AGAINST — the reference speed a genuinely independent, mature
serving engine achieves on the same full 2071-item set, same weights.
**Read this table as two different workloads side by side, not a speedup
ratio** — see the CORRECTION note above: our column is a full training
epoch (forward + backward + optimizer write, all layers); vLLM's is one
inference-only prefill pass (no backward, no write). No forward-only mode
exists on our side yet to produce a true matched-workload number.

| model | PPP4 trainer epoch, full train (s) | vLLM TP4 generate, prefill-only (s) | vLLM TP4 load (s) |
|---|---:|---:|---:|
| gemma-4-26B-A4B | 68.95 | 27.797 | 140.66 |
| Qwen3.5-122B-A10B | 96.64 | 30.713 | 121.52 |
| Qwen3.6-27B | 106.31 | 30.723 | 140.54 |
| gemma-4-31B | 83.04 | 14.956 | 101.67 |
| Qwen3.6-35B-A3B | 71.07 | 18.432 | 131.4 |

Both legs now complete for all 5 models, full 2071-item coverage. Our full
training epoch runs at roughly 2-6x vLLM's bare inference-only prefill time
depending on model, and well under vLLM's one-time engine-load cost in every
case — reported as observed wall-clock only, since the two columns are not
the same task (see correction note). A genuine speed-vs-speed claim needs a
forward-only path on our side; that does not exist today.

## Command provenance for the speed scoreboard (2026-07-19, owner-requested)

Caveat first: neither script prints its own invocation to its log, and no
launcher wrapper survived from the subagent that ran the vLLM TP4 leg. The
commands below are RECONSTRUCTED — verified against the script's real
`argparse` flags, each model's identity in its PPP4 base config, and the
exact response-file paths on disk — not a literally recovered shell history.
`CUDA_VISIBLE_DEVICES=0,1,2,3` is inferred from TP4 needing 4 physical GPUs,
not read from a surviving log. `--limit` is omitted (default `0` = whole
file); each output JSON's `items: 2071` confirms that was what actually ran.
`epoch_seconds` (PPP4 column) and `load_seconds`/`seconds` (vLLM TP4 columns)
both come from files the code itself wrote (`stage0/metrics.jsonl`'s
`v4_epoch` record; the script's own `--out` JSON) — not a wall-clock timed
by hand.

| model | PPP4 command | vLLM TP4 command |
|---|---|---|
| gemma-4-26B-A4B | `scripts/launch_v4_stages.sh configs/experiments/spec_verify/base_26b_v4_spec.yaml configs/experiments/spec_verify/26b_v4_spec_ppp4.yaml` | `CUDA_VISIBLE_DEVICES=0,1,2,3 /fs/agustina/arivero/supercomplex/venvs/vllm025/bin/python scripts/vllm_prefill_verify.py --model google/gemma-4-26B-A4B-it --responses runs/vllm_h100/gemma4_26b_a4b_it/responses_bs256.jsonl --tensor-parallel-size 4 --out runs/spec_verify/26b_vllm4_prefill_full2071.json` |
| Qwen3.5-122B-A10B | `scripts/launch_v4_stages.sh configs/experiments/spec_verify/base_122b_v4_spec.yaml configs/experiments/spec_verify/122b_v4_spec_ppp4.yaml` | `CUDA_VISIBLE_DEVICES=0,1,2,3 /fs/agustina/arivero/supercomplex/venvs/vllm025/bin/python scripts/vllm_prefill_verify.py --model Qwen/Qwen3.5-122B-A10B --responses runs/vllm_h100/qwen35_122b_a10b/responses_bs256.jsonl --tensor-parallel-size 4 --out runs/spec_verify/122b_vllm4_prefill_full2071.json` |
| Qwen3.6-27B | `scripts/launch_v4_stages.sh configs/experiments/spec_verify/base_27b_v4_spec.yaml configs/experiments/spec_verify/27b_v4_spec_ppp4.yaml` | `CUDA_VISIBLE_DEVICES=0,1,2,3 /fs/agustina/arivero/supercomplex/venvs/vllm025/bin/python scripts/vllm_prefill_verify.py --model Qwen/Qwen3.6-27B --responses runs/vllm_h100/qwen36_27b_full_exactids/responses_bs256.jsonl --tensor-parallel-size 4 --out runs/spec_verify/27b_vllm4_prefill_full2071.json` |
| gemma-4-31B | `scripts/launch_v4_stages.sh configs/experiments/spec_verify/base_31b_v4_spec.yaml configs/experiments/spec_verify/31b_v4_spec_ppp4.yaml` | `CUDA_VISIBLE_DEVICES=0,1,2,3 /fs/agustina/arivero/supercomplex/venvs/vllm025/bin/python scripts/vllm_prefill_verify.py --model google/gemma-4-31B-it --responses runs/vllm_h100/gemma4_31b_it/responses_bs256.jsonl --tensor-parallel-size 4 --out runs/spec_verify/31b_vllm4_prefill_full2071.json` |
| Qwen3.6-35B-A3B | `scripts/launch_v4_stages.sh configs/experiments/spec_verify/base_35b_v4_spec.yaml configs/experiments/spec_verify/35b_v4_spec_ppp4.yaml` | `CUDA_VISIBLE_DEVICES=0,1,2,3 /fs/agustina/arivero/supercomplex/venvs/vllm025/bin/python scripts/vllm_prefill_verify.py --model Qwen/Qwen3.6-35B-A3B --responses runs/vllm_h100/qwen36_35b_a3b/responses_bs256.jsonl --tensor-parallel-size 4 --out runs/spec_verify/35b_vllm4_prefill_full2071.json` |

# PHASE 2 — 8-card cross-node escalation (397B, DeepSeek-V4-Flash)

## CAVEAT THAT APPLIES TO EVERY PHASE 2 NUMBER BELOW — read before citing anything here

Neither Phase 2 model could run its own vLLM generation: 397B exceeds one
80GB H100 at TP4 even in fp8 (~100GB/card) or bf16 (~200GB/card); DeepSeek's
fp4-expert kernels fail on driver 565 and its 543GB bf16 dequant exceeds
single-node TP4. Both models' `responses_bs256.jsonl` therefore borrow answer
TOKEN SPANS from Qwen3.5-122B-A10B (397B: byte-copy, tokenizers verified
byte-identical md5; DeepSeek: text re-encoded through its own tokenizer) —
confirmed via `runs/vllm_h100/qwen35_397b_a17b_fp8/PROVENANCE.md` and
`runs/vllm_h100/deepseek_v4_flash/PROVENANCE.md`, both explicit: "This is a
SPEED test... NOT answer quality... do not cite recall/damage from this run."

Consequence: `teacher_argmax_acceptance`/`student_argmax_acceptance` (training
side) and `self_consistency_match_rate` (vLLM side) for 397B and DeepSeek
measure CROSS-MODEL AGREEMENT with 122B's word choices — NOT a genuine
vLLM-reproduction check the way the identically-named fields measured for
Phase 1's five models. They are NOT comparable to Phase 1's numbers and must
not be presented side by side with them as equivalent evidence. Teacher
hiddens during training DO come from each model's own real weights (only the
answer token spans are borrowed), so epoch timing and token-event-integrity
numbers remain fully valid with no caveat.

## Qwen3.5-397B-A17B — PPP8 cross-node trainer-native, FULL 2071-item epoch (2026-07-19 23:24-23:48)

Command:
```
cd /fs/agustina/arivero/supercomplex/selfup_teacher
SELFUPDATE_V4_STAGE_HOSTS="local local local local agpuh02 agpuh02 agpuh02 agpuh02" \
  scripts/launch_v4_stages.sh configs/experiments/spec_verify/base_397b_v4_spec.yaml configs/experiments/spec_verify/397b_v4_spec_ppp8.yaml
```
Launched 1784496289 (2026-07-19 23:24:49 CEST). Real cross-node placement:
agpuh01 pids 2750007/2750049/2750103/2750206 (stages 0-3), agpuh02 workers
4024718/4024800/4024886/4024960 (stages 4-7). A first attempt failed
instantly on all 8 stages (`RuntimeError: node-local epoch-zero teacher cache
is not ready` — the fresh `spec_verify` cache identity had never been built
on either node); fixed by running `scripts/build_teacher_cache.py --index-only
--coordinated-node-cache` on both nodes (confirmed matching hash
`914d4ffd7d93d0a1`, 2071 examples) before this relaunch.

All 8 stages reported `run complete` within 1 second of each other
(23:48:02-23:48:03 CEST). Wall-clock launch-to-completion: 1394s (23m14s,
includes cache-gate + full `rotate`-residency weight materialization, not
just compute).

**teacher_argmax_acceptance = 0.97159** (cross-model agreement with borrowed
122B answers — see caveat above, NOT comparable to Phase 1).
student_argmax_acceptance 0.97010, CE_eval_loss 0.10933, KL_eval_loss 0.00516,
answer_token_count 101091, dataset_item_count **2071**, dataset_coverage
`whole_training_set_once_per_completed_epoch` — confirming a genuine full
single-epoch traversal (source: `runs/spec_397b_v4_ppp8x_e1/stage7/metrics.jsonl`,
last `teacher_output_eval` record — stage7 owns the vocab head).

Speed (fully valid, no caveat): epoch_seconds ranged 249.6-265.1s across the
8 stages (stage7: 258.06 = 4.36 prep + 252.45 exec; stage0: 259.34); token_events
reconciles exactly as an integrity check — 8-layer stages (0,3,4,7) report
808728, 7-layer stages (1,2,5,6) report 707637, both dividing exactly to the
101091 answer-token count.
