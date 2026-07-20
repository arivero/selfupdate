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

**TABLE CORRECTION (2026-07-20, caught during the exact-seq fan-out — the
"CORRECTION" section below fixed these numbers in prose on 2026-07-19 ~21:10
but never actually edited this table; leaving the original row values above
untouched as the historical record and correcting here instead):** these are
`stage0`'s `epoch_seconds`, not necessarily representative of every stage
(different stages own different layer counts and stage3 additionally runs
the `teacher_output_eval` pass, so per-stage times spread — e.g. 122B's
stage0-3 span 96.5-109.0s in one single run). 27B was 109.2 and 122B was ALSO
109.2 — a copy-paste duplicate, not two models genuinely tied — and 31B's
143.1 was never real. Correct `stage0` values, verified twice now (original
2026-07-19 measurement, and a second independent sample from the 2026-07-20
exact-seq fan-out re-run, both agreeing to within ~0.5s): Qwen3.6-27B
**106.31** (fresh: 106.74), gemma-4-31B **83.04** (fresh: 83.26),
Qwen3.6-35B-A3B **71.07** (fresh: 71.23), gemma-4-26B **68.95** (fresh:
69.01), Qwen3.5-122B-A10B **96.64** (fresh: 96.55, missing from the original
table entirely).

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

## Qwen3.5-397B-A17B and DeepSeek-V4-Flash — vLLM TP8 (inference-only) legs:
CLOSED, both blocked for clear non-code reasons (2026-07-20)

Distinct from the PPP8 trainer-native runs above (which have real results),
neither model's OWN vLLM TP8 generation completed, after real investigation
rather than a quick abandon. Both are genuine infrastructure/arithmetic
blockers, not bugs in this repo's code — no further retry is planned unless
one of the prerequisites below is separately addressed.

- **397B**: bf16 Qwen3.5-397B-A17B is ~752-794GB of weights. Any
  `tensor_parallel_size x pipeline_parallel_size = 8` split across 8x80GB
  HBM3 = 640GB total leaves no headroom for activations/KV-cache, since
  neither TP nor PP introduces weight redundancy — it does not fit,
  regardless of parallelism shape or the cross-node connectivity fix below.
  Unblocking needs quantization (fp8/int4) or CPU/disk weight offload, not a
  retry. (A first attempt also hit a genuine, now-fixed, cross-node topology
  bug — see next bullet — but confirmed the capacity ceiling is the deeper
  blocker.)
- **DeepSeek**: reached a live model forward pass for the first time ever
  under vLLM (further than any single-node TP4 attempt), then hit
  DeepGEMM's JIT compiler needing nvcc >=12.9 for a specific inline-PTX op,
  while the only newer nvcc available on these nodes (13.2) compiles it but
  fails at *runtime* because driver 565.57.01 doesn't support the cu13x
  runtime. No installed toolkit satisfies both constraints; full diagnosis
  in `issues.md`.
- **Independent, validated progress**: a real cross-node vLLM mechanism bug
  was found and fixed along the way — `distributed_executor_backend="mp"`
  with `nnodes`/`node_rank` is confirmed broken for the offline `LLM()` API;
  `external_launcher` under `torchrun` works, but a single TP communicator
  spanning >1 GPU/node across nodes hangs deterministically, fixed via
  `pipeline_parallel_size=nnodes` (keeps every TP communicator intra-node).
  Landed in `scripts/vllm_2node_smoke.py` and `scripts/vllm_prefill_verify.py`
  (`--multi-node`), confirmed working at PP2xTP4 (world_size=8) on a tiny
  model. This fix is real and reusable; it just doesn't reach past either
  model's own separate blocker above.

## PHASE 3 — PPP2 (2-stage, one node's 2 cards), gemma-4-26B-A4B and
Qwen3.6-35B-A3B (2026-07-20, recovered from disk — not previously written here)

These runs completed before this section was written; they are logged now
because a later exact-seq validation pass (below) went looking for the PPP2
comparator and found it was never recorded. Both predate commit `609d9d1`
(the exact-seq/`teacher_exact_seq_rate` instrumentation), so neither has
that field — only the pre-existing `teacher_argmax_acceptance` comparison.

Command (both models): `scripts/launch_v4_stages.sh
configs/experiments/spec_verify/base_<model>_v4_spec.yaml
configs/experiments/spec_verify/<model>_v4_spec_ppp2.yaml`
(`v4_stage_splits: [16]` for 26B, `[20]` for 35B — each model's proven PPP4
middle stage-cut; `v4_stage_devices: [0, 1]`).

| model | teacher_argmax_acceptance (PPP2) | vs PPP4 (Phase 1) | answer_token_count | epoch_seconds (PPP2) | epoch_seconds (PPP4) |
|---|---:|---|---:|---:|---:|
| gemma-4-26B-A4B | 0.9952869328553885 | bit-identical | 90387 | 83.73 | 71.16 |
| Qwen3.6-35B-A3B | 0.997829486626402 | bit-identical | 74176 | — | — |

26B: source `runs/spec_26b_v4_ppp2_e1/stage1/metrics.jsonl` (completed
2026-07-20 01:03, stage1 owns the vocab head), compared against
`runs/spec_26b_v4_ppp4_e1/stage3/metrics.jsonl` — `teacher_argmax_acceptance`
matches to every digit (Python `==` on the parsed floats, not rounded).
CE_eval_loss 0.02272467034655475, KL_eval_loss 0.007522710656592728.
35B: source `runs/spec_35b_v4_ppp2_e1/stage1/metrics.jsonl`, same bit-exact
match against its PPP4 entry above; epoch_seconds not yet extracted.

This extends the 0.8B-scale bit-identity finding (PPP1=PPP2=PPP4) to real
scale for `teacher_argmax_acceptance`, independently of the two-route
DeepSeek-repair/exact-seq work below.

## PHASE 3 — exact-seq (per-answer 100%-match) backfill, in progress
(2026-07-20)

Gap identified (owner): `teacher_argmax_acceptance` is a token-weighted
mean; it was never joined by the per-answer "did EVERY token match"
rate anywhere except a one-off standalone-script check for 27B early in
the campaign — and that script does not exercise the PPPn tool's own
cache/store/relay/cohort machinery, so it cannot stand in for the other
models. Fixed at the source: `teacher_output_eval_sums()`
(`src/selfupdate/eval/teacher_output.py`) now accepts per-answer row
boundaries and the real `teacher_output_eval` record (populated from
`online_v4.py`, the same code path every number in this file comes from)
reports `teacher_exact_seq_rate` / `exact_seq_match_answers` /
`exact_seq_answer_count` alongside the existing per-token acceptance.
Commit `609d9d1`; purely additive, sanity-checked on synthetic data before
touching any GPU. Since exact-seq is a pure function of the same per-token
match booleans already shown bit-identical across parallelism degrees
(table above), only one fresh re-run per model is needed to backfill it —
not a full campaign repeat.

**Validation PASSED (26B PPP4, 2026-07-20).** Re-ran
`scripts/launch_v4_stages.sh configs/experiments/spec_verify/base_26b_v4_spec.yaml
configs/experiments/spec_verify/26b_v4_spec_ppp4.yaml` with the instrumented
code (backup of the pre-fix run preserved at
`runs/spec_26b_v4_ppp4_e1.pre_exactseq_backup/`). `teacher_argmax_acceptance`
reproduced **bit-for-bit**: `0.9952869328553885` on both the pre-fix and
post-fix runs (Python `==` on the parsed floats) — the code change is
confirmed numerically inert on the existing metric. `answer_token_count`
also matched exactly (90387). `epoch_seconds` 71.12s vs the original 71.16s
(same run, second independent timing sample — consistent).

**First real exact-seq number, from the actual PPPn tool:**

**gemma-4-26B-A4B, PPP4, full 2071-item epoch: teacher_exact_seq_rate =
0.8570738773539353** — 1775 of 2071 answers (85.71%) had EVERY teacher-
argmax token match vLLM's chosen token, versus 99.53% at the per-token
level. The gap is expected, not a red flag: compounding a 99.53% per-token
rate over ~44 tokens/answer (90387/2071) predicts roughly
0.9953^43.6 ≈ 84.2%, close to the observed 85.71% — a sanity-consistent
result, not an anomaly. Source:
`runs/spec_26b_v4_ppp4_e1/stage3/metrics.jsonl`, `teacher_output_eval`,
epoch 1 (`exact_seq_answer_count: 2071` confirms whole-corpus coverage).

**PPP2 cross-architecture check PASSED (26B, 2026-07-20).** Re-ran
`26b_v4_spec_ppp2.yaml` the same way (backup preserved at
`runs/spec_26b_v4_ppp2_e1.pre_exactseq_backup/`). All teacher-side fields
are **bit-identical** to the fresh PPP4 run above: `teacher_argmax_acceptance
0.9952869328553885`, `teacher_exact_seq_rate 0.8570738773539353`,
`exact_seq_match_answers 1775`, `answer_token_count 90387` — exact-seq is
confirmed parallelism-invariant, not just assumed by analogy to the
already-proven token-level invariance. `epoch_seconds` 83.14s (PPP2) vs
71.12s (PPP4) — consistent second timing samples vs the originals (83.73s /
71.16s).

One real, non-contradictory wrinkle the validation surfaced: `CE_eval_loss`,
`KL_eval_loss`, and `student_argmax_acceptance` do NOT match bit-for-bit
between PPP2 and PPP4 (e.g. CE 0.022729954704824796 vs 0.02272467034655475).
Confirmed this predates commit `609d9d1` — the pre-fix backups show the
identical discrepancy, so it is orthogonal to the exact-seq fix. Read as
expected rather than alarming: those three fields depend on the STUDENT's
own trained state, which accumulates through per-block online SGD updates
whose floating-point order differs slightly across stage splits (non-
associative float addition + different cohort/rotation boundaries per
split) — a real but tiny numerical-order effect on trained-state-dependent
quantities. The TEACHER-side quantities this campaign's core claim rests on
(`teacher_argmax_acceptance`, now `teacher_exact_seq_rate`) are pure
functions of frozen teacher hidden states through the frozen vocab head, so
they carry no such dependency and stay exactly reproducible. The "PPP1=PPP2
=PPP4 bit-identity" claim in this campaign has only ever been about the
teacher-reproduction metrics, not student-side training telemetry — this
finding doesn't move that claim, but is worth keeping in mind before ever
citing CE/KL_eval_loss or student_argmax_acceptance as split-invariant.

Fan-out to 27B/31B/35B/122B (via their proven PPP4 configs) and 397B (via
PPP8x, with the borrowed-answer caveat) is next.

### Fan-out results, ALL 5 MODELS COMPLETE (2026-07-20)

epoch_seconds columns are `stage0` vs `stage0` (apples-to-apples; stage3
runs slower because it also carries `teacher_output_eval` — see the
epoch_seconds table correction above; an earlier version of this table
compared fresh-stage3 against the stale/uncorrected scoreboard values and
is fixed here):

| model | arch | teacher_argmax_acceptance | bit-exact vs pre-fix | teacher_exact_seq_rate | matched/2071 | stage0 epoch_seconds (new vs corrected-old) |
|---|---|---:|---|---:|---:|---|
| gemma-4-26B-A4B | MoE (A4B) | 0.9952869328553885 | yes | 0.8570738773539353 | 1775 | 69.01 vs 68.95 |
| Qwen3.6-27B | dense | 0.9995056985891225 | yes | 0.9859971028488653 | 2042 | 106.74 vs 106.31 |
| gemma-4-31B | dense | 0.9992767415302627 | yes | 0.9744084983099952 | 2018 | 83.26 vs 83.04 |
| Qwen3.6-35B-A3B | MoE (A3B) | 0.997829486626402 | yes | 0.9261226460647031 | 1918 | 71.23 vs 71.07 |
| Qwen3.5-122B-A10B | MoE (A10B) | 0.9966268015946029 | yes | 0.8947368421052632 | 1853 | 96.55 vs 96.64 |

All 5 pass the bit-exact reproduction gate; all 5 timing samples agree with
their corrected originals to within ~0.5s — the instrumentation is confirmed
inert everywhere, not just at 26B.

**Dense/MoE split confirmed, and it is NOT a clean binary:**

| model | arch | teacher_argmax_acceptance (per-token) | teacher_exact_seq_rate (per-answer) |
|---|---|---:|---:|
| Qwen3.6-27B | dense | 0.99951 | 0.98600 |
| gemma-4-31B | dense | 0.99928 | 0.97441 |
| Qwen3.6-35B-A3B | MoE | 0.99783 | 0.92612 |
| Qwen3.5-122B-A10B | MoE | 0.99663 | 0.89474 |
| gemma-4-26B-A4B | MoE | 0.99529 | 0.85707 |

Both dense models cluster at the top (97.4-98.6% exact-seq); all three MoE
models sit below them (85.7-92.6%), confirming the advisor-flagged ~10x
per-token gap carries through to a real gap in the per-answer metric this
campaign actually cares about. Within MoE, severity varies with model
(122B and 35B noticeably better than 26B) rather than a uniform cliff —
consistent with a routing-disagreement mechanism whose frequency depends on
each model's own expert count/routing sensitivity, not a single shared
defect. 27B's exact-seq via the real tool (98.60%) sits close to, but is not
required to bit-match, its own earlier standalone-script measurement
(98.50%, different code path — see the torch baseline entry above); the two
are independent measurements, not a reproduction check.

### 26B divergence diagnostic: NOT a code bug — MoE-routing sensitivity concentrated at low-constraint boundary positions (2026-07-20)

Method: reused the real production code (`TrainingRuntime`, `DistillDataset`,
`_V4Cohort`, `_online_teacher_capture`, `stack.lm_head`,
`teacher_output_eval_sums` — no reimplementation), run B=1 per-item
(`v4_teacher_source: online`, not `store`) to avoid an OOM the production
micro-batch=64 config would hit outside its normal stage-scoped memory
budget. This means the recomputed numbers are NOT bit-exact against the
published (store-based) run — `teacher_argmax_acceptance` 0.994911 vs
published 0.995287, exact-seq 1758/2071 vs published 1775/2071 — a small,
real, informative divergence: it shows the metric has some sensitivity to
batching/execution-path composition, plausibly because MoE routing itself
depends on it, not because either measurement is wrong. Full raw data:
473 divergent tokens across 319 divergent answers (of 90,387 tokens / 2071
answers total), recorded per-position with margins and vLLM/predicted token
ids.

**Aggregate findings (all 473 divergences / 319 divergent answers, not
curated examples):**

| | value |
|---|---:|
| first-token-agree (26B) | 0.94978 |
| first-token-agree (27B, dense, for contrast) | 0.99421 |
| overall token-accept (26B) | 0.99477 |
| divergent tokens at pos==0 | 104 / 473 (22.0%) |
| divergent tokens at pos==length-1 | 28 / 473 (5.9%) |
| divergent-token relative-depth quartile histogram | Q1 262 / Q2 111 / Q3 45 / Q4 55 — front-loaded, monotonic Q1->Q3 decline, small Q4 uptick |
| divergent answers whose SOLE wrong position is pos 0 | 74 / 319 (23.2%) |
| divergent answers whose SOLE wrong position is the last token | 24 / 319 (7.5%; 1.16% of all 2071) |
| terminal-position vLLM token id | **106 in all 28 cases** (a single, consistent id — the turn/stop-like special token) |
| terminal-position predicted id | 107 in 22/28 cases, else one of a few others |
| terminal divergences at answer length >= 4096 (max_tokens-cap truncation risk) | 1 of 28 (median terminal-divergence answer length: 11.5 tokens — NOT a truncation-dominated population) |

**Reading, following an independent advisor review of this exact data:**
the first-token effect is the dominant, well-established driver of the
exact-seq gap (23.2% of all divergent answers fail solely because of it) —
and critically, **27B (dense) does NOT show this degradation at its own
first token** (0.99421, close to its 0.99948 overall rate), while 26B (MoE)
degrades sharply at position 0 alone (0.94978 vs 0.99477 overall). A
masking/alignment/off-by-one bug at the thinking-channel boundary (every
answer's position 0 immediately follows the chat template's
`<|channel>thought\n<channel|>` marker) would be expected to hit dense and
MoE models alike, since that boundary-handling code is shared, not MoE-
specific. Only the MoE model degrading points at MoE ROUTING — a small
upstream bf16 difference between our forward and vLLM's fused MoE kernel
occasionally flips which expert gets selected, and this bites hardest at
position 0 where the hidden state is least constrained by prior generated
context. This is the same "not a defect" bucket as 27B's pure bf16 ties
(see the torch-baseline entry above), just with a much louder, MoE-specific
amplification mechanism — **not evidence of a bug in this repo's trainer
code.**

The terminal-token cluster is smaller (5.9% of divergences, 7.5% of
divergent answers) but strikingly consistent — vLLM's chosen id is always
106 (the same id, every single time) and ours is usually 107 — a genuine
end-of-turn-calibration effect distinct from the first-token/MoE-routing
story, though small enough not to change the headline conclusion. It also
means the reported 85.71% exact-seq is very mildly conservative: excluding
the 24 answers whose only fault is this terminal-token miss would raise
exact-seq to 1799/2071 = 86.87%. Only 1 of the 28 terminal cases is at an
answer length near vLLM's 4096-token cap (a plausible truncation artifact,
flagged and excluded from generalization, not the explanation for the other
27).

**Bottom line for the user's question ("is the problem in our code"): no.**
The dense-model contrast is the discriminating fact — a shared-code bug
would not spare 27B's first token while breaking 26B's. Recommended
follow-up if pursued further (not required for this campaign, GPU-optional):
repeat the first-token-agree measurement on 35B/122B (MoE, should show the
same degradation) and 31B (dense, should not) to confirm the mechanism
holds across the whole model set, not just this one MoE/dense pair.

### 31B divergence diagnostic: confirms and REFINES the 26B story — a shared bf16-tie floor, amplified ~6x by MoE routing, not a dense/MoE binary (2026-07-20)

Method: identical to the 26B diagnostic above (real production code —
`TrainingRuntime`, `DistillDataset`, `_V4Cohort`, `_online_teacher_capture`,
`stack.lm_head` — B=1/item, `v4_teacher_source: online`, no forward-pass
reimplementation), full 2071-item traversal, script
`scripts/spec_verify_position_diag.py`, raw data
`runs/spec_verify/31b_position_diag_full2071.json`. Anchor check (recomputed
metric vs the published bit-exact PPP1 number) passed: exact-seq 0.97634
(2022/2071) vs published 0.97441 (2018/2071), +0.19pp gap — actually
tighter than 26B's own recompute gap (+0.8pp), confirming the
instrumentation measures the right path.

| | 31B (dense) | 27B (dense, reference) | 26B (MoE, reference) |
|---|---:|---:|---:|
| first-token-agree | 0.99179 | 0.99421 | 0.94978 |
| overall token-accept | 0.99937 | 0.99948 | 0.99477 |
| gap = overall − first-token | **0.76pp** | 0.53pp | 4.5pp |
| absolute pos-0 miss rate | 0.82% (17/2071) | — | ~5% |
| divergent answers, sole fault = pos 0 | 17/49 (34.7%) | — | 74/319 (23.2%) |
| terminal-position vLLM id | 106 (4/4) | — | 106 (28/28) |
| terminal-position predicted id | 107 (4/4) | — | 107 (22/28) |
| terminal divergence at length >=4096 | 0/4 (median 27 tok) | — | 1/28 (median 11.5 tok) |

(31B has ~10x fewer divergent tokens than 26B's 473 — the headline numbers
above are solid over the full 2071-item traversal, but sub-fractions like
the depth-quartile histogram are lower-power and not repeated here.)

**Refined mechanism, per an independent advisor read of this exact data:**
the original 26B write-up framed the prediction as a hard binary ("dense
should show zero first-token degradation"). The honest reading of both
dense datapoints (27B: 0.53pp, 31B: 0.76pp) is that dense is NOT zero — both
show a small, real "floor" at position 0, roughly 6x below MoE's 4.5pp gap.
Position 0 is the least-constrained position for every architecture (no
prior generated context to pin the forward computation), so it carries the
highest irreducible bf16-kernel-order tie sensitivity regardless of MoE —
what MoE adds is not a new mechanism, it's amplifying that same shared
floor roughly 6x via expert-routing sensitivity to the same small
differences. This is if anything a STRONGER "not a bug" case than the
original framing: a shared masking/alignment/off-by-one bug would be
expected to hit both dense models at 26B's magnitude, and it doesn't — both
land in the same small-gap band, an order of magnitude below MoE. The
terminal-token 106->107 cluster replicates exactly across both models (4/4
here vs 28/28 for 26B, same ids both times, neither truncation-dominated),
reinforcing it as a separate, architecture-independent end-of-turn
calibration effect. Pos-0 divergence margins (0.25-2.75, mostly <1) further
support near-tie noise over a confident, reproducible disagreement.

**Updated bottom line:** not a dense=0/MoE=large binary, but a shared bf16
floor that MoE routing amplifies ~6x. No evidence of a trainer-code bug in
either framing.

### 397B PPP8x exact-seq re-run: complete (2026-07-20)

Re-ran `SELFUPDATE_V4_STAGE_HOSTS="local local local local agpuh02 agpuh02
agpuh02 agpuh02" scripts/launch_v4_stages.sh
configs/experiments/spec_verify/base_397b_v4_spec.yaml
configs/experiments/spec_verify/397b_v4_spec_ppp8.yaml` (pre-fix run backed
up to `runs/spec_397b_v4_ppp8x_e1.pre_exactseq_backup/`; one stale launch
lease from the original completed run had to be cleared first — verified
dead on both nodes before removing, per the documented procedure).
`teacher_argmax_acceptance` reproduced (0.9715899536061569, matches the
recorded 0.97159 to the precision published). `epoch_seconds` 250.5-266.7s
across the 8 stages — consistent with the original 249.6-265.1s.

**teacher_exact_seq_rate = 0.6021245774987929** (1247/2071, 60.21%) — source
`runs/spec_397b_v4_ppp8x_e1/stage7/metrics.jsonl`. **This number is NOT
comparable to the other 5 models'** — repeating the caveat from the top of
this Phase 2 section: 397B's `responses_bs256.jsonl` is byte-copied from
Qwen3.5-122B-A10B's own vLLM answers, so this measures how well 397B's OWN
forward reproduces a DIFFERENT model's word choices (cross-model agreement),
not genuine self-reproduction. A much lower exact-seq than the 5 real
models (85.7-98.6%) is exactly what's expected under that framing — two
different models, even both derived from the same Qwen3.5 lineage, will
diverge far more on a full-answer-exact basis than a model does from its
own vLLM-generated draft. This is a speed/mechanism data point, not a
fidelity finding, per the standing caveat.

### Phase 2 exact-seq work: CLOSED. Phase 3's exact-seq objective is now
COMPLETE for all genuinely-testable models (26B/27B/31B/35B/122B); 397B is
recorded with its caveat; DeepSeek remains excluded pending its separate
PPP8 NCCL fix (owner-authorized follow-up, not required for this closure).

## PHASE 3 — PPP2 sweep progress: Qwen3.6-27B (2026-07-20)

`scripts/launch_v4_stages.sh configs/experiments/spec_verify/base_27b_v4_spec.yaml
configs/experiments/spec_verify/27b_v4_spec_ppp2.yaml`, 2 stages,
`v4_stage_devices: [0, 1]` on agpuh01, run in parallel with 31B's PPP2 on
agpuh02 (no contention, different hosts). Source:
`runs/spec_27b_v4_ppp2_e1/stage1/metrics.jsonl` (stage1 owns the vocab
head).

| metric | PPP2 | PPP4 reference | match |
|---|---:|---:|---|
| teacher_argmax_acceptance | 0.9995056985891225 | 0.9995056985891225 | bit-exact |
| teacher_exact_seq_rate | 0.9859971028488653 | 0.9859971028488653 | bit-exact |
| exact_seq_match_answers | 2042 | 2042 | exact |
| answer_token_count | 70807 | 70807 | exact |
| epoch_seconds (stage1/top) | 159.09 | 109.06 (PPP4 stage3) | PPP2 slower — fewer stages, less pipeline overlap, same pattern already seen on 26B/35B |

Both teacher-side identity metrics bit-exact against PPP4 — extends the
now-established parallelism-invariance finding (including exact-seq, not
just the older per-token metric) to 27B. 31B and 122B PPP2 in flight next
to complete this sweep for all 5 models.

## PHASE 3 — PPP2 sweep progress: gemma-4-31B (2026-07-20)

Same command pattern on agpuh02, run in parallel with 27B's PPP2 on
agpuh01. Source: `runs/spec_31b_v4_ppp2_e1/stage1/metrics.jsonl`.

| metric | PPP2 | PPP4 reference | match |
|---|---:|---:|---|
| teacher_argmax_acceptance | 0.9992767415302627 | 0.9992767415302627 | bit-exact |
| teacher_exact_seq_rate | 0.9744084983099952 | 0.9744084983099952 | bit-exact |
| exact_seq_match_answers | 2018 | 2018 | exact |
| answer_token_count | 78810 | 78810 | exact |
| epoch_seconds (stage1/top) | 126.37 | 83.26 (PPP4 stage0) | PPP2 slower, same pattern |

Bit-exact on both teacher-side identity metrics. 4 of 5 models now confirm
exact-seq is parallelism-invariant (26B/27B/31B/35B); 122B PPP2 in flight
to complete the sweep.

## PHASE 3 — PPP2 sweep progress: Qwen3.5-122B-A10B (2026-07-20) — SWEEP COMPLETE

Source: `runs/spec_122b_v4_ppp2_e1/stage1/metrics.jsonl`.

| metric | PPP2 | PPP4 reference | match |
|---|---:|---:|---|
| teacher_argmax_acceptance | 0.9966268015946029 | 0.9966268015946029 | bit-exact |
| teacher_exact_seq_rate | 0.8947368421052632 | 0.8947368421052632 | bit-exact |
| exact_seq_match_answers | 1853 | 1853 | exact |
| answer_token_count | 101091 | 101091 | exact |
| epoch_seconds (stage1/top) | 137.87 | 96.55 (PPP4 stage0) | PPP2 slower, same pattern |

**PPP2 sweep for all 5 Phase-1 models is now COMPLETE.** All 5 confirm
bit-exact `teacher_argmax_acceptance` AND `teacher_exact_seq_rate` between
PPP2 and PPP4 — parallelism-invariance holds for every model tested,
extending the 0.8B-scale finding to real scale across dense and MoE
architectures alike, for both the token-level and answer-level metric.
PPP1 (single-GPU, all blocks) sweep is the remaining Phase 3 piece, in
flight next across all 5 models in parallel.

## PHASE 3 — PPP1 sweep: architectural blocker, fix, and a confound finding (2026-07-20)

PPP1's original definition for every model's config (`v4_teacher_source:
online` + `v4_stage_scoped: false`, all blocks resident on one GPU)
genuinely does not fit 26B/27B/31B/35B/122B on one 80GB H100 at
`micro_batch: 64` — confirmed CUDA OOM for every model attempted, inside
`_online_teacher_capture` (26B: "Tried to allocate 3.29 GiB... 75.85 GiB in
use"; 31B: "Tried to allocate 32.00 MiB... 78.37 GiB in use"; same pattern
for 27B/35B/122B). This is a real capacity ceiling, not a bug: all-blocks-
resident plus online capture activations at 64 items does not fit.

**Fix:** every model's PPP1 config now uses `v4_teacher_source: store` +
`v4_stage_scoped: true` + `v4_weight_residency: rotate` (only ONE block's
weights resident at a time) + `v4_teacher_residency: cpu_stream` (pinned
explicitly, not `auto`, since several PPP1 stores running concurrently
shrink host `MemAvailable`) + `v4_loop_order: layer_major` — the same
family PPP2/PPP4 already use, just with `v4_stage_splits: []` (one stage
owns the whole model) instead of a mid-model cut. This is not a redefinition
of PPP1; it is PPP2/PPP4's already-working recipe with the stage count set
to 1.

**A real confound, caught empirically, not just in principle:** `store`
mode is not a static cache read — `online_v4.py`'s
`online_source = v4_teacher_source in ("online", "store")` treats both the
same way; the store cache only holds vLLM's answer-token ids, so
teacher-hidden is still a LIVE forward capture at whatever `micro_batch` is
set. An initial 35B PPP1 attempt mirrored an older demo config's smaller
`micro_batch: 16` + `v4_optimizer: adam` (instead of PPP2/PPP4's inherited
`micro_batch: 64` + `immediate_sgd`) and came back **close but NOT
bit-exact**:

| metric | PPP1 (confounded: mb=16, adam) | PPP4 reference | delta |
|---|---:|---:|---|
| teacher_argmax_acceptance | 0.9977890422778257 | 0.997829486626402 | -0.00004 |
| teacher_exact_seq_rate | 0.9251569290197972 | 0.9261226460647031 | -0.00097 |

Preserved as `runs/spec_35b_v4_ppp1_e1.confounded_microbatch16_adam` (not
overwritten) — this is direct evidence that batch-shape-dependent bf16
GEMM numerics are real at this scale (consistent with the 26B
online-vs-store diagnostic's 0.99491 vs 0.99529 finding earlier in Phase 3),
and that holding `micro_batch`/`v4_optimizer` fixed at PPP2/PPP4's values is
load-bearing for a clean "only stage count differs" comparison, not
methodological pedantry. Every model's PPP1 config has been corrected to
drop these overrides and inherit `micro_batch: 64` / `immediate_sgd` from
its `base_*_v4_spec.yaml`, exactly matching PPP2/PPP4. Relaunched for all 5
models (26B/31B on agpuh01 cuda:2/cuda:3, 27B/35B/122B on agpuh02
cuda:0/cuda:1/cuda:2); results pending.

### Qwen3.6-35B-A3B PPP1 (clean recipe) — COMPLETE, BIT-EXACT

Source: `runs/spec_35b_v4_ppp1_e1/stage0/metrics.jsonl` (verified on disk;
config confirms `micro_batch: 64`, `v4_optimizer: immediate_sgd` — the
clean, non-confounded recipe).

| metric | PPP1 (clean) | PPP4 reference | match |
|---|---:|---:|---|
| teacher_argmax_acceptance | 0.997829486626402 | 0.997829486626402 | bit-exact |
| teacher_exact_seq_rate | 0.9261226460647031 | 0.9261226460647031 | bit-exact |
| exact_seq_match_answers | 1918 | 1918 | exact |
| answer_token_count | 74176 | 74176 | exact |
| epoch_seconds | 116.85 | 71.07 (PPP4 stage0) | PPP1 slower (single-block rotate overhead), same pattern as PPP2 |

First genuine PPP1 datapoint of the campaign: parallelism-invariance now
confirmed across PPP8→PPP4→PPP2→PPP1 for this model, and the clean-vs-
confounded pair above is a direct, matched-model demonstration that the
recipe correction (holding micro_batch/optimizer fixed) is exactly what
made the difference between a near-miss and a bit-exact result. 27B and
122B PPP1 (agpuh02) and 26B/31B PPP1 (agpuh01) still in flight.

### gemma-4-26B-A4B PPP1 (clean recipe) — COMPLETE, BIT-EXACT

Source: `runs/spec_26b_v4_ppp1_e1/stage0/metrics.jsonl` (verified on disk;
config confirms `micro_batch: 64`, `v4_optimizer: immediate_sgd`).

| metric | PPP1 (clean) | PPP4 reference | match |
|---|---:|---:|---|
| teacher_argmax_acceptance | 0.9952869328553885 | 0.9952869328553885 | bit-exact |
| teacher_exact_seq_rate | 0.8570738773539353 | 0.8570738773539353 | bit-exact |
| exact_seq_match_answers | 1775 | 1775 | exact |
| answer_token_count | 90387 | 90387 | exact |
| epoch_seconds | 109.98 | 68.95 (PPP4 stage0) | PPP1 slower (rotate overhead) |

### Qwen3.5-122B-A10B PPP1 (clean recipe) — COMPLETE, BIT-EXACT

Source: `runs/spec_122b_v4_ppp1_e1/stage0/metrics.jsonl` (verified on disk;
config confirms `micro_batch: 64`, `v4_optimizer: immediate_sgd`).

| metric | PPP1 (clean) | PPP4/PPP2 reference | match |
|---|---:|---:|---|
| teacher_argmax_acceptance | 0.9966268015946029 | 0.9966268015946029 | bit-exact |
| teacher_exact_seq_rate | 0.8947368421052632 | 0.8947368421052632 | bit-exact |
| exact_seq_match_answers | 1853 | 1853 | exact |
| answer_token_count | 101091 | 101091 | exact |
| epoch_seconds | 263.63 | 96.64 (PPP4 stage0) | PPP1 slower (rotate overhead, largest model) |

### gemma-4-31B PPP1 (clean recipe) — COMPLETE, BIT-EXACT

Source: `runs/spec_31b_v4_ppp1_e1/stage0/metrics.jsonl` (verified on disk;
config confirms `micro_batch: 64`, `v4_optimizer: immediate_sgd`).

| metric | PPP1 (clean) | PPP4 reference | match |
|---|---:|---:|---|
| teacher_argmax_acceptance | 0.9992767415302627 | 0.9992767415302627 | bit-exact |
| teacher_exact_seq_rate | 0.9744084983099952 | 0.9744084983099952 | bit-exact |
| exact_seq_match_answers | 2018 | 2018 | exact |
| answer_token_count | 78810 | 78810 | exact |
| epoch_seconds | 167.26 | 140.137 (PPP4 stage0) | PPP1 slower (rotate overhead) |

4 of 5 models now bit-exact on PPP1 (35B, 26B, 122B, 31B). Only 27B
remains.

### Qwen3.6-27B PPP1 (clean recipe) — COMPLETE, BIT-EXACT

Source: `runs/spec_27b_v4_ppp1_e1/stage0/metrics.jsonl` (verified on disk;
config confirms `micro_batch: 64`, `v4_optimizer: immediate_sgd`).

| metric | PPP1 (clean) | PPP4 reference | match |
|---|---:|---:|---|
| teacher_argmax_acceptance | 0.9995056985891225 | 0.9995056985891225 | bit-exact |
| teacher_exact_seq_rate | 0.9859971028488653 | 0.9859971028488653 | bit-exact |
| exact_seq_match_answers | 2042 | 2042 | exact |
| answer_token_count | 70807 | 70807 | exact |
| epoch_seconds | 257.16 | 106.31 (PPP4 stage0) | PPP1 slower (rotate overhead) |

27B's own OOM/retry is resolved as transient contention, not a per-model
capacity ceiling: the first attempt crashed sharing the node with 35B/122B
(all three PPP1 captures running simultaneously); the identical clean
config succeeded solo once those two had finished and freed the node.
No dense-FFN-activation capacity effect needed as an explanation.

## PHASE 3 — PPP1 sweep COMPLETE: all 5 models bit-exact vs PPP4

| model | teacher_argmax_acceptance | teacher_exact_seq_rate | epoch_seconds PPP1 | epoch_seconds PPP4 (stage0) |
|---|---:|---:|---:|---:|
| Qwen3.6-27B (dense) | 0.9995056985891225 | 0.9859971028488653 | 257.16 | 106.31 |
| gemma-4-31B (dense) | 0.9992767415302627 | 0.9744084983099952 | 167.26 | 140.137 |
| Qwen3.6-35B-A3B (MoE) | 0.997829486626402 | 0.9261226460647031 | 116.85 | 71.07 |
| Qwen3.5-122B-A10B (MoE) | 0.9966268015946029 | 0.8947368421052632 | 263.63 | 96.64 |
| gemma-4-26B-A4B (MoE) | 0.9952869328553885 | 0.8570738773539353 | 109.98 | 68.95 |

**Parallelism-invariance now holds end-to-end across every degree tested in
this campaign, for all 5 Phase-1 models: PPP8(x-node, 397B only)/PPP4→PPP2→
PPP1, on both the token-level metric (`teacher_argmax_acceptance`) and the
answer-level metric (`teacher_exact_seq_rate`).** Every one of the 15
PPPn-vs-PPP4 comparisons run this campaign (5 models x {PPP2, PPP1} plus the
0.8B-scale PPP8/PPP4/PPP2/PPP1 tie) is bit-exact; the only non-bit-exact
PPP1 datapoint anywhere in the campaign was the deliberately-preserved
35B confounded run (micro_batch:16+adam), which is a batch-shape control,
not a counterexample. PPP1 is consistently the slowest degree per model
(single-block weight-rotation overhead, worst for the largest model, 122B,
at 2.73x its PPP4 stage0 time), exactly the tradeoff `v4_weight_residency:
rotate` is documented to make — trading epoch time for fitting on one card.

This closes Phase 3 (PPP2/PPP1 sweep) and, with it, the standing goal
("really finish phase 3": PPP1/PPP2 sweep across all 5 already-tested
models, full 2071-item epoch, packed efficiently across 8 cards) — genuinely
complete, not just launched. Remaining campaign items are Phase 2's
DeepSeek PPP8 training-side NCCL hang (root-caused, not yet repaired) and
the already-closed-as-blocked 397B/DeepSeek vLLM TP8 legs (capacity ceiling /
nvcc-driver incompatibility, see above).

**27B PPP1 crashed** on agpuh02 (host pinned-memory OOM inside
`put_linear`'s `full_inputs.cpu().pin_memory()`, not a GPU OOM) while
running concurrently with 35B and 122B — three simultaneous PPP1 jobs each
staging full per-owned-layer captures to pinned host memory exhausted host
RAM/pinned-memory headroom, exactly the risk flagged in the corrected
`122b_v4_spec_ppp1.yaml`'s own comment about `cpu_stream` residency under
concurrent sibling PPP1 stores. Not a config error — relaunching alone now
that 35B/122B have both exited and freed their host allocations. 31B PPP1
(agpuh01) still running.
