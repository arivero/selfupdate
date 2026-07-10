# Issues / Follow-Ups

Post-campaign state (2026-07-04). The 24-40h campaign is recorded in
EXPERIMENTS.md (closing table) and runs/report.pdf. Closed items are
removed from this file (git history keeps them); 2026-07-10 pass removed
the campaign done-list and the completed hot-loop ladder.

## Future Work

1. **Window capacity as a budget**: study k as a budgetable capacity
   (triggers vs anchors vs depth). (The final_k8/708-chain conditional
   resolved in C1: k=8 restored the chain; thinking_selective landed in
   C2-12 and continues as C3 #2.)
4. **Tuned-lens program** (partially landed): translators + C2-11
   re-profiles are in tree (train_tuned_lens.py, tl_i_tunedlenskl). The
   Wave-I "tuned-lens-CE auxiliary" half is FORBIDDEN as label-CE; only
   the teacher-sourced tuned-lens-KL variant is a legal continuation.
5. **Scale**: final recipe at 4B/8B full-FT (sequential/offload_adam for
   VRAM — tail_only is expunged on this branch), 14B+ LoRA; Don Quijote
   data engineering.
6. **Anchor corpus breadth**: anchors_es.txt is 6 fragments; a rotating
   larger corpus may improve anchor-KL further.

## Unconsidered lenses/losses (owner question, 2026-07-10)

Candidates never tried, each doctrine-checked (teacher-sourced,
depth-uniform, Frozen-Vocabulary compatible):

1. **Attention-KL**: match teacher-vs-student per-head attention
   distributions at aligned positions. The mechanism story (README:
   "divergence comes from attention into the privileged block") has
   never been trained on directly — every loss so far targets the
   residual stream, none the attention that writes it. Depth-uniform by
   construction; distribution-shaped over POSITIONS not vocabulary, so
   Law 7's groove mechanism (completion-direction concentration) does
   not obviously apply — verify under the loss-safety protocol anyway.
2. **Per-layer anchor (anchor-lens)**: anchor-KL is output-only; the
   depth-uniform version is "stay near BASE trajectories on anchor
   text" — per-layer vocab_mse/nmse toward base-model states on anchor
   inputs. Directly aimed at C3 #9 (1.7B intrusion may live mid-stack
   where an output-level anchor cannot see it).
3. **Delta matching**: match the block UPDATE (h_L − h_{L−1}) instead of
   the state h_L. layer_residuals shows residuals compound with depth;
   state matching penalizes inherited upstream error, delta matching
   trains each block's own contribution only. Natural companion to the
   strict schedule; cheap to add to losses.py.
4. **Whitened/covariance metric**: Mahalanobis in the base model's
   activation covariance — completes the metric family between nmse
   (isotropic) and vocab_mse (Gram W^T W). p-uniform, so Law 7 predicts
   safe; tests whether vocab coordinates are special or just
   well-conditioned.
5. **Embedding lens on untied models**: decode hidden states through the
   INPUT embedding. Was an "idea" in C1 (identical to logit lens on tied
   ≤4B); now actionable — the C2modern bases (Gemma 4, Qwen3.6-27B) are
   untied, giving an independent per-layer probe for free.
6. **Contrastive hidden loss** (InfoNCE vs in-sequence negatives):
   sharper than MSE without vocabulary-distribution shaping; unknown
   safety — run small under the Law 7 protocol before believing it.
7. **Reverse-KL readout** (KL(student‖teacher), mode-seeking): still
   bounded by teacher top-1 (96.8%), so C2-34 predicts it cannot beat
   the last-3% law — worth one cheap arm only as a tightness check of
   that bound.

## Test-suite economics (2026-07-10; owner: "token sinkholes")

Suite: 126 tests, ~51 s warm / ~149 s cold. No single hog (slowest test
4.7 s) — the cost is structural: ~10 GPU test modules EACH load
Qwen3-0.6B separately (module-scoped fixtures, no session sharing), so
most wall-clock is repeated model loads, not assertions. Worst
offenders by measurement: test_online_teacher (3 of the 4 slowest tests
+ 14 torch.jit deprecation warnings — `jit.script_method` removal risk
on the next torch bump), test_anchor, test_mixed_schedule,
test_position_invariance. Proposals:
- session-scoped shared 0.6B BlockStack fixture (tests freeze
  non-blocks; give mutating tests function-scoped adapters/deepcopies);
- `-m slow` marker on GPU-model tests so the certify gate can run a
  fast lane first;
- replace the torch.jit usage in the online-teacher path/tests before
  torch removes it.

## Open review findings (multi-agent review 2026-07-10)

Fixed same-day: ALIA left-pad scoring corruption (+ artifacts
regenerated), quijote rung conflation (rung-level corpora everywhere),
validator holes H1-H4 (anchor/stride/readout-weight/tail-only-per-class
+ hidden_loss=zero disguise). Still open, priority order:

1. `eval/tasks.py` EOS lookup: `convert_tokens_to_ids("<|im_end|>")`
   returns unk id 0 on SentencePiece (Mistral) — generation never stops;
   use `chatfmt.stop_token_id` as recite.py does.
2. `evaluate.py --layer-residuals` adopts model+poem from the checkpoint
   but examples/mask geometry from base.yaml — Quijote/stub_gap
   checkpoints measured against wrong dataset unless --experiment is
   passed; adopt the whole data/mask block or refuse.
3. `_stage_source` mkdir-lock has no stale-owner detection: a killed
   copy job wedges every later job on that model/node silently.
4. `cache.hidden_dtype` is hash-only (writer hardcodes fp16, no
   overflow guard — 8B+ outlier channels could cache as inf).
5. `find_poem_spans`: no word-boundary anchors (mid-word censor starts);
   single-punctuation deviation escapes whole-verse censoring
   (thinking_selective arms).
6. `router_aligned` + `window_dedup`: fails deep in item 1 with a
   misleading "graph leak" tripwire — reject at validation.
7. Minors: poem.py window ranges never reach the last verse (fix in
   v-next dataset gen; v1 is byte-guarded); config merge is one-level
   (nested dict override resets siblings); tasks_eval never calls
   model.eval() (matters when dropout appears); 4 dead PENDING rows in
   scripts/queue.tsv reference deleted configs; analyze.py reads
   retired "general" key unguarded; dead CLI flags on evaluate.py
   (--batch-size etc. silently ignored post-retirement); "gold" →
   "reference" rename pending in retention_eval.py / surprise_probe.py /
   cross_report.py / make_figs.py; only ARC-Easy is repo-pinned
   (hellaswag/arc_challenge/wikitext float on HF revisions); /tmp eval
   staging never cleaned (~170 GB/node at full fleet); audit_configs
   does not scan queue TSVs and its OLD_KEYS blacklist is name-based.

## Campaign roadmap beyond C2 (sketched 2026-07-04, owner question)

- **C3 — conversations (Stage B):** conversation-to-weights (privileged =
  oldest turns; QA-about-censored-turns eval); attention-scored span
  selection via the head taxonomy ("worth of attention" operational);
  cycle mechanics: early-stop-on-readout (C2-6), heterogeneous batching
  (C2-8/9), destruction gate as automated accept/reject; gpt-oss
  thinking_selective with harmony harvests; 8B-14B full-FT via
  offload_adam + sliding-window prefetch; before/after MoE routing-shift
  probe (C2-15 follow-up).
- **C4 — the person (Stage C):** 120B MoE on H100s, streamed-block
  consolidation during serving idle; primary metrics = RAG-independence
  curve + query-sophistication drift (docs/evolving_person.md); weeks-long
  continual run with nightly destruction gate (slow-drift watch); fleet
  evolution: experience-log replay vs gated diff-merging; intrusion
  metric as privacy audit; live-Socratic demo as the closing exhibit.

C2 built the instruments; C3 masters the unit of experience (one
conversation, one cycle); C4 composes cycles into a life.

## C3 model candidates beyond 14B (scouted 2026-07-04, web sources in chat)

Constraint: 4x L40S 46GB (184 GB node; 92 GB per 2-card PP job); need
thinking mode + tool use + HF layout compatible with BlockStack.

- **Qwen3.6-35B-A3B** (Apr 2026, Apache 2.0): 35B-total/3B-active sparse
  MoE. Primary C3 candidate — family continuity with the whole ladder,
  2-card PP (~70 GB bf16), MoE-router instrument applies, thinking mode.
- **Gemma 4 26B-A4B** (Apr 2026, Apache 2.0): 25.2B MoE, 3.8B active,
  256K ctx, native tool use / MCP. Second family for generality; 2-card.
  Also 31B dense (workstation tier) as a dense scale point; 12B unified
  multimodal as single-card option.
- **DeepSeek V4 Flash**: 284B/13B-active, 1M ctx ("Engram conditional
  memory" — relevant to our memory program conceptually). FP8 ~284 GB »
  our node; INT4 ~142 GB would fit 4 cards but quantized hidden states
  are a research risk for trajectory matching. C4-class target on H100s
  (the owner's 4xH100 scenario), alongside gpt-oss-120b.
- GLM-4.7-Flash (~30B MoE) / GLM-5.2, Kimi K2.7 (1T/32B active): noted;
  K2-class is beyond any near-term node.
- Caveat: blog-grade specs — verify model cards + licenses + BlockStack
  layout (fails loudly by design) before committing arms.

**Owner addition (2026-07-04): Qwen3.6-27B as the parallelism bridge
model.** Dense 27B (Apr 2026, Apache 2.0, thinking mode, 262K ctx,
SWE-bench 77.2): two-card on L40S (54 GB bf16 → PP2/TP2 mandatory) AND
one-card on H100 80GB (traditional reference possible). Plan: once TP+PP
are understood on current models (PP2 repro + 32B arm in flight), run
the same 27B recipe as {single-H100 reference, PP2, TP2} and compare —
parallelism correctness against a no-parallelism ground truth, and
layerwise-vs-traditional at a size both can run. Its "Thinking
Preservation" mechanism is adjacent to thinking_selective — investigate
at harvest time.

Single-L40S 27B addendum: bf16 impossible (54 GB weights alone), but the
official Qwen3.6-27B-FP8 checkpoint (~27 GB) + bf16 LoRA + adapters-off
teacher ≈ 31-33 GB fits one card. Risks: FP8 forward through our block
walk + kernels==0.12.0 pin; SCIENCE: FP8-quantized teacher trajectories
(what does trajectory distillation lose under a quantized teacher? —
same question that gates INT4-base training of V4-Flash-class at C4).
Full 27B grid: {1xH100 bf16 ref, PP2 bf16, TP2 bf16, 1xL40S FP8-LoRA}.

Qwen3.6 compatibility check (2026-07-04, transformers 5.12.1 — no
upgrade needed, kernels pin safe): configs load; 3.6 reuses qwen3_5
classes. 27B = MULTIMODAL composite (text_config: qwen3_5_text, 64
layers, hidden 5120, UNTIED head — PP-friendly) + vision_config; the
text tower is not at model.model.* → BlockStack and _pp_device_map need
a small layout adapter (the designed fail-loudly path, docs/scaling.md).
35B-A3B = qwen3_5_moe_text, 40 layers, 256 experts top-8 (finer routing
than gpt-oss's 32 — better router-probe resolution). Adoption cost ≈
half a day: layout adapter + template-pieces verification + thinking
harvest ("Thinking Preservation" mode). Why the 3.6 series was absent
from C1/C2: released 2026-04, post-dating the program design and the
assistant's knowledge cutoff — an inertia blind spot caught by the owner
2026-07-04; matched-ablation continuity justified staying on Qwen3
within-campaign, but C3 arms should default to 3.6-generation bases.

## Trainer hot-loop — CLOSED 2026-07-10 (knowledge kept below)

Diagnosed sync-bound 2026-07-05 (owner question): `.item()` per block was
1.46x of the walk. The C3 engineering ladder is COMPLETE — GPU-side
logging, padded/bucketed batching with equivalence tests, window forward
dedup, one-AdamW foreach policy (2026-07-09); then the 2026-07-10
refactor: TrainingRuntime + explicit OptimizerPlan, ONE batched walk
(item mode = B=1 padded batch, bit-exact), streamed pinned-CPU
offload_adam (0.949 → 0.358 s/step at 0.6B), sliding-window trajectory
release, hook-free PP block walk, memory-budget planner, and the
certification harness (certs/pre single-device + certs/pp2 pipeline
references). Details and the change gate: docs/runtime.md and AGENTS.md
"Training Runtime & Certification". Timing regimes are NOT comparable
across the refactor boundary — do not mix pre/post ms-per-item numbers
in one table.

Kept for the record (do-not-rebuild guidance):

NEGATIVE RESULT (2026-07-10, refactor session): async pinned-memory target
prefetch (side CUDA stream, pin_memory + event-synced staging of layer L+1
while block L computes) was implemented and MEASURED SLOWER on L40S at 0.6B:
item mode -9%, slide8-dedup padded B4 -25%, no memory win. The per-tensor
pin_memory cost exceeds what the async copy hides — the batched walk already
covers these small (<5 MB/layer) pageable H2D copies. Do not rebuild without
first measuring a pinned-POOL variant at 4B+ scale where targets are >20 MB
per layer. The related real win that DID land: sliding-window trajectory
states are now released at their last root use (activation residency W states
instead of full depth; -180 MB at 0.6B slide8 B=8, scales with H*B*T*n).

PP2 hook measurement (2026-07-10, L40S 0.6B seq600 fwd+bwd walk): single
54-55 ms/item; PP2 with accelerate dispatch hooks 61-71; hooks stripped +
explicit boundary moves 56-67 (~8% of the PP2 walk is hook dispatch — the
Python pytree traversal, not redundant transfers: pre-moving inputs under
intact hooks changes nothing). PP2 is SLOWER than single-GPU for this
depth-sequential workload in every variant — PP is a memory technology here.
Within a grad-accum window weights are frozen, so cross-item device overlap
(item i+1 on partition 0 while item i runs partition 1) would be EXACT, not
stale — the honest PP throughput move if ever needed. End-to-end (real
trainer, 0.6B) the hook-free walk is within run noise; the isolated-walk 8%
is the honest number, and the explicit boundary moves are the 120B
streaming contract anyway.

PP2 CERTIFICATION LANDED (2026-07-10): certs/pp2/*.json — the real trainer
under model.pipeline_split=14 certified against the SINGLE-DEVICE
references in certs/pre (semantic config hash excludes placement knobs).
The first attempt immediately caught a LATENT pre-refactor bug: a readout
window on a tied-vocab model (Qwen3 <=1.7B) computes the L=n loss on the
vocab card (cuda:0) while in-window losses live on the block card — the
backward-scalar sum mixed devices and crashed. Readout+PP2+tied had simply
never been run. Fixed (accumulate on one device, scalar moves,
autograd-recorded); single-device numerics untouched (no-op .to). TP2
remains probe-only (parallel_bench): collectives inside every linear lose
badly at trainable sizes; use PP at block boundaries, TP only if a single
block cannot fit.

## Per-layer residuals at checkpoints — CLOSED 2026-07-10
`evaluate.py --layer-residuals` landed (one teacher + one student pass,
per-layer nmse/l2mse/vocab_mse/norm-ratio on the aligned span; writes
layer_residuals.{json,csv,png} next to recite.json). First profile on
lw_r_s43_pinned: shallow layers track the teacher tightly (h1 nmse
0.002), residuals grow with depth and depart sharply inside the readout
window (h21-h28: 0.17-0.83) — storage quality now measurable separately
from training loss.
