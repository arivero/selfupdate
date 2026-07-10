# Issues / Follow-Ups

Post-campaign state (2026-07-04). The 24-40h campaign is recorded in
EXPERIMENTS.md (closing table) and runs/report.pdf. Closed items are
removed from this file (git history keeps them); 2026-07-10 pass removed
the campaign done-list and the completed hot-loop ladder.

## Future Work

1. **Window capacity**: if final_k8 did not restore the 708-verse chain,
   build v4.1 with extra long-window replicas (the dilution hypothesis);
   study k as a budgetable capacity (triggers vs anchors vs depth).
2. **thinking_selective mask** (1d): full design in the campaign plan file
   (multi-privileged-span masking, find_poem_spans matcher,
   prefix-truncation fallback). Context update: reasoning-tuned families
   RESIST the recipe (Phi 0.918, gpt-oss 1.0) — selective think-censoring
   may be the way readout training reaches their output channels.
3. **Reasoning-family question**: why think/analysis-channel models fail;
   try training with the channel present in the student prompt.
4. **Tuned-lens program** (Wave I plan, still pending): per-layer
   translators for calibrated depth profiles; tuned-lens-CE auxiliary.
5. **Scale**: final recipe at 4B/8B full-FT (sequential/offload_adam for
   VRAM — tail_only is expunged on this branch), 14B+ LoRA; Don Quijote
   data engineering.
6. **Anchor corpus breadth**: anchors_es.txt is 6 fragments; a rotating
   larger corpus may improve anchor-KL further.

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

## Missing instrument: per-layer residuals AT CHECKPOINTS (2026-07-05)
Training-time per-layer losses are logged (metrics.jsonl per_layer) and
now plotted per run (scripts/layer_loss_plots.py -> eval/layer_losses.png).
NOT logged: eval-time per-layer hidden residuals of a CHECKPOINT against
its teacher (train-loss curves conflate optimization state with storage
quality). C3: add a --layer-residuals mode to evaluate.py (one teacher
pass + one student pass, per-layer vocab_mse/nmse on the aligned span)
and store alongside recite.json. Cheap (2 forwards/item), pairs with
weight_deltas.csv to give storage QUALITY next to storage LOCATION.
