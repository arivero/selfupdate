# Related work — comparison with other distillation repos/libraries

Companion to [`related_work_zo_and_distilllens.md`](related_work_zo_and_distilllens.md)
(which covers zeroth-order fine-tuning and the DistillLens paper). This note
places `selfupdate`'s v4.6 layerwise protocol relative to several external
distillation tools and frameworks, as requested in
[arivero/selfupdate#1](https://github.com/arivero/selfupdate/issues/1).

Scope note: these are brief orientation summaries from public
descriptions/READMEs, not exhaustive audits of each project's source. Anyone
using this note to argue novelty in `paper/paper1` should re-verify claims
against the current state of each project before citing them.

---

## 1. DistillKit

A toolkit for hidden-state and logit distillation between Hugging Face
models, aimed at practitioners who want a turnkey CLI/config-driven recipe
(teacher logits or intermediate features -> student loss) rather than a
research framework.

**How it differs from this branch:**
- Its typical recipes optimize a **global, end-to-end student objective**
  (final-logit KL and/or a small number of feature-matching layers), not a
  **per-block, gradient-isolated** loss applied uniformly across every layer.
- It does not enforce a **Frozen-Vocabulary Principle**: the student's
  embedding/LM head are ordinarily part of the trained model.
- It is task/checkpoint oriented (produce a smaller deployable student), not
  aimed at studying *where in depth* behavior is being learned or at forward
  (layer-by-layer, non-backprop-through-the-whole-network) training dynamics.

## 2. DistillFlow

A multi-strategy distillation framework that supports combining several KD
recipes (response-based, feature-based, relation-based) in one pipeline,
generally oriented at flexibility/composability of losses rather than
enforcing one structural law.

**How it differs from this branch:**
- selfupdate is deliberately **not** multi-strategy: `train/validate.py`
  rejects any non-v4 (non-block-local, non-teacher-forced) configuration.
  Composability is treated as a confound (see `CLAUDE.md`'s "Config DEFAULTS
  are experiment variables" lesson), not a feature to expose.
- Feature-based terms in general multi-strategy frameworks are usually *added
  on top of* an end-to-end trainable student; here the block-local hidden
  match **is** the entire training signal — there is no additional
  final-logit/task loss contributing gradient.

## 3. Hugging Face TRL distillation workflows (online teacher/student)

TRL's online distillation trainers run teacher and student together at
training time (rather than from a precomputed cache) and typically optimize a
KL divergence between student and teacher **next-token distributions**
(GKD-style on-policy or off-policy), i.e., a **behavioral/logit-space**
objective, generation- or completion-level.

**How it relates:**
- selfupdate's `train.online_teacher: true` mode (`runtime.py`/`online_v4.py`)
  is the same "no precomputed cache" idea operationally, but the *loss* is a
  different kind: TRL's online recipes match final output distributions;
  selfupdate matches **intermediate teacher hidden states block-by-block**,
  with the final-logit KL (`KL-eval-loss`) demoted to an eval-only metric with
  structurally zero optimizer weight (see "Publication-Critical Constraints"
  in `CLAUDE.md`). This is the sharpest single point of contrast with TRL: TRL
  makes the behavioral KL the objective; selfupdate forbids exactly that as a
  training target on this branch.

## 4. NVIDIA NeMo distillation pipelines

NeMo's distillation tooling (built on Megatron-Core) targets
**production-scale** logit/hidden-state KD for large, sharded models, with
strong support for tensor/pipeline-parallel teacher-student setups.

**How it relates:**
- Architecturally closest in *spirit* to selfupdate's PPP (independent
  block-owner processes) on the Hardware Ladder's upper rungs — both need to
  keep a large teacher and student resident under parallelism. The difference
  is the training law, not the parallelism mechanics: NeMo's recipes
  generally still optimize an end-to-end (or few-layer) student loss with a
  standard trainable embedding/head, whereas selfupdate's PPP workers each own
  a **disjoint, frozen-input/frozen-target block range** with no cross-block
  training graph and a frozen vocabulary at both ends.
- NeMo is a production/scale-first framework; selfupdate is presently a
  research protocol (0.6B-397B "Hardware Ladder", see `CLAUDE.md`) whose
  claims are about *locality* of learning, not deployment throughput.

## 5. Progressive blockwise KD and task-aware layerwise distillation (TED)

TED ("Task-aware layEr-wise Distillation") and related progressive/blockwise
KD methods train student layers to match teacher layers progressively (often
layer-by-layer or stage-by-stage), which is the closest published family to
this branch's "layerwise" name.

**How it differs from this branch — the naming trap to avoid:**
- Published progressive/blockwise KD methods commonly (a) train a
  **narrower/shallower student architecture** (a genuine model-compression
  setting, not same-width block replacement), (b) mix a **task loss**
  (downstream label supervision, filtered through TED's task-aware
  projections) with the layer-matching loss, and/or (c) allow later-stage
  losses to backprop through **already-trained earlier layers**, i.e. the
  network is not strictly gradient-isolated block-by-block once training
  reaches later stages.
- selfupdate's structural law is narrower and stricter: same-width blocks,
  **only teacher-sourced** targets (no task/reference-text loss — see the
  "Historical training-target law" section of `CLAUDE.md`), and **hard
  gradient isolation** per block (or per sliding k-connected window) — a
  later block's loss never contributes gradient to an earlier block's
  weights. "Layerwise" in selfupdate specifically means this isolation
  property; it should not be read as a synonym for progressive/blockwise KD
  in general, and any comparison to TED-style methods should call out
  whether their student is narrower (compression) rather than same-width
  (this branch's setting is same-width, i.e., not compression).

## 6. ktransformers (kvcache-ai/ktransformers)

[ktransformers](https://github.com/kvcache-ai/ktransformers) is an inference
optimization framework (heterogeneous CPU/GPU offload, custom kernels,
quantization) for serving large/MoE models efficiently. It is not a
distillation or training framework.

**How it relates:**
- Not directly comparable on training methodology; it is relevant only as a
  potential **inference-side** tool for serving/evaluating large teacher (or
  merged student) checkpoints cheaply on constrained hardware — adjacent to
  this branch's MoE/122B+ "Hardware Ladder" rung and Don Quijote-scale
  evaluation, not to the block-local training law itself. Any future
  integration would be about cheaper eval/generation, not about the
  layerwise objective.

---

## Summary table

| Project | Training objective | Trains embedding/head? | Cross-block/layer gradient? | Student width vs teacher |
|---|---|---|---|---|
| DistillKit | end-to-end logit/feature KD | usually yes | yes (global) | usually smaller |
| DistillFlow | composable multi-strategy KD | usually yes | yes (global) | usually smaller |
| TRL online (GKD-style) | on/off-policy behavioral KL | yes | yes (global) | usually smaller |
| NeMo distillation | end-to-end/few-layer KD at scale | usually yes | yes (global) | usually smaller |
| TED / progressive blockwise KD | layer-matching + task loss | often yes | often yes (progressive) | usually smaller (compression) |
| ktransformers | n/a (inference-only) | n/a | n/a | n/a |
| **selfupdate (v4.6, this branch)** | **block-local teacher-hidden matching only** | **never (frozen)** | **no (hard isolation per block/window)** | **same-width** |

## Action items

- [ ] Re-verify each project's current training recipe before citing in
      `paper/paper1`; these are orientation summaries, not audits.
- [ ] Cross-reference this table with the DistillLens contrasts in
      `docs/related_work_zo_and_distilllens.md` when drafting the related-work
      section.
- [ ] If ktransformers or a similar inference framework is ever adopted for
      Don Quijote-scale evaluation, document that integration separately —
      it does not change the training law described here.
