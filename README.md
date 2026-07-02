# selfupdate — self-updating LLMs via self-distillation of context

The same model plays teacher and student. The teacher's prompt contains privileged
context — a RAG-retrieved passage, or its own visible `<think>` trace — while the
student receives the identical prompt with that context hidden, and is trained to
reproduce the teacher's behavior anyway. Because teacher and student share the
architecture *and the initial weights*, we can distill two ways:

1. **Classical KD** — KL divergence on output logits, backprop through the whole student.
2. **Layer-wise hidden-state matching** — compare student and teacher hidden states
   layer by layer at aligned token positions. Backprop stays confined to a single
   block, which cuts activation memory to one layer and opens a path to updating
   120B-class models one layer at a time on limited GPUs. Variants: local losses on
   all layers simultaneously (`summed`), or progressive layer-by-layer fitting with
   frozen-prefix activation caching (`sequential`).

Research questions:

1. Training efficiency of logit-KD vs the layer-wise variants.
2. **Where** (which layers) memorized content lands under each method — measured by
   per-layer weight-delta norms, tuned-lens depth profiles, layer grafting/ablation,
   and LoRA adapter norms.
3. Whether different training methods converge to the same storage locations.

## The Pierre Menard program

- **Stage 1 (single RTX 3060, 12 GB):** memorize *La tierra de Alvargonzález*
  (Antonio Machado, *Campos de Castilla*, 1912 — freshly in the public domain, so
  not deeply present in existing model weights) with Qwen3-0.6B → 1.7B → 4B.
- **Stage 2 (4×H100):** scale to ~120B dense and DeepSeek-class MoE models; the
  memorization target scales to **Don Quijote** — the model, like Pierre Menard,
  must come to produce the Quixote as its own.

## Layout

```
configs/            base + experiment YAMLs
data/poem/          raw.txt (fetched by scripts/fetch_poem.py) + examples.jsonl
caches/             teacher caches (gitignored)
runs/               experiment outputs (gitignored)
scripts/            fetch_poem, build_dataset, build_teacher_cache, train, evaluate, analyze
src/selfupdate/     masking, data, teacher, train, eval, utils
tests/              alignment / position-invariance / cache / loss / locality tests
```

## Method notes

- Every example is four text segments: `shared_prefix | privileged | shared_mid | answer`.
  The teacher sees all four; the student skips `privileged`. Segments are tokenized
  separately and concatenated as IDs, and token identity between the two sequences is
  asserted at dataset build time. The aligned span is `shared_mid + answer`.
- Qwen3 uses pure RoPE with full attention, so a constant position offset is
  output-invariant (unit-tested): the teacher/student divergence at aligned positions
  comes only from attention into the privileged block — exactly the signal we distill.
- **Compaction axis:** the student's side of the privileged block is either removed
  outright (`remove`, zero size), replaced by a short uninformative placeholder
  (`stub`), or replaced by the placeholder plus a position-id gap (`stub_gap`) that
  makes the student's RoPE geometry match the teacher's exactly — isolating the
  missing attention targets as the only difference. Which compaction distills and
  localizes best is a compared research question.
- The teacher is the frozen initial checkpoint. Per-layer hidden states (fp16) and
  top-k logits (k=128, plus fp32 logsumexp for an exact tail bucket) are precomputed
  once into sharded safetensors; the teacher never occupies GPU memory during training.
- The hidden-layer comparison loss (nmse / l2mse), the positions convention, and the
  proofs that layerwise training is genuinely block-local (no logits, no cross-block
  gradients — each enforced by a test) are documented in
  [docs/hidden_loss.md](docs/hidden_loss.md).

## Related work

Every reference below was verified against arXiv before inclusion.

### Context distillation and internalizing reasoning

- Askell et al. 2021, *A General Language Assistant as a Laboratory for Alignment*,
  arXiv:2112.00861 — origin of "context distillation": KL-train a model to imitate its
  own prompted distribution so the prompt bakes into the weights. Our exact template.
- Snell, Klein & Zhong 2022, *Learning by Distilling Context*, arXiv:2209.15189 —
  internalizes instructions and scratchpads; distillation beat direct fine-tuning by
  ~9% on SPIDER. Their forward KL on teacher tokens is our default KD loss.
- Kujanpää et al. 2024, *Efficient Knowledge Injection in LLMs via Self-Distillation*,
  arXiv:2412.14964 — **closest precedent**: "prompt distillation" internalizes documents
  via self-distillation (same model with the document in context as teacher), beating
  SFT and sometimes RAG. It does no layer-localization analysis — that gap is our focus.
- Deng et al. 2023, *Implicit Chain-of-Thought Reasoning via Knowledge Distillation*,
  arXiv:2311.01460 — student predicts teacher hidden states across layers; precedent for
  our layer matching, reported hard to optimize. The follow-up (Deng et al. 2024,
  arXiv:2405.14838) switches to a token-removal curriculum — our fallback schedule.
- Yu et al. 2024, *Distilling System 2 into System 1*, arXiv:2407.06023 — self-teacher /
  self-student at scale, SFT on outputs only; motivation rather than method.
- Mu et al. 2023, *Learning to Compress Prompts with Gist Tokens*, arXiv:2304.08467 —
  prompt→activation compression; the contrast case for weight internalization.
- Yang et al. 2024, *Self-Distillation Bridges the Distribution Gap (SDFT)*,
  arXiv:2402.13669 — self-distilled targets mitigate catastrophic forgetting; a
  mitigation to fold in if memorization degrades general ability.
- Chen et al. 2025, *DistilledPRAG*, arXiv:2509.01088 — distills a RAG teacher into a
  retrieval-free student matching both logits and hidden states (via generated LoRA);
  the strongest methodological sibling for combined losses.

### Memorization and layer localization

- Stoehr et al. 2024, *Localizing Paragraph Memorization in Language Models*,
  arXiv:2403.19851 — verbatim memorization shows gradient concentration in **lower
  layers** plus a rare-token attention head. Predicts where our delta norms should peak.
- Huang et al. 2024, *Demystifying Verbatim Memorization in LLMs*, arXiv:2407.17817 —
  inject-then-probe protocol (methodologically closest to ours); finds verbatim
  memorization **distributed** rather than weight-isolable. The counterpoint to Stoehr.
- Meng et al. 2022, ROME (arXiv:2202.05262) and MEMIT (arXiv:2210.07229) — facts
  localize to mid-layer MLPs; the *factual* contrast case, plus causal tracing as a tool.
- Hase et al. 2023, *Does Localization Inform Editing?*, arXiv:2301.04213 — where a
  memory is carried differs from where editing works; our four localization readouts
  may legitimately disagree, and we report them side by side.
- Carlini et al. 2022, *Quantifying Memorization Across Neural Language Models*,
  arXiv:2202.07646 — memorization scales log-linearly with capacity, duplication, and
  context length; sets expectations for poem → Quijote.
- Dai et al. 2022, *Knowledge Neurons*, arXiv:2104.08696; Geva et al. 2021,
  *Transformer FFN Layers Are Key-Value Memories*; Belrose et al. 2023, *Tuned Lens*,
  arXiv:2303.08112 (our per-layer probe; the original logit lens is nostalgebraist 2020);
  Ovadia et al. 2023, *Fine-Tuning or Retrieval?*, arXiv:2312.05934 (RAG beats plain
  fine-tuning for knowledge injection — motivates distillation over plain FT);
  Allen-Zhu & Li 2024, arXiv:2404.05405 (~2 bits/param capacity context).
- Lasy et al. 2025, arXiv:2506.21588 — memorization initiation vs maintenance circuits.

The localization literature disagrees — Stoehr (low layers, localizable) vs Huang
(distributed, entangled) vs ROME/MEMIT (mid-layer MLPs, but for facts, not sequences).
Our controlled inject-then-probe setup with *matched training signals across methods*
sits directly on that open question.

### Layer-wise / local training and layer-wise distillation

- Jiao et al. 2019, *TinyBERT*, arXiv:1909.10351 — per-layer hidden+attention MSE with
  learnable projections; our same-weights setup drops the projections entirely.
- Sun et al. 2019, *Patient Knowledge Distillation*, arXiv:1908.09355 — supervision
  spread across depth (PKD-Skip) beats top-only (PKD-Last); supports the `summed`
  variant. We copy their L2-normalize-before-MSE stabilizer.
- Wang et al. 2020, *MiniLM*, arXiv:2002.10957 — last-layer-only attention-relation
  distillation; the opposing bet, used as an ablation baseline.
- Hinton 2022, *Forward-Forward*, arXiv:2212.13345; Löwe et al. 2019, *Greedy InfoMax*,
  arXiv:1905.11786; Belilovsky et al. 2019, arXiv:1812.11446 (+ decoupled follow-up
  arXiv:1901.08164); Bengio et al., NeurIPS 2006 — the local/greedy-training lineage.
  Known failure mode (Chen 2026, arXiv:2606.06539): layer-local objectives degrade with
  task complexity through compounding per-layer error. Our local targets are *exact
  teacher hidden states*, which bounds per-layer drift; we measure error-vs-depth.
- Amid et al. 2022, *LocoProp*, arXiv:2106.06199 — theory of per-layer local losses
  with local targets (still one global backward; framing, not memory savings).
- Pan et al. 2024, *LISA*, arXiv:2403.17919; Luo et al. 2024, *BAdam*, arXiv:2404.02827 —
  subset-of-layers / block-coordinate fine-tuning at 70B scale on small GPUs; precedent
  for one-layer-at-a-time 120B training. BAdam still backprops the global loss; our
  local loss removes the full backward pass. Zhao et al. 2024, *GaLore*,
  arXiv:2403.03507 — composable optimizer-state savings.
- Furlanello et al. 2018, *Born-Again Networks*, arXiv:1805.04770 — same-architecture
  self-distillation, output-only, re-initialized student.

No verified prior work combines same-architecture **and same-initial-weights**
layer-aligned hidden matching for local per-block training. That combination is this
project's novelty claim.

## Scaling

The path to 120B-class dense and DeepSeek/GLM-class MoE models on 4×H100 —
including where vLLM/sglang fit (teacher trace generation and KD logits) and
where they cannot (per-layer hidden states need a layer-streamed forward) —
is documented in [docs/scaling.md](docs/scaling.md).

## Transformers 5.x notes

`dtype=` (not `torch_dtype=`) in `from_pretrained`; KV cache is `Cache` objects (we
pass `use_cache=False`); `apply_chat_template(..., enable_thinking=…)` for Qwen3; Qwen3
decoder layers receive precomputed `position_embeddings` from the outer model, so the
per-block runner computes RoPE itself from the shared `rotary_emb` module.
