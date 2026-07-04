# Issues / Follow-Ups

Post-campaign state (2026-07-04). The 24-40h campaign is recorded in
EXPERIMENTS.md (closing table) and runs/report.pdf.

## Done (campaign, 2026-07-03/04)

- Schema-3 caches rebuilt (v1/v2/v3/v4 at 0.6B; v2/v4 at 1.7B).
- Full test suite green throughout (52 tests at close).
- smoke test for non-layerwise rejection: superseded by the loss/schedule
  registries raising ValueError (covered by tests).
- Wave I-K: loss sweep, routing, scale, families, understanding probes,
  innovation arms — see EXPERIMENTS.md.
- Artifacts: results.md / curves.png / forget_curves.png / report.pdf.

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
5. **Scale**: final recipe at 4B/8B full-FT (sequential/tail_only for
   VRAM), 14B+ LoRA; Don Quijote data engineering.
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
