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
