# selfupdate_kd — classical self-distillation of context

The same model plays teacher and student. The teacher's prompt contains
privileged context — a RAG passage, or its own visible `<think>` trace — while
the student receives the same prompt with that context hidden. The student is
trained with classical knowledge distillation to reproduce the teacher's output
distribution anyway.

This branch is intentionally narrow: **classical KL-based distillation only**.
The research question is where that classical KD update writes the new memory:
which transformer layers move, which layers make the memory readable, and how
that localization changes with model size, LoRA rank, CE weight, and prompt
compaction.

## Research Questions

1. When top-k KL is enough, and when gold answer CE is required for free-run
   recitation.
2. Which layers are modified by classical KD, measured by per-layer weight-delta
   norms and LoRA adapter norms.
3. Which modified layers are causally important, measured by graft/ablate
   experiments and logit-lens depth profiles.
4. How localization changes across 0.6B, 1.7B, 4B, 8B, 14B, and 32B Qwen3
   models, then later Quijote-scale targets.

## Layout

```
configs/            base + KD experiment YAMLs
data/poem/          raw.txt plus examples.jsonl variants
caches/             teacher top-k logit caches (gitignored)
runs/               experiment outputs and checkpoints (gitignored)
scripts/            dataset/cache/train/evaluate/analyze/report helpers
src/selfupdate/     masking, data, teacher cache, KD train, eval, utils
tests/              alignment / cache / KD loss / online-teacher tests
```

## Method Notes

- Every example is four text segments:
  `shared_prefix | privileged | shared_mid | answer`.
  The teacher sees all four; the student skips `privileged`.
- Token identity between teacher and student is asserted at dataset build time.
  The aligned span is `shared_mid + answer`.
- Qwen3 uses RoPE with full attention. A constant position offset is
  output-invariant, so teacher/student divergence at aligned positions comes
  from attention into the privileged block.
- Student compaction variants are `remove`, `stub`, `stub_gap`, and
  `remove_gap`. The current default is `remove`.
- Disk teacher cache stores only top-k logits and full-row logsumexp for the
  aligned span. LoRA runs can skip the cache with `train.online_teacher: true`;
  adapters off are the frozen teacher, adapters on are the student.
- Full fine-tune KD freezes embeddings, final norm, and lm_head so layer
  localization is measured on transformer blocks.

## Bootstrap

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e . && .venv/bin/pip install pytest
.venv/bin/python scripts/fetch_poem.py
.venv/bin/python scripts/build_dataset.py
.venv/bin/python scripts/build_teacher_cache.py
.venv/bin/python -m pytest tests/ -q
```

Use the cluster interpreter documented in `AGENTS.md` on Agustina. Always set
`PYTORCH_ALLOC_CONF=expandable_segments:True` before training.

## Main Commands

```bash
.venv/bin/python scripts/train.py --experiment configs/experiments/kd_ce_0p6b_rag.yaml
.venv/bin/python scripts/evaluate.py --checkpoint runs/kd_ce_0p6b_rag/checkpoint
.venv/bin/python scripts/analyze.py --deltas kd_full_0p6b_rag kd_ce_0p6b_rag
.venv/bin/python scripts/report.py
```

## Related Work Focus

- Askell et al. 2021 and Snell et al. 2022 define context distillation: train
  a model to imitate its own prompted distribution so the prompt becomes
  parametric behavior.
- Kujanpaa et al. 2024 is the closest document-internalization precedent.
- Stoehr et al. 2024, Huang et al. 2024, ROME/MEMIT, and Hase et al. 2023
  motivate measuring where memories are written and whether causal edit
  locations match storage locations.
- Carlini et al. 2022 and Allen-Zhu & Li 2024 frame scaling expectations for
  poem-to-Quijote memorization capacity.

## Scaling

The KD path scales by keeping the teacher frozen and cheap:

- Cached mode: precompute teacher top-k logits once, then train many KD runs.
- Online-teacher mode: LoRA adapters off/on share one resident model and avoid
  disk caches for large models.
- Full fine-tune KD needs sharding beyond small models; beyond that, use LoRA
  and study layer localization through adapter norms and causal analysis.

See [docs/scaling.md](docs/scaling.md) for the current cluster plan.
