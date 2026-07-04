# selfupdate - layerwise forward distillation of context

The same model plays teacher and student. The teacher prompt contains
privileged context - a RAG passage or a visible `<think>` trace - while the
student receives the same prompt with that context hidden. Training asks each
student block to reproduce the teacher's hidden state at aligned token
positions.

The research target in this branch is layerwise forward distillation:
block-local hidden-state learning, plus the smallest local readout auxiliary
needed for free-run behavior. Whole-network logit distillation is not an active
method in this tree.

## The Pierre Menard Program

- Stage 1: memorize *La tierra de Alvargonzalez* (Antonio Machado, 1912) with
  Qwen3-0.6B through larger Qwen checkpoints.
- Stage 2: scale the same masking and layerwise training machinery to
  Don Quijote on 120B-class dense or MoE models.

## Layout

```
configs/            base + layerwise experiment YAMLs
data/poem/          raw.txt + generated examples.jsonl variants
caches/             teacher hidden-state caches (gitignored)
runs/               experiment outputs/checkpoints (gitignored)
scripts/            dataset/cache/train/eval/analysis/scheduler tools
src/selfupdate/     masking, data, teacher cache, layerwise train, eval, utils
tests/              alignment / cache / locality / layerwise hybrid tests
```

## Method Notes

- Every example is segmented as `shared_prefix | privileged | shared_mid |
  answer`. The teacher sees all four segments; the student skips
  `privileged`. The aligned span is `shared_mid + answer`.
- Qwen3 uses RoPE with full attention. A constant position offset is
  output-invariant, so teacher/student divergence at aligned positions comes
  from attention into the privileged block: the signal being distilled.
- The student-side privileged block can be removed, replaced by a stub token,
  or position-rebased. Current evidence favors removal.
- Teacher caches store per-layer hidden states only. Online-teacher LoRA runs
  skip the disk cache: adapters off is the frozen teacher, adapters on is the
  student.
- The core loss is hidden matching (`nmse` or `l2mse`). The active behavioral
  hybrids are local gold-token CE through the frozen readout: last-block CE,
  per-block lens-CE, and tail-CE over a bounded top window.

See [docs/hidden_loss.md](docs/hidden_loss.md) for locality proofs and
[docs/scaling.md](docs/scaling.md) for the large-model plan.

## Current Finding

Storage and readout dissociate. Hidden matching writes distributed,
redundant storage (best metric: `vocab_mse`, MSE in the frozen
vocabulary's coordinates — portable across readouts); a bounded tail
window turns it into behavior, and can even be trained as a second phase
on a frozen fully-local body (`tail_only`). The readout is where the
pathologies live: it is template-locked (cured by maieutic dialogue
data) and intrudes on neighbor-genre text ("catastrophic remembering",
halved by an anchor-KL to the base model). Final recipe and the full
evidence chain: `EXPERIMENTS.md`. 0.6B recites the whole 715-verse
romance self-chained with its first error at verse 708.

## Bootstrap

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e . && .venv/bin/pip install pytest
.venv/bin/python scripts/fetch_poem.py
.venv/bin/python scripts/build_dataset.py
.venv/bin/python scripts/build_teacher_cache.py
.venv/bin/python -m pytest tests/ -q
```

On the L40S cluster, use the interpreter and CUDA-wheel guidance in
`AGENTS.md`; do not rely on `/usr/bin/python3`.
