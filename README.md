# selfupdate — v4.6 teacher-forced block distillation

This checkout has one training law.  For every block `L`, the input and target
are adjacent hidden states from the frozen teacher:

```text
teacher h[L-1] -> trainable block L -> compare with teacher h[L]
```

The trainable student block output carries gradients through that block's
weights, but no training loss consumes the student's own upstream trajectory.
Blocks and answer positions are therefore independent, so multi-GPU execution uses PPP:
independent processes own disjoint contiguous block ranges.  There are no
training activation boundaries, student relays, or wavefront dependencies.

The student still runs an ordinary full censored forward pass for validation
and generation.  Those states flow through the full model to predict tokens;
the resulting cross-entropy/KL, recall, and damage measurements have zero
optimizer weight and never enter backward.

## Scientific contract

- Inputs are detached teacher `h[L-1]`; targets are teacher `h[L]`.
- Attention context is teacher-recorded and detached.  `student_refresh`, when
  selected, refreshes projections of teacher inputs; it is not a student
  hidden-state trajectory.
- Only the owned block receives gradients.  Embedding, final norm, vocabulary
  head, foreign blocks, and all validation paths remain frozen.
- Censorship is `flow_mask`; `intact` is the diagnostic control.
- `CE-eval-loss` and `KL-eval-loss` are whole-training-set evaluation metrics,
  never objectives.
- The staged checkpoint manifest assigns every block to exactly one owner;
  `scripts/merge_v4_adapters.py` merges by ownership, never averaging.

The authoritative protocol is
[docs/training_pipeline_v4.md](docs/training_pipeline_v4.md).  See
[docs/runtime.md](docs/runtime.md) for execution and
[docs/programmer_walkthrough.md](docs/programmer_walkthrough.md) for a compact
code tour.  Earlier v1–v3 protocols remain in explicitly archived documents
and Git history only; their training code and active configurations are not in
this source tree.

## Layout

```text
configs/            v4 experiment and evaluation YAMLs
data/               datasets and vendored evaluation inputs
caches/             teacher caches (gitignored)
runs/               run outputs/checkpoints (gitignored)
scripts/            v4 cache/train/eval/analysis/launcher tools
src/selfupdate/     v4 runtime, teacher store, evaluation, and utilities
defactorised/       frozen pre-clean standalone teaching snapshot
compressed/         compressed companion to that frozen snapshot
```

The last two directories deliberately preserve the script genealogy produced
before the v4 source cleanup.  They are demonstration/archaeology artifacts,
not alternate supported training entry points.

## Runtime

Create the disposable node-local environment; never install the package
editable and never put the venv on Lustre:

```bash
scripts/venv_setup.sh
scripts/venv_check.sh
PY=/tmp/$USER/selfupdate-venv/bin/python
$PY scripts/audit_configs.py
```

Every entry point pins this checkout's `src` itself.  See
[docs/h100_bringup.md](docs/h100_bringup.md) for model/cache staging and the
CUDA/runtime pins.

Single-process v4 training:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
$PY scripts/train.py --config configs/base.yaml \
  --experiment configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml
```

Independent PPP stages use `scripts/launch_v4_stages.sh`; the stage count,
physical devices, ownership cuts, relay transport, teacher source, and
residency are read from the experiment config.  The launcher is coordination
only: training itself has no inter-stage student activation edge.

## Evaluation and certification (v4.6)

The trainer performs the ordinary live PP student battery at synchronized
epoch boundaries with the live stage-owned weights; rotary and
architecture-specific cache side channels stay native. There is no
reconstructed evaluator. `scripts/compare_v4_shard_numerics.py` compares
single-process and independent-shard artifacts;
[certs/README.md](certs/README.md) describes the current on-demand workflow.
Locality certification runs at training end and withholds checkpoint
publication if gradients escape the owned block or touch the frozen
vocabulary stack.

Migration from removed evaluator entry points is documented in
[`docs/v4_6_migration.md`](docs/v4_6_migration.md).

Historical experiment evidence remains in `EXPERIMENTS.md`, `issues.md`, and
the dated report documents.  It is evidence about retired protocols, not a
way to re-enable them in this checkout.
