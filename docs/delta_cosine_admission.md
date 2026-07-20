# Delta-cosine GPU admission

This is the smallest fresh placement/numerics gate for `delta_cosine`. It is
mechanics-only: Qwen3-0.6B, the existing 100-item smoke corpus, one epoch,
seed 17, immediate SGD. It is not scientific evidence and is exempt from the
12,000-item campaign floor.

Both legs use the fill-once `store`, stage-scoped loading, the current inline
live-store locality certificate, and the subprocess battery. PPP1 owns all 28
blocks on one GPU. PPP2 cuts after layer 14 across two GPUs. Therefore both
placements execute the interior objective
`cos(student_block_L(x)-x, teacher_h[L]-x)` for middle layers, and both execute
the final layer's absolute post-norm cosine fallback at L28.

The older `h100_smoke_qwen3_0p6b_v4_1proc_e1` and
`h100_smoke_qwen3_0p6b_v4_4stage_e*` artifacts are useful runtime precedents,
but cannot be reused as references: they trained Huber loss on older commits
and predate the live-store certificate. The admission pair must be fresh on
one clean HEAD.

## Launch (delegate each logged launch)

Run sequentially from a clean run identity so both legs start from the same
base snapshot and current commit. Run both on a host carrying the matching
node-epoch0 smoke cache; `_require_node_cache` refuses before model loading if
it is absent. On agpuh01 the matching 100-example Qwen3-0.6B cache identity was
ready when this plan was written.

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/tmp/$USER/selfupdate-venv/bin/python

$PY scripts/train.py \
  --config configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml \
  --experiment configs/experiments/h100_smoke/delta_cosine_0p6b_store_ppp1_e1.yaml

scripts/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml \
  configs/experiments/h100_smoke/delta_cosine_0p6b_store_ppp2_e1.yaml
```

`v4_battery.py` is not launched by hand. `v4_battery_mode: subprocess` makes
the trainer invoke it after every owner publishes its adapter shard, at epoch
zero and epoch one. The one-corpus recall battery is retained; the unrelated
standard-damage probe is disabled to keep admission short.

## Required evidence

For PPP1, inspect `runs/h100_admit_delta_cosine_0p6b_store_ppp1_e1/metrics.jsonl`.
For PPP2, inspect both `stage0/metrics.jsonl` and `stage1/metrics.jsonl` under
`runs/h100_admit_delta_cosine_0p6b_store_ppp2_e1/`.

Require:

- exactly one complete `v4_epoch` per owner with `loss_kind=delta_cosine`;
- finite L14 and L15 losses, and finite L28 loss in both reconstructed
  placements;
- one non-skipped `locality_certification` per owner with `passed=true`,
  `certificate_source=live_fill_once_store`, positive finite local signal for
  every owned layer, zero foreign/frozen-vocabulary gradients, and exact
  adapter/optimizer preservation;
- epoch-zero and epoch-one `v4_battery_subprocess` rows in PPP1 and PPP2 stage
  0, with the expected grafted tensor count and successful child exit;
- checkpoints only after those certificates pass.

Then require exact placement numerics (do not relax tolerance pre-emptively):

```bash
$PY scripts/compare_v4_shard_numerics.py \
  runs/h100_admit_delta_cosine_0p6b_store_ppp1_e1 \
  runs/h100_admit_delta_cosine_0p6b_store_ppp2_e1 \
  --strict-current
```

The comparator should report 28 loss cells, 28 gradient cells, worst relative
delta `0.000e+00`, and PASS. Strict mode also rejects a dirty/different source
tree, a different loss kind, incomplete/overlapping/extra stage ownership, and
missing/skipped/failed or wrongly scoped locality certificates. If it does not
pass, diagnose the first mismatch before considering any tolerance.

## Historical mode

Without `--strict-current`, `compare_v4_shard_numerics.py` retains its original
loss-only behavior for historical artifacts that predate gradient rows or the
current locality certificate. That compatibility mode is archaeology, not an
admission gate.
