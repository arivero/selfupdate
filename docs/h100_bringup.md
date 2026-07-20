# H100 bring-up for pipeline v4

This is the current host-runtime checklist.  Retired PPn/v1–v3 launch recipes
are available in Git history, not as executable guidance in this checkout.

## 1. Verify the allocation and device

Use physical CUDA ids; this cluster does not renumber them by cgroup.

```bash
hostname -s
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total \
  --format=csv
```

Do not infer availability from a launch-time reservation.  Check live compute
PIDs and VRAM immediately before a run.  A v4 process must be pinned to the
physical id recorded in its config/launcher environment.

## 2. Build the node-local runtime

```bash
scripts/venv_setup.sh
scripts/venv_check.sh
PY=/tmp/$USER/selfupdate-venv/bin/python
```

The venv and UV/compiler caches belong under `/tmp/$USER`, never Lustre.  Do
not copy a venv between nodes and do not `pip install -e .`; entry points pin
this checkout's `src`.  The current pins are recorded in `AGENTS.md`.

For captured probes/log inspection, suppress progress bars:

```bash
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

## 3. Stage model and teacher material separately

Model snapshots and teacher states are different resources:

- stage model files with `scripts/stage_hf_cache.sh` and point `HF_HOME` at
  the completed node-local stage;
- choose `train.v4_teacher_source`: `cache`, `online`, or fill-once `store`;
- for `cache.runtime_policy: node_epoch0`, publish the matching cache under
  the configured `/dev/shm` root before any model load;
- for `store`, the PPP launcher coordinates the one-time teacher relay and
  each stage retains only its owned teacher material.

Never copy the historical full cache root.  Cache identity includes the
dataset, masking view, generated answer ids, hidden layout, and runtime
compatibility.  A failed identity gate is repaired by rebuilding the intended
cache, not by weakening the check.

## 4. Validate without loading weights

```bash
$PY scripts/audit_configs.py
```

Every trainable config must satisfy pipeline v4.  In particular, the block
input is teacher `h[L-1]`, the differentiable output is produced by student
block L, and the target is teacher `h[L]`.  `model.pipeline_split(s)` and all
student-trajectory training protocols are rejected.

## 5. Launch

Single process:

```bash
$PY scripts/train.py \
  --config configs/base.yaml \
  --experiment configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml
```

Independent PPP stages:

```bash
scripts/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml \
  <v4-overlay.yaml>
```

Use the launcher's actual `--help` when selecting optional arguments.  Stage
count and ownership come from `v4_stage_splits`; physical ids come from
`v4_stage_devices`.  Each process owns a disjoint block range.  No training
activation crosses stages; relay traffic is teacher-store coordination or
the separate full-forward validation pass.

Any launch expected to produce a log must be delegated to a small-class agent
as required by `AGENTS.md`.  The owning agent still reviews fresh logs,
liveness, checkpoints, evaluation completion, and scientific telemetry.

## 6. Certify the result

- Training-end locality certification must pass before checkpoint publication:
  foreign blocks and embedding/norm/head gradients are exactly zero.
- Compare equal-seed single-process and PPP artifacts with
  `scripts/compare_v4_shard_numerics.py`.
- Run the v4.5 trainer with `v4_battery_mode: distributed` for the ordinary
  censored live-student token metrics. Unsupported architectures select the
  trainer-owned reconstructed fallback. Both run without backward and have
  optimizer weight zero.
- Merge staged adapters with `scripts/merge_v4_adapters.py`; ownership is
  disjoint, so merge selects tensors rather than averaging them.

## Common traps

- A bare `cuda` device string may not identify the physical card intended by
  the run; use the configured physical id.
- `CUDA_VISIBLE_DEVICES` remapping conflicts with the repository's physical
  NVML ownership tripwire.
- Store-fill time and first-load cache warmth are not steady-state training
  speed.  Report them separately.
- Weight staging does not warm Python imports; use
  `scripts/warm_python_runtime.sh` before a multi-worker cold start.
- Keep TorchInductor/Triton caches node-local and cap compiler/native CPU
  thread pools as described in `AGENTS.md`.
