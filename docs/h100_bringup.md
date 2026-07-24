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
- The v4.6 trainer always uses the synchronous live-owner battery for staged
  evaluation. There is no battery-mode knob or reconstructed fallback.
  Evaluation runs without backward and with optimizer weight zero; rotary,
  shared-KV, per-layer-input, hybrid-cache, and mHC state stay stage-owned.
- The campaign gate and its evaluation are entirely in-training.  Do not
  merge adapters or create a checkpoint merely to evaluate a demo run;
  merging is an explicit later publication/export operation, not part of
  Slurm training or certification.

## 7. sbatch campaign templates

Campaign sbatch scripts under `scripts/` (e.g. `spec_g26b_a4b_campaign.sbatch`)
are committed as reusable templates and must carry no host- or account-
specific value: no absolute paths, no Slurm account/partition/nodelist. Two
constraints shape this:

- `#SBATCH` directive lines are parsed by `sbatch` at submission time, before
  the script runs as a shell script — they CANNOT reference shell variables.
  `#SBATCH --account=$SLURM_CAMPAIGN_ACCOUNT` silently does not work. Account,
  partition, and node selection must therefore be passed on the `sbatch`
  command line, not baked into the script header.
- Everything else host-specific (the SSL cert bundle path, the vLLM
  interpreter path) is read from `.campaign_env.sh` at the repository root —
  gitignored, one per checkout/account, never committed. The script sources
  it and fails loudly (`: "${VAR:?...}"`) if a required variable is unset.
  Copy `.campaign_env.sh.example` to `.campaign_env.sh` and fill in real
  values once per checkout.

Required variables (documented here, defined in `.campaign_env.sh`):

| variable | meaning |
|---|---|
| `SLURM_CAMPAIGN_ACCOUNT` | `sbatch --account=...` value for this cluster account |
| `SLURM_CAMPAIGN_PARTITION` | GPU partition to submit to (e.g. the H100 partition) |
| `SLURM_CAMPAIGN_NODELIST` | optional; pin a specific node, or leave empty to let Slurm place it |
| `SELFUPDATE_SSL_CERT_FILE` | certifi bundle path for this account's Python HTTPS (see AGENTS.md) |
| `SELFUPDATE_VLLM_PYTHON` | interpreter with vLLM installed, for the generation stage |

**vLLM is deliberately NOT vendored in this repo and NOT rebuilt in `/tmp`,**
unlike the main training venv (`scripts/venv_setup.sh`, which genuinely is
built fresh per node in ~30s). vLLM's build is complex and driver-dependent —
this account keeps several pre-built variants side by side at
`/fs/agustina/arivero/supercomplex/venvs/` (`vllm025`, `vllm126`, `vllmAda`,
plain `vllm`, ...) for different torch/CUDA/hardware combinations, each the
product of real trial-and-error, not something to casually reproduce.
`SELFUPDATE_VLLM_PYTHON` should point at one of these existing persistent
installs (`vllm025` = vllm 0.25.0+cu129, confirmed working for gemma-4 answer
generation on H100 as of 2026-07-23) — never attempt to `pip install vllm`
into the node-local training venv or build a fresh one in `/tmp` as part of a
campaign script.

Submit with:

```bash
source .campaign_env.sh
sbatch --account="$SLURM_CAMPAIGN_ACCOUNT" \
       --partition="$SLURM_CAMPAIGN_PARTITION" \
       ${SLURM_CAMPAIGN_NODELIST:+--nodelist="$SLURM_CAMPAIGN_NODELIST"} \
       scripts/spec_g26b_a4b_campaign.sbatch
```

Run `sbatch` from the repository root.  Slurm executes a spool copy of the
script, so the templates deliberately derive `ROOT` from `SLURM_SUBMIT_DIR`,
not from `$0` or `BASH_SOURCE`.

The campaign templates execute this order inside the allocation: build the
node-local training venv; stage model snapshots to `/dev/shm`; generate the
answers; build one immutable durable teacher cache; copy that *specific cache
identity* to `/dev/shm/$USER/selfupdate-teacher-cache`; then run the PPP gate
and its one-process reference through `launch_v4_stages.sh`.  The
`SELFUPDATE_TEACHER_CACHE_ROOT` override changes only the physical cache root,
not the cache identity.  On a multi-node PPP8 allocation, each allocated node
must receive its own copy before remote stages start; model `HF_HOME` and the
teacher-cache root are both forwarded to those stages.

The stage launcher detaches the individual worker processes but the sbatch
script retains the Slurm allocation and watches their lease until completion.
The 120-minute bound applies independently to each one-epoch numerics-gate
side; the allocation itself has the 24-hour Slurm wall time.  Current
`v4_progress` rows are written to each stage's `metrics.jsonl` every 60
seconds when the config enables them; use those rows for progress rather than
inferring it from GPU utilization.

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
