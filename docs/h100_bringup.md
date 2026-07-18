# H100 bring-up: training a fresh checkout on `agpuh01`

Written 2026-07-17 bringing branch `lwteacher` (checkout
`/fs/agustina/arivero/supercomplex/selfup_teacher`) up on the Hopper node from
nothing. A later agent landing on a fresh checkout should not have to
rediscover any of this. "The problem is usually the python" — this file is
about the python.

Read `AGENTS.md` first for the laws; this file is the recipe and the traps.

## What a fresh checkout does NOT have

Every runtime artifact is gitignored, so a new clone/worktree has none of it:

| Missing | Why | Fix |
|---|---|---|
| Python runtime | gitignored | `scripts/venv_setup.sh` (~30 s) |
| `caches/` (teacher states) | gitignored | `scripts/build_teacher_cache.py` |
| `runs/<run>/` | gitignored | created by the run |

Model weights are the one thing already present: the account HF cache under
`$HOME/.cache/huggingface` holds `Qwen3-0.6B`, `Qwen3.5-0.8B`, `Qwen3.5-4B`,
`Qwen3.6-27B` and others. Nothing needs downloading for a smoke.

## Step 1 — the venv (~30 s)

```bash
scripts/venv_setup.sh
scripts/venv_check.sh
```

`venv_check.sh` verifies the interpreter, that torch's CUDA build matches the
node driver, that a real bf16 CUDA matmul executes, that the pins are exact,
and that `selfupdate` resolves to THIS checkout. Measured on `agpuh01`
2026-07-17 (driver 565.57.01, 4x H100 80GB HBM3): Python 3.12.10, torch
2.11.0+cu128 / CUDA 12.8, transformers 5.12.1, peft 0.19.1, accelerate 1.14.0,
kernels 0.12.0.

Then run everything from the repo root through that interpreter:

```bash
PY=/tmp/$USER/selfupdate-venv/bin/python
$PY scripts/train.py --config ... --experiment ...
```

Timings measured on `agpuh01`: venv create 2.4 s, torch install 29.7 s, the
rest 1.4 s — about 33 s total for a 6.8 GB venv. Cold `import torch` +
`import selfupdate` ~21 s; warm, negligible.

### Why /tmp, why not Lustre, why not a container

- **Never on Lustre.** A venv is tens of thousands of small files and Lustre's
  metadata cost dominates. A cold `import torch` from the Lustre venv at
  `selfupdate_lw/.venv` was measured **not finishing within two minutes** on
  agpuh01. The same content in `/tmp` imports in ~20 s cold.
- **Create, never copy.** A venv bakes absolute paths into `pyvenv.cfg` and
  every console-script shebang; it cannot be relocated. `/tmp` is node-local,
  so build once per node. It is disposable — rebuild, never repair.
- **No container.** The Singularity runtime was deleted 2026-07-17. It was
  host-dependent (cu128 image fails on driver-560 L40S), shipped no `git`
  (which breaks `runlog.py`, see below), and `--cleanenv` silently dropped
  host env vars. Do not reintroduce it.

## Step 2 — the teacher cache is the real blocker

`/dev/shm` is node-local. The ada campaign's caches live on `agpul04/05/06`
and are **unreachable from agpuh01**; there is no teacher cache on Lustre in
either checkout. Anything that trains needs one — even `scripts/v3_smoke.py`
calls `rt.load_cache()`.

The campaign configs (`pareto_v3/*`, `layerwise34_timing/*`) use
`cache.runtime_policy: node_epoch0` plus a `cache.generation_responses_path`
pointing at vLLM outputs under `runs/` that do not exist in this tree. Do not
chase those files for a smoke: `generation_responses_path` is **optional**
(default `""`), and when empty `build_teacher_cache.py` generates the teacher
answers itself with a greedy forward. Slower, but self-contained.

```bash
$PY scripts/build_teacher_cache.py \
  --config configs/experiments/h100_smoke/base_qwen3_0p6b_lora.yaml \
  --experiment configs/experiments/h100_smoke/qwen3_0p6b_pp1_serial.yaml
```

Measured: ~13 generated tok/s at batch 36 for Qwen3-0.6B, so the 100-item
smoke set takes roughly ten minutes end to end.

## Step 3 — the smoke configs

`configs/experiments/h100_smoke/` holds a self-contained base plus two
**placement-only** overlays. `load_config(base, experiment)` merges exactly one
overlay onto one base, which is why the base is self-contained rather than
layered on `configs/base.yaml`.

| File | Role |
|---|---|
| `base_qwen3_0p6b_lora.yaml` | Qwen3-0.6B, pipeline-v3.2, LoRA, 1 epoch, 100 items, durable cache |
| `qwen3_0p6b_pp1_serial.yaml` | one H100, `pp_execution: serial` |
| `qwen3_0p6b_pp3_wavefront.yaml` | three H100s, `pipeline_splits: [9, 18]`, `pp_execution: wavefront` |

The overlays change placement only, so the pair is a real
same-method/different-placement comparison — mirroring
`scripts/train_certify.py`, whose semantic config hash deliberately excludes
`pipeline_split(s)`/`device`/`run_name` because PP is meant to be
numerics-neutral.

Validate before spending a GPU-minute (loads no weights):

```bash
$PY - <<'PY'
import sys; sys.path.insert(0, "src")
from selfupdate.config import load_config
from selfupdate.train.validate import validate_knob_schedule
BASE = "configs/experiments/h100_smoke/base_qwen3_0p6b_lora.yaml"
for name in ("qwen3_0p6b_pp1_serial", "qwen3_0p6b_pp3_wavefront"):
    cfg = load_config(BASE, f"configs/experiments/h100_smoke/{name}.yaml")
    validate_knob_schedule(cfg)
    print("OK", name, cfg.model.pipeline_splits, cfg.model.pipeline_devices, cfg.train.pp_execution)
PY
```

## Traps that cost real time

- **`cache.source_compaction` is a training-READER selector, not a build
  knob.** `teacher/cache.py` resolves `source_compaction or mask.compaction`
  into the cache identity hash. The campaign configs pair `source_compaction:
  remove` with a `flow_mask` student because their v3 `node_epoch0` path builds
  a deliberate CROSS-VIEW cache under `--coordinated-node-cache`. A durable
  cache has no such licence and the builder rejects it:
  *"cache.source_compaction is a training-reader selector; cache generation
  must leave it empty or equal mask.compaction"*. Copying that knob without its
  enabling context is the "defaults are experiment variables" trap.
- **Do not narrow `CUDA_VISIBLE_DEVICES` for a PP run.** Trainers consume
  physical ids from `model.pipeline_devices`; narrowing renumbers cards to
  logical `0..n` and the runtime correctly rejects the mapping. Launch PP with
  `CUDA_VISIBLE_DEVICES` unset. (Same trap bit the ada campaign — see the
  retained traceback note in `docs/layerwise34_overnight_handoff.md`.)
- **`pp_execution: wavefront` is gated to v3.2** (`train/validate.py`): it also
  requires `history_policy: causal_bk`, `update_granularity: online`,
  `stale_gradient_window > 0`, `grad_accum: 1`, `schedule: summed`. Multi-stage
  wavefront additionally needs a non-empty `train.partition_profile_id` — a
  free-form provenance label, not a lookup key.
- **`layerwise_project_version` must be `"3.4"`** or validation raises.
  Self-contained bases must set it.
- **This checkout is a git WORKTREE**, not a clone: `.git` is a *file* pointing
  at `selfupdate_lw/.git/worktrees/selfup`. Anything that resolves provenance
  by reading `.git/HEAD` must follow that indirection, and anything that cannot
  see the parent repo cannot resolve it at all. `runlog.py` shells out to `git`
  (`diff`, `ls-files`, `rev-parse`) for every v2/v3 run, so **git must be on
  PATH** — it is, on the host. This is a large part of why the container was
  removed.
- **`echo "EXIT=$?"` after a command masks its failure** — the echo's own exit
  code is what gets reported. Verify the artifact, not the exit code. A
  "successful" teacher-cache build wrote no cache at all this way.
- **Backgrounding a launch from inside a subagent loses it**: the job dies with
  the subagent's shell — empty log, no process.
- **PP distribution must be verified, not assumed.** 0.6B fits on one card, so
  a pipeline collapsing onto one device is a false pass. Check all three cards
  hold memory:
  `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader`.
- Keep `TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error`
  set whenever an agent captures output.

## Scope warning

The smoke base trains 100 items for 1 epoch. That is **mechanics only** — far
below the 12,000-item floor `AGENTS.md` sets for any comparison. No loss curve,
recall number, or loss-kind claim may be drawn from it. It answers exactly one
question: does this checkout train, on one H100 and as PP3.

## Warming the Lustre vLLM venv (and the ssh landing-dir trap)

A cold `import vllm` from the Lustre `venvs/vllm025` stalls for 10-15+
minutes in importlib metadata round trips (a process shows state I, seconds
of CPU over minutes of wall). Fix: run the parallel warmer FIRST, and
re-run it after any large model load (page-cache eviction resets the win):

    bash /fs/.../selfup_teacher/scripts/warm_python_runtime.sh \
        /fs/.../venvs/vllm025/bin/python vllm

Measured 2026-07-18: the stalled DeepSeek demo import unblocked minutes
after the warmer ran; a warm import takes ~1 min.

TRAP: `ssh host 'scripts/foo.sh ...'` resolves relative paths from the ssh
LANDING directory (here /fs/.../supercomplex, not the repo) and dies with
"No such file" — three demo launches and two warmer attempts failed this
way in one night. Remote invocations must use wrapper scripts with
absolute paths (see scripts/demo_deepseek_retry.sh) or an explicit cd.

## Launch-expectation recipe (owner, 2026-07-18)

A launch is a prediction. Before every launch, state the expected
observable timeline — which line appears in which log at roughly what
time — and act on the FIRST discrepancy:

1. t+0: launcher prints one `stage k -> pid` line per stage. A missing
   line is itself the failure; do not wait for later milestones.
2. t+seconds: each stage log opens with this launch's separator
   (`==== launch <id> stage <k> <timestamp> ====`) followed by the runlog
   header row in its metrics.jsonl. Anything ABOVE the separator is a
   previous attempt — never triage from it.
3. t+1-10 min: the GPU-occupancy map (nvidia-smi compute-apps, per host)
   matches the device plan exactly — right pids on right cards, nothing
   extra, nothing missing. This check catches what log-reading cannot
   (frozen spawn loops, pattern-missed zombies, stray contexts).
4. t+capture-window: first capture/epoch rows per stage.

When a discrepancy needs more signal, raise verbosity SELECTIVELY for
the suspect section (TRANSFORMERS_VERBOSITY, NCCL_DEBUG=INFO for
transport bring-up, per-script --debug flags) instead of relaunching
blind. Every 2026-07-18 mis-triage was visible at the first broken
expectation; the cost was incurred by waiting for the next one.
