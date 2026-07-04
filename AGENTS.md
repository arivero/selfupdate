# Agent guide - selfupdate layerwise branch

Orientation for a fresh agent/session. Read `README.md` for the science; this
file is operational.

## Source Of Truth

Everything needed lives in this repo. Do not depend on host-local agent memory.

- `EXPERIMENTS.md` - live layerwise plan and status board.
- `docs/hidden_loss.md`, `docs/scaling.md`, `docs/memory.md` - loss math,
  locality proofs, scale plan, memory accounting.
- `runs/results.md`, `runs/report.pdf`, `runs/curves.png` - generated
  artifacts when present.
- `runs/*/metrics.jsonl` and `runs/pipeline_*.log` - raw dynamics and history.
- `runs/*/checkpoint` - checkpoints, gitignored.

`.venv/` is gitignored and may be a symlink to a sibling checkout's venv
(cloning a venv on Lustre is all small-file metadata cost — don't). The
shared venv carries **no** editable install of `selfupdate`: a bare
`import selfupdate` fails loudly by design. Every entry point pins its own
tree instead — `scripts/*.py` via `sys.path.insert(0, <repo>/src)` and
pytest via `tests/conftest.py`. Keep that guard in any new script; never
`pip install -e .` into the shared venv (it would silently route imports
across checkouts). The bootstrap's `pip install -e .` applies only to a
fresh per-tree venv.

## Branch Focus

This branch is for layerwise forward distillation only. Active training methods
are in `src/selfupdate/train/layerwise.py`:

- `summed`
- `sequential`
- `teacher_censored`
- bounded `tail_ce_blocks`
- local `last_block_ce` / `lens_ce` auxiliaries

Do not reintroduce non-layerwise training configs, queues, docs, or dispatch.

## Publication-Critical Constraints (owner directives, 2026-07-04)

- **Never train only the last k blocks "under any subterfuge."** A
  tail-only readout window with CE is, to a referee, classical
  distillation of the top layers — it invalidates the layerwise/forward
  claim. The sanctioned form is the SLIDING k-connected window
  (`conn_window` + `conn_stride: 1`): every block updated with uniform
  k-deep credit; the top window carries the CE only because logits exist
  there. Tail-only arms are allowed solely as labeled ABLATIONS.
- The embedding and logits matrix are never trained, in any window
  scheme (Frozen-Vocabulary Principle; four locks + runtime tripwire).

## Hard-Won Lessons

- Strict hidden matching stores signal but weakly recites; the readout is the
  hard part.
- Tail-CE is the current best lever: keep hidden matching as storage, connect a
  small final block window for behavioral credit.
- `teacher_censored` is useful both as a schedule and as a localization
  readout; context integration peaked near layer 7 in Qwen3-0.6B artifacts.
- Eval on the full corpus. The 8-example training subset can hide severe
  coverage bias.
- Two concurrent GPU jobs need a VRAM guard with random stagger.
- `pkill -f pattern` can kill the invoking shell if the target appears in the
  command line; use bracketed patterns.
- VRAM checks at launch underestimate peak because optimizer state appears at
  the first step.
- Do not sweep Lustre with broad recursive search. Search inside the repo only,
  and prefer `git ls-files` / `rg`.
- `kernels` must stay `==0.12.0` with transformers 5.12.1: 0.16 breaks ALL
  model loading (`ValueError: Either a revision or a version ...`).
- Scheduler VRAM reservations are launch-time checks, not leases: a 40GB
  job can be OOM'd later by a 3GB eval placed into its margin. Exclusive
  jobs (gpt-oss-class) should run when the queue is otherwise drained, or
  use the n_gpus exclusivity path.
- Python HTTPS on this cluster needs
  `export SSL_CERT_FILE=/fs/agustina/arivero/supercomplex/.local/lib/python3.11/site-packages/certifi/cacert.pem`
  (urllib otherwise rejects the proxy chain). curl works without it;
  scripts/fetch_quijote.py keeps a curl fallback as belt-and-braces.
- GPU tests contend with campaign jobs on cuda:0 — pin with
  `CUDA_VISIBLE_DEVICES=<free>` when lanes are busy.

## L40S Cluster Environment

- `$HOME` is on Lustre. Use `~/...`, not `/home/...`.
- `/usr/bin/python3` is 3.6.8. Use the venv or
  `/opt/ohpc/pub/apps/anaconda/anaconda-2025/bin/python3`.
- Driver 560.35 = CUDA 12.6. Install PyTorch wheels with the cu128 index when
  rebuilding the venv.
- No nvcc on PATH by default; CUDA modules exist but pip wheels normally bundle
  runtime libraries.
- Scheduler pattern:

```bash
GPUS="0 1 2 3" MAX_PER_GPU=3 nohup setsid bash scripts/gpu_scheduler.sh >> runs/pipeline_sched_main.log 2>&1 &
```

## Bootstrap

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e . && .venv/bin/pip install pytest
.venv/bin/python scripts/fetch_poem.py
.venv/bin/python scripts/build_dataset.py
.venv/bin/python scripts/build_teacher_cache.py
.venv/bin/python -m pytest tests/ -q
```

Always set this before training:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

Online-teacher LoRA runs (`train.online_teacher: true`) need no teacher cache.

## Multi-Node Conventions

- This cluster does not constrain GPU devices by cgroup. Pass physical GPU ids
  verbatim; do not renumber `CUDA_VISIBLE_DEVICES`.
- One queue file per node/lane.
- Per-node scheduler state: `SCHED=runs/.sched-$(hostname -s)`.
- Per-node job log: `JOBLOG=runs/pipeline_sched_<host>.log`.
- `evaluate.py --base` needs lane-specific `--out` paths during concurrent
  base evals.
- Results refresher/report shipper should run on only one node.

## Operational Conventions

- Never abort a training run before it has seen at least 12,000 training items.
  Early noisy plateaus have recovered in the layerwise tail runs, and matched
  item budget is needed for comparisons.
- Configs are `configs/base.yaml` plus small YAMLs in `configs/experiments/`.
- Run outputs land in `runs/<run_name>/`.
- Long work runs detached via `nohup setsid ... >> runs/pipeline*.log 2>&1 &`.
- Re-run tests after changes touching masking, aligned spans, cache layer-index
  conventions, or detach discipline in `train/layerwise.py`.

## Hardware Ladder

- 0.6B: mechanics, locality tests, strict-vs-tail ablations.
- 1.7B/4B/8B: readout-window scaling and memory curve.
- 14B/32B: online-teacher LoRA, sharding where needed.
- MoE/120B-class: streamed blocks, post-combine hidden matching, Don Quijote.

## Current Pointer

The final campaign recipe (2026-07-04): `hidden_loss: vocab_mse` +
maieutic v4 data (`examples_v4.jsonl`) + `tail_ce_blocks: 4` +
`anchor_kl_weight: 0.5`, `tail_ce_blocks: 8` when all three readout
properties are needed at once (+ `frozen_teacher_copy` for full-FT). Two-phase
form (strict body then `schedule: tail_only` with `init_from`) matches or
beats joint training and keeps storage fully block-local. Next work:
scale the final recipe (1.7B+, families), the reasoning-family question,
thinking_selective masking (designed, unbuilt — see plan file).
