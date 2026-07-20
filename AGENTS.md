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

The Python runtime is a node-local venv in `/tmp`, built by
`scripts/venv_setup.sh` — never a venv on Lustre, and never a container. See
the Python Runtime section below for the reasoning and the pins. No venv is
tracked or expected inside the checkout: `.venv/` is gitignored and a fresh
tree deliberately has none.

The venv carries **no** editable install of `selfupdate`: a bare
`import selfupdate` fails loudly by design. Every entry point pins its own
tree instead — `scripts/*.py` via `sys.path.insert(0, <repo>/src)`
(`tests/` and its conftest were deleted 2026-07-11, see Training Runtime
section). Keep that guard in any new script; never `pip install -e .` into the
venv (it would silently route imports across checkouts).

## Python Runtime

Build a node-local venv and run everything through its interpreter:

```bash
scripts/venv_setup.sh                       # ~30 s, once per node
scripts/venv_check.sh                       # verifies torch/CUDA/pins/imports
/tmp/$USER/selfupdate-venv/bin/python scripts/train.py --config ... --experiment ...
```

There is NO container runtime on this branch. The Singularity SIF/overlay
setup was deleted 2026-07-17: it was host-dependent (the cu128 image fails on
the driver-560 L40S nodes), it shipped no `git` binary — which broke run-log
provenance outright, since `runlog.py` shells out to `git` for every v2/v3 run
— and under `--cleanenv` it silently dropped host environment variables. In a
git WORKTREE checkout it was unrecoverable: `.git` is a file pointing into the
parent repo, which the container never bound, so even the in-container
`.git/HEAD` fallback could not resolve. The venv has none of these problems:
it runs on the host, where `git` simply works. Do not reintroduce it.

**The venv MUST live in node-local `/tmp`, never on Lustre.** A venv is tens
of thousands of small files and Lustre's metadata cost dominates: a cold
`import torch` from a Lustre venv was measured NOT finishing within two
minutes on agpuh01, while the same venv in `/tmp` builds in ~30 s and imports
in ~20 s cold. **A venv cannot be moved** — `pyvenv.cfg` and every
console-script shebang bake in absolute paths — so create one per node rather
than copying. `/tmp` is node-local on the tested nodes, so this is once per
node. It is disposable: delete and rebuild, never repair.

Pinned by `scripts/venv_setup.sh`, measured on `agpuh01` 2026-07-17
(driver 565.57.01, H100 80GB HBM3):

- Python 3.12.10, `torch==2.11.0+cu128` (CUDA 12.8), from the cu128 index.
- `transformers==5.12.1`, `accelerate==1.14.0`, `peft==0.19.1`,
  `kernels==0.12.0`, plus safetensors/pyyaml/pandas/tabulate/matplotlib/tqdm.
- `kernels` must stay `==0.12.0` with transformers 5.12.1: 0.16 breaks ALL
  model loading (`ValueError: Either a revision or a version ...`).
- The torch pin is the one dependency that must match the node's driver. cu128
  needs a >=12.8-capable driver. Check `nvidia-smi` before assuming a node.

`uv` (`/fs/agustina/arivero/supercomplex/.local/bin/uv`) does the resolve and
install; it is what makes the ~30 s build possible. Keep `UV_CACHE_DIR` in
`/tmp` too. Python HTTPS on this cluster needs `SSL_CERT_FILE` set to the
certifi bundle (`venv_setup.sh` does this).

The venv deliberately carries **no** editable install of `selfupdate`: a bare
`import selfupdate` fails loudly by design. Every entry point pins its own
tree instead — `scripts/*.py` via `sys.path.insert(0, <repo>/src)`. Keep that
guard in any new script; never `pip install -e .` into a shared venv (it would
silently route imports across checkouts). One venv can serve several checkouts
precisely because of this.

Verify a new node with `scripts/venv_check.sh`, or by hand:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv
/tmp/$USER/selfupdate-venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print((x @ x).float().mean().item())
PY
```

Model snapshots resolve from `$HOME/.cache/huggingface` (Lustre) by default;
stage hot models to `/dev/shm` or `/tmp` with `scripts/stage_hf_cache.sh` and
point `HF_HOME` at the stage. Keep TorchInductor/Triton caches node-local
(`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` under `/tmp/$USER`).

Full bring-up recipe, including the teacher-cache bootstrap and the traps that
cost a session, is in `docs/h100_bringup.md`.

## Training Runtime & Certification (v4-only, 2026-07-20)

`src/selfupdate/train/layerwise.py` is a thin v4 entry point;
`online_v4.py` owns the teacher-hidden block steps and validation relay;
`runtime.py` owns loading/cache/frozen-vocabulary/save; `v4_store.py` owns the
fill-once teacher store; `rotation.py`/`shard_load.py` own scaling transport;
and `validate.py` rejects every non-v4 training configuration. Read
`docs/runtime.md` before touching execution machinery.

The training law is structural: block L consumes detached teacher h[L-1]; the
trainable student block produces its local output with gradients through its
own weights; that output is matched to teacher h[L]. What exists only in the
no-grad validation/generation path is the student's end-to-end trajectory.
Never feed one student block's output into a later training block.

There is NO stored test or certification gate (owner decision 2026-07-11:
tests and stored fingerprints act as specifications that agents ossify
around; both were deleted — `tests/` in fd7138d, `certs/pre`+`certs/pp2`
in this pass; git history keeps them). The specification is the prose laws
in this file plus the RUNTIME enforcement in the code: `_validate_knob_schedule`
raises at dispatch, the frozen-vocab fingerprint tripwire at save, the
graph-leak/MoE tripwires in the walk, `scripts/audit_configs.py`.

For a numerics-preserving trainer change, mint fresh single-process and PPP
artifacts on current HEAD and compare them with
`scripts/compare_v4_shard_numerics.py`; then run the ordinary token-prediction
battery with `scripts/v4_battery.py`. References are disposable and never
stored as a frozen specification. See `certs/README.md`.

## Branch Focus (current, 2026-07-20)

This branch is pipeline-v4 only: teacher h[L-1] -> owned block L -> teacher
h[L], with frozen teacher attention context and no cross-block training graph.
PPP means independent block-owner processes, not activation pipelining.
Local hidden objectives including `lens_kl` remain permitted through the
frozen vocabulary measurement stack.

Behavioral readout and final-logit training are not active methods. The
readout runtime has been deleted; the old implementation remains recoverable
from Git history for archaeology only. Do not add `readout_*` keys to new
configs or revive that runtime on this branch.

Do not reintroduce non-layerwise training configs, queues, docs, or dispatch.

## Publication-Critical Constraints (current policy)

- Every optimizer objective is block-local: its input is detached teacher
  h[L-1], its target is teacher h[L], and backward may update only block L's
  trainable parameters. End-to-end student trajectory states are
  validation-only; the local student block output is the differentiable side
  of every v4 training loss.
- `lens_kl` is permitted only as a local metric through the frozen final norm
  and LM head. The head, embedding, and logits matrix never receive updates,
  and the metric must not create a graph across blocks.
- No behavioral readout, final-logit objective, teacher-KL readout, or
  reference-text log loss is a training target. Original text is evaluation
  reference only; teacher states are the training source. The readout runtime
  deletion is intentional and recoverable from Git, not a missing experiment.
- `CE-eval-loss` and `KL-eval-loss` are output-distance EVALUATION ONLY. They
  are measured over every teacher-realized answer token in the WHOLE training
  set once per completed epoch during its ordinary traversal; they are not a
  validation subset. They NEVER enter `HiddenLoss`, backward, gradient
  accumulation, parameter writes, learning-rate selection, or any optimizer;
  their optimizer weight is structurally zero. Reports must show both values,
  evaluated token/item counts, whole-training-set coverage, and these flags.
  Do not confuse `KL-eval-loss` with the separately named, permitted
  block-local training metric `lens_kl`.
- Lens/objective treatment is depth-uniform. Do not use depth-increasing
  weights, deep-only lens losses, or a readout-shaped auxiliary under another
  name. Report gradient-share attribution with every scientific claim.

## Historical readout-era constraints and evidence (archived)

The following records describe the 2026-07-04/05 readout campaign and are not
current guidance. They remain here so old checkpoints and reports can be
interpreted without silently promoting them to frontier evidence.

- **Never train only the last k blocks "under any subterfuge."** A
  tail-only readout window with CE is, to a referee, classical
  distillation of the top layers — it invalidates the layerwise/forward
  claim. The sanctioned form is the SLIDING k-connected window
  (`conn_window` + `conn_stride: 1`): every block updated with uniform
  k-deep credit; the top window may carry a teacher-sourced readout only
  because logits exist there. **HARD STOP (owner, 2026-07-04 ~22:00): the tailpure ablation
  batch queued tonight is the LAST tail experiment of the project. After
  it completes, NO new arm may use a tail-only window (conn_window
  0/absent + tail_ce_blocks > 0) — not as a baseline, not as a repro
  reference, not "under any subterfuge". New arms use sliding windows
  (conn_window + conn_stride: 1) or fully local schedules. Parallelism
  repro references switch from final_k8 to the slide8 checkpoint.
  Tail/classical-distillation experiments BELONG TO the sibling checkout
  ../selfupdate_kd — route them there, keep this branch pure layerwise.**
  Precise window semantics (gradient-isolation, NOT memory management;
  endpoint vs in-window loss; teacher- vs student-stream input):
  docs/windows.md — read it before touching `window_step` or `conn_window`.
- The embedding and logits matrix are never trained, in any window
  scheme (Frozen-Vocabulary Principle; four locks + runtime tripwire).
- **Historical training-target law (owner, 2026-07-05): it was INCORRECT to train
  logits toward the original text.** Eval against the original text is
  correct (that is what recall means); training toward it is task
  supervision, which belongs to ../selfupdate_kd. On this branch every
  behavioral training term must be TEACHER-SOURCED: readout =
  `readout_source: teacher_kl` pinned explicitly; local per-layer behavioral
  signal = hidden_loss 'lens_kl' (per-layer teacher distributions
  through the frozen head). Reference-text cross-entropy and label-targeting
  lens losses are forbidden in this branch, not valid controls.
  WHY 'gold' was purged as a word: it is a reference-text training term in
  which training target and eval reference are the SAME object.
  Distillation splits the roles — both teacher and student are
  EVALUATED against the original text, but the student is TRAINED from
  the teacher. A vocabulary lacking that distinction re-merges the
  roles silently; the lexicon now types them: reference (eval,
  everyone) / teacher_* (training source) / reference-text training
  (forbidden conflation). Reporting rule: expand every loss
  abbreviation on first use per report — "CE" is written "cross-entropy
  (log loss) against <target>"; two-letter jargon is where a day of
  misclassification hid (2026-07-05).
- **Historical naming contract (owner-refined 2026-07-04):** "auxiliary" = ANY
  signal injected at the logit layer or its weights WITH DEPTH BIAS.
  Lens losses (CE, KL, vocab_mse, whatever) are legitimate LAYERWISE
  losses when applied on ALL layers with similar weight or a justifiable
  alternancy — the frozen vocabulary is a per-layer measurement device.
  They become a disguised output-backprop the moment their weight grows
  toward the output. EXPLICITLY FORBIDDEN DISGUISES: "weighted k-block
  method with bigger weights in the last blocks", "lens_ce only on deep
  blocks", or any depth-increasing weight profile — these are the tail
  wearing a costume; do not let mission-fulfillment pressure reintroduce
  them. Depth-UNIFORM treatment is the invariant; report gradient-share
  attribution (scripts/signal_attribution.py) next to every claim.

## Hard-Won Lessons

- Config DEFAULTS are experiment variables: flipping one mid-campaign
  silently forks every queued/in-flight arm that didn't pin the knob
  (the "PP2 failure" of 2026-07-05 was this, not a parallelism bug).
  Repro configs pin every knob that distinguishes them from their
  reference; flip defaults only between campaigns.

- Corollary (2026-07-17): a knob copied WITHOUT its enabling context is the
  same bug wearing the opposite costume. `cache.source_compaction: remove`
  next to `mask.compaction: flow_mask` looks like a stray inconsistency and is
  not: `build_teacher_cache.py` builds its masker without `keep_privileged`,
  so it ALWAYS writes remove-view student metadata (`s0`/`position_gap`),
  while a flow_mask reader sets `keep_privileged=True` and recomputes a
  different `s0`. `source_compaction: remove` is the reader's explicit
  "this cache is another censorship view; its student metadata is not mine",
  which sets `cross_view` in `DistillDataset` and skips the `student_match`
  assert — the teacher-aligned `t0`/`A` must still match exactly. Remove it
  and training dies with `cache/examples mismatch ...; rebuild the cache`.
  Since pipeline-v3 censorship must be flow_mask/pad_random/intact
  ("removal modes are retired", `train/validate.py`), a v3 flow_mask run on a
  builder-made cache REQUIRES that cross-view, which is licensed only by
  `--coordinated-node-cache`, which in turn requires
  `cache.runtime_policy: node_epoch0` (i.e. `/dev/shm`). The three knobs are
  one decision; do not "simplify" any one of them alone.

- The trainer hot loop is SYNC-bound: `.item()` per block = a GPU
  round-trip per block per item purely for logging (measured 1.46x
  recoverable; see issues.md "sync-bound"). Never add `.item()`/`.cpu()`
  /`print` inside the block walk; accumulate on GPU, flush per accum
  boundary.

- Historical readout-era result: strict hidden matching stored signal but
  weakly recited; the readout was the hard part. This result motivated the
  current deletion and is not a license to reintroduce it.
- Historical C2-26/30 result: hidden matching plus uniform k=8 credit and a
  mimicry-free top window read out clean. Tail windows remain banned, and the
  entire readout mechanism is now archived rather than active.
- Historical `teacher_censored` result: it was useful as a schedule and as a
  localization readout; context integration peaked near layer 7 in Qwen3-0.6B
  artifacts. The localization observation remains historical evidence.
- Pipeline-v3 one-GPU lane/wavefront rearrangements remain dispatch-bound at
  roughly 9-12 token-events/s on Qwen3-0.6B L40S; bounded student lanes added
  memory without speed, and grad-ready hooks slowed the student path. Exact
  measurements and the narrower teacher-lane gain are in `issues.md`. Test
  multi-GPU partitioning or fixed-shape capture/fusion before inventing more
  one-GPU scheduling variants.
- Pipeline-v3 teacher-hidden stale windows are the measured speed bridge, not
  an exact-online synonym. On the 256-token 0.6B probe K=8 was 7.6x faster
  than K=1 but its exact trainable delta diverged by 15.4% globally (layer 1:
  74.4%). Keep K named in configs/reports and compare at a matched logical
  token budget. Mask-free cached attention is valid for q=1 only; K>1 must
  carry causal masking inside the chunk (current code uses one K×prefix mask).
- Pipeline 3.1 names B as simultaneous-user serving parallelism and K as
  within-answer lookahead. B256K1 is next-token online compatible; B256K16
  requires prefetched teacher output or confirmed speculative tokens. The
  `causal_bk_probe` policy is smoke-only and normal training must reject it
  until the B×K memory/speed/LR-scaling screen is promoted.
- Fixed-shape static-cache eager is numerically equivalent to dynamic K=1,
  but the current CUDA-graph replay has a reproducible 1.16% trainable-delta
  divergence despite fixed in-graph gradient buffers. It is a speed prototype
  (~52/s at 0.6B), not a certified campaign path; see `issues.md`.
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
- When an agent captures Python output (ad-hoc probes, log tails), set
  `TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error`
  — otherwise the 0→100% weight-loading progress bars flood the transcript
  and burn tokens.

## L40S Cluster Environment

- `$HOME` is on Lustre. Use `~/...`, not `/home/...`.
- `/usr/bin/python3` is 3.6.8. Use the venv or
  `/opt/ohpc/pub/apps/anaconda/anaconda-2025/bin/python3`.
- Driver 560.35 = CUDA 12.6. The repo container and shared `.venv` now carry
  torch 2.11/cu128 and fail CUDA initialization on this driver
  (`reset_peak_memory_stats: invalid argument`). For L40S campaign work use
  `scripts/l40s_exec.sh`, which reuses the existing torch 2.7.1+cu126
  interpreter and shadows only Transformers 5.12.1, PEFT 0.19.1, and kernels
  0.12.0 from `/tmp/$USER/selfupdate-l40-python`. Build that thin layer with
  `scripts/l40s_setup.sh` through a small delegated agent. Never install torch
  into the layer. The cu128 container remains the H100/new-driver runtime.
  This wrapper is itself the Python launcher: invoke
  `scripts/l40s_exec.sh scripts/train.py ...`, not
  `scripts/l40s_exec.sh python scripts/train.py ...`; the latter is rejected
  with a usage error.
  The base venv's compiled `causal_conv1d` wheel needs glibc >=2.32. The
  wrapper loads `glibc/2.35` and starts Python through `$GLIB235_LINUX_SO`
  with the module, `/lib64`, `/usr/lib64`, and pre-module library paths. Do
  not merely `module load glibc/2.35`: that mixes the old loader with the new
  libc and fails on a `GLIBC_PRIVATE` symbol. The slower torch implementation
  is an explicit diagnostic mode only:
  `SELFUPDATE_L40S_CAUSAL_CONV=torch scripts/l40s_exec.sh ...`.
  The wrapper restores the pre-module `LD_LIBRARY_PATH` before entering
  Python. This is required because child tools such as Triton's `gcc` use the
  host loader and otherwise fail on the same `GLIBC_PRIVATE` mismatch.
- `scripts/l40s_exec.sh` deliberately sets Hugging Face offline mode: model
  snapshots resolve from `/dev/shm/$USER/selfupdate-hf-cache`. Standard-damage
  evaluation must therefore not fetch datasets during training. Its fixed
  100-item inputs are vendored in `data/eval/{arc_easy,arc_challenge,
  hellaswag}_v1.json` with source revisions embedded. If those files must be
  rebuilt, use `scripts/vendor_standard_eval.py` once with
  `HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0`; commit and review the resulting
  data before launching an offline campaign.
- Model-weight staging and teacher-state staging are separate. The former is
  `/dev/shm/$USER/selfupdate-hf-cache`; the latter is
  `/dev/shm/$USER/selfupdate-teacher-cache`. Layerwise training lazily maps
  multi-GB safetensor shards, so leaving teacher states on Lustre can show
  near-zero CPU and GPU utilization while both wait on storage page faults.
  Stage only campaign caches with `scripts/stage_teacher_cache_shm.sh`; its
  ready marker makes `scripts/l40s_exec.sh` export
  `SELFUPDATE_TEACHER_CACHE_ROOT` for subsequent workers. Never copy the whole
  historical root (943 GB measured 2026-07-14); Qwen3.5-4B alone is ~35 GB.
  The copy-based path is historical. Pipeline v4 uses its configured
  `cache.runtime_policy` plus `v4_teacher_source` (`cache`, `online`, or the
  fill-once `store`). Multi-stage launches go through
  `scripts/launch_v4_stages.sh`; `/dev/shm` is node-local, so every host must
  publish or fill its own numerically local teacher material before training.
- No nvcc on PATH by default; CUDA modules exist but pip wheels normally bundle
  runtime libraries.
- Native CPU thread pools are uncapped by default and can oversubscribe the
  64-CPU host. `OMP_NUM_THREADS=64` is not a process-wide 64-thread cap here:
  PyTorch/OpenMP/MKL can still create multiple pools. The train entry point and
  launch scripts default to `SELFUPDATE_CPU_THREADS=8` and clamp it at 22, which
  measured near 64 OS threads after CPU matmul. The trainer already uses
  `num_workers=0`, so extra CPU load comes from runtime pools, not DataLoader
  workers.
- TorchInductor has a separate compiler pool and ignores those native-thread
  caps. Its default is up to 32 workers per trainer; four shape-varying
  Qwen3.5 jobs created roughly 128 compiler workers and left the GPUs mostly
  waiting. `scripts/l40s_exec.sh` therefore defaults
  `TORCHINDUCTOR_COMPILE_THREADS=2` and puts its reusable cache in node-local
  `/tmp/$USER/selfupdate-torchinductor`. A speed probe must cover the real
  sequence-length distribution long enough to expose compilation churn; a
  six-step warm-up is not an epoch-time certificate.
- vLLM has a separate compiler cache root and otherwise writes under
  `~/.cache/vllm` on Lustre. The benchmark and L40S vLLM campaign launcher
  default `VLLM_CACHE_ROOT`, `VLLM_CONFIG_ROOT`,
  `TORCHINDUCTOR_CACHE_DIR`, and `TRITON_CACHE_DIR` to node-local
  `/tmp/$USER/selfupdate-vllm-*`. This disposable compiled-code state is
  separate from model snapshots in `/dev/shm/$USER/selfupdate-hf-cache` and
  teacher states in `/dev/shm/$USER/selfupdate-teacher-cache`. A cold
  Qwen3.5-0.8B launch on 2026-07-15 accidentally used Lustre and spent 92.73
  seconds in `torch.compile`; it was already healthy and was not restarted,
  while all subsequent launches use the node-local defaults.
- Model staging does not warm Python. Before starting many workers on a cold
  node, delegate one `scripts/warm_python_runtime.sh <python> ...` launch per
  runtime. It parallel-stats the Lustre venv/base trees and pre-imports the
  named modules into the node cache without copying the venv. The H100 Slurm
  launcher has the same pattern inline; skipping it left an L40S GPU empty
  during minute-scale vLLM imports on 2026-07-15.
- Scheduler pattern:

```bash
GPUS="0 1 2 3" MAX_PER_GPU=3 nohup setsid bash scripts/gpu_scheduler.sh >> runs/pipeline_sched_main.log 2>&1 &
```

Live-process inspection on L40S (real abbreviated shape from the 2026-07-14
Qwen3.5-4B campaign):

```bash
ps auxww | rg 'gpu_scheduler.sh|scripts/train.py' | rg -v 'rg '
# arivero 965283 ... bash scripts/gpu_scheduler.sh
# arivero 965331 ... bash scripts/gpu_scheduler.sh
# arivero 965337 ... ld-linux-x86-64.so.2 --library-path ... python scripts/train.py ...remove_answer.yaml
# arivero 965382 ... bash scripts/gpu_scheduler.sh
# arivero 965414 ... ld-linux-x86-64.so.2 --library-path ... python scripts/train.py ...remove_token.yaml
```

The first line is the detached scheduler. Each arm then appears as a small
scheduler-owned bash subshell plus a glibc-loader/Python process; seeing the
loader as the process executable is expected. Cross-check physical placement
with `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name
--format=csv,noheader`. Two live `scripts/train.py` PIDs with the same
`--experiment`, especially on different GPUs, are a duplicate launch and both
write the same run directory. Stop the extra process immediately and inspect
the scheduler lock; the lock PID handoff is atomic in current code.

Remote scheduler leases are deliberately not reaped automatically: a process
on another node cannot be declared dead from this node's PID namespace. Before
manually removing a remote lease, identify its recorded host/PID, verify on
that host that both scheduler and worker are absent, then remove only that
lease. Edit a live queue by writing the complete replacement to a sibling
temporary file and atomically renaming it over the queue; never expose a
partially rewritten queue to a scheduler.

## Bootstrap

```bash
scripts/venv_setup.sh          # node-local venv in /tmp, ~30 s
scripts/venv_check.sh          # verify torch/CUDA/pins on THIS node
PY=/tmp/$USER/selfupdate-venv/bin/python
$PY scripts/fetch_poem.py
$PY scripts/build_dataset.py
$PY scripts/build_teacher_cache.py
$PY scripts/audit_configs.py
```

No `pip install -e .`: entry points pin their own tree (Source Of Truth).

Always set this before training:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

Online-teacher LoRA runs (`train.online_teacher: true`) need no teacher cache.

## Multi-Node Conventions

- This cluster does not constrain GPU devices by cgroup. Pass physical GPU ids
  verbatim; do not renumber `CUDA_VISIBLE_DEVICES`.
- Nodes sharing Lustre may read the SAME queue for dynamic load balancing when
  they also share `GPU_LEASE_ROOT`.  The allocator mutex and `done_file` lease
  are global across hosts, while GPU-index capacity is host-scoped (GPU 0 on
  `agpul04` is independent of GPU 0 on `agpul05`).  This is the preferred
  homogeneous L40S campaign pattern; use separate queues only when routing
  different hardware classes or policies.
- Per-node scheduler state: `SCHED=runs/.sched-$(hostname -s)`.
- Per-node scheduler log: `runs/pipeline_sched_<host>.log`; use a per-node
  `JOBLOG_DIR=runs/pareto_v2_4b_worker_logs/<host>` so worker evidence does not
  interleave.
- `evaluate.py --base` needs lane-specific `--out` paths during concurrent
  base evals.
- Results refresher/report shipper should run on only one node.

## Operational Conventions

- **Scientific report completeness (owner, 2026-07-11):** a report is not
  complete if it contains only endpoint recall/damage tables. Before issuing
  or committing `runs/report.pdf`, regenerate and include, for every in-scope
  completed run: (1) per-layer loss by epoch as BOTH a heatmap and a temporal
  line plot (horizontal axis = epoch, vertical axis = loss, one trace per
  layer), plus a cross-run layer summary;
  (2) per-layer parameter modification versus epoch 0/base; (3) recall by
  corpus including epoch 0; (4) standard-benchmark damage and the recall-vs-
  damage frontier; and (5) an explicit coverage/provenance page naming missing
  artifacts, batching regime, loss kind, connected-window width, and evaluation
  source. Never silently omit a run because one artifact is absent: show it as
  missing and queue the calculation. Reports for a named campaign must filter
  to that campaign and must not ingest historical runs merely because an old
  directory was touched or re-evaluated recently. At minimum run
  `scripts/layer_loss_plots.py`, `scripts/delta_profiles.py`, the campaign
  report builder, and the report PDF builder in that order. The agent owns
  verifying that the resulting PDF visibly contains these pages.

- **Evaluation terminology (owner, 2026-07-16):** `epoch zero` is the
  untrained network evaluated under the same prompts, inputs, decoding,
  subsets, and scoring conditions as the student checkpoints. The base
  network evaluated with the original uncensored RAG is a separate teacher
  reference/control; historical records variously call it the teacher
  ceiling, teacher reference, or intact-RAG control. Do not conflate it with
  epoch zero, and do not invent a compound name for the epoch-zero/checkpoint
  standard-benchmark comparison. State the two evaluated conditions directly.

- **Agent-owned supervision (owner, 2026-07-11):** the agent—not a watcher,
  scheduler, or status script—owns a live campaign. While campaign work is
  running, personally review fresh worker/scheduler logs, liveness, checkpoint
  and evaluation completion, and scientific telemetry at least every 30
  minutes. When Codex Scheduled work / a task heartbeat is available, use its
  recurring task-in-thread facility for this cadence; do not emulate agent
  supervision by holding a shell sleep loop open. Monitoring scripts only
  prepare compact evidence for that review.
  Investigate every new error and promptly patch confirmed code or queue
  defects; never treat a healthy-looking GPU or a log line as a substitute for
  active supervision.
  When reading a scheduler-wide log, use `scripts/pipeline_tail.sh` rather
  than raw `tail`: it removes token-heavy loading/progress-bar updates while
  preserving warnings, tracebacks, commands, completion, and failure lines.
  Worker logs remain the source for a detailed progress diagnosis.
- **Logged-launch delegation (owner, 2026-07-14):** any launch expected to
  generate a log must be delegated to a small-class subagent (Luna level, or
  Haiku/Sonnet level). The subagent owns launching the command and reporting
  back only the relevant log lines: the exact command and start marker,
  material warnings/errors or tracebacks, progress/completion evidence, and
  the exit or failure state. Do not stream routine progress bars, model-loading
  chatter, or large raw log tails into the parent transcript. This delegation
  keeps log handling compact; it does not transfer the parent agent's campaign
  supervision, diagnosis, or scientific-review responsibility.
- **RAG target-generation gate (owner, 2026-07-12):** a failed RAG gate is a
  diagnosis obligation, never a threshold-relaxation or queue-bypass event.
  The agent must inspect completion cuts and exact tokenized tool conversation,
  repair the retrieval invitation/placement when the teacher ignores RAG,
  rebuild the affected dataset/cache under a new identity, and rerun the
  real-RAG vs no-RAG epoch-zero and random-context controls. See
  `docs/rag_generation_gate.md`.
- Never abort a training run before it has seen at least 12,000 training items.
  Early noisy plateaus have recovered in the layerwise tail runs, and matched
  item budget is needed for comparisons.
- Configs are `configs/base.yaml` plus small YAMLs in `configs/experiments/`.
- Run outputs land in `runs/<run_name>/`.
- Long work runs detached via `nohup setsid ... >> runs/pipeline*.log 2>&1 &`.
- After changes touching masking, aligned spans, cache layer-index conventions,
  or detach discipline, run `scripts/audit_configs.py`, compare single-process
  and PPP artifacts with `scripts/compare_v4_shard_numerics.py`, and run
  `scripts/v4_battery.py`; stored fingerprints remain intentionally absent.

## Hardware Ladder

- 0.6B/4B: v4 mechanics, locality, and single-vs-PPP numerics.
- 27B/35B: teacher-store and resident/rotary scaling.
- 122B/397B/MoE: stage-scoped PPP, rotating weights, and Don Quijote.

## Current Pointer

Current campaign guidance is pipeline v4 and `docs/training_pipeline_v4.md`.
The final synthesis may group atomic reports by campaign, model, loss,
censorship, teacher source/residency, and PPP ownership; it must exclude
archived v1-v3/readout diagnostics from frontier claims.

The following pointer is historical campaign context, not a current method
recommendation:

Campaign 2 closed 2026-07-05 16:00 (EXPERIMENTS.md: CLOSING TABLE, ten
laws, ledger corrections; paper/paper1.pdf; docs/casebook.md at
signal-anatomy standard). Crown: slide8pure 0.007/99.3%/CLEAN
(intrusion 2.5% at n=200), 84.4% trajectory-driven. The last-3% law
(C2-34) has ~5 replications. NOTE the two silent-default confounds of
2026-07-05 — tail_ce_kind is now an UNSET sentinel and every windowed
config must choose explicitly.
INHERITANCE VERDICTS READ 2026-07-10 (EXPERIMENTS.md "Inheritance
verdicts" section): seed claim REPLICATED (s43 0.0076/0.991/1.5%);
C2-35 RESOLVED (disjoint 0.023/7% clean — the collapse was the
confound); PP2 blocker CLOSED (pp2fix 0.011 + certs/pp2); xs 1.7B
spectrum recalls but is DIRTY (22-40% intrusion — cleanliness at 1.7B
is an open C3 question); lw_r_crown17_pinned never ran and needs an
OWNER DECISION (task_label readout no longer exists on this branch).
Conclusion ledger: runs/conclusions.yaml (validate with
scripts/conclusion_check.py); cross-model matrix: scripts/model_matrix.py.
C3 queue: (1) teacher-stream k-windows;
(2) premise-gated thinking teacher_kl; (3) Qwen3.6-27B bridge grid +
Gemma4-E4B (embed-scaling adapter); (4) wide-channel ragchannel;
(5) DONE 2026-07-10 — trainer refactor (docs/runtime.md);
(6) reincarnation; (7) intrusion prompts already 200; (8) DONE
2026-07-10 — evaluate.py --layer-residuals; (9) NEW: 1.7B cleanliness
question from the xs spectrum.
