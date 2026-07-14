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
tree instead — `scripts/*.py` via `sys.path.insert(0, <repo>/src)`
(`tests/` and its conftest were deleted 2026-07-11, see Training Runtime
section). Keep that guard in any new script; never
`pip install -e .` into the shared venv (it would silently route imports
across checkouts). The bootstrap's `pip install -e .` applies only to a
fresh per-tree venv.

## Container Runtime

Preferred runtime for Lustre-heavy jobs is the repo-local Singularity setup,
not a copied venv:

```bash
scripts/container_exec.sh python scripts/train.py ...
```

Artifacts:

- `containers/pytorch-2.11.0-cu128-cudnn9-runtime.sif` - official
  `docker://pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` converted to SIF;
  contains Python 3.12.3, torch 2.11.0+cu128, CUDA runtime 12.8, cuDNN 9.19.
- `containers/selfupdate-python-deps-cu128.sqsh` - read-only squashfs overlay
  with Python add-ons (`transformers==5.12.1`, `accelerate==1.14.0`,
  `peft==0.19.1`, pandas/matplotlib/pytest/etc.). It deliberately does NOT
  contain another torch/CUDA stack.
- `scripts/container_exec.sh` - binds this checkout as `/work`, sets
  `PYTHONPATH=/dev-python:/opt/selfupdate-python:/work/src`, uses `--nv`, and
  keeps Singularity cache/tmp/container-home under `/tmp/$USER` instead of
  `/home`.

Confirmed on H100 node `agpuh01` with driver 565.57.01 / NVIDIA H100 80GB
HBM3: torch 2.11.0+cu128 reports CUDA 12.8, `torch.cuda.is_available() == True`,
and a bf16 CUDA matmul succeeds. Check any new node with:

```bash
nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv
scripts/container_exec.sh python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print((x @ x).float().mean().item())
PY
```

Do not let Singularity write caches to `/home/arivero`: Singularity 3.7 may use
the passwd home (`/home/arivero`) even when shell `$HOME` points at Lustre.
Always set `SINGULARITY_CACHEDIR`, `SINGULARITY_TMPDIR`, and `TMPDIR` outside
`/home`; the launcher already does this. `/tmp` is node-local XFS on tested
nodes and is appropriate for transient conversion/cache, while durable SIF/SQSH
artifacts live in this repo on Lustre.

For development installs, use the writable dev layer mounted at `/dev-python`.
It defaults to host path `/tmp/$USER/selfupdate-dev-python` and shadows the
read-only overlay:

```bash
scripts/container_pip.sh install --no-deps <package>
scripts/container_exec.sh python -c "import <package>"
```

Set `SELFUPDATE_DEV_PYTHON_HOST=/some/path` if the dev layer must persist
across node-local `/tmp` cleanup. Avoid putting this on `/home`; if it lives on
Lustre, remember it is loose Python files again and may have venv-like metadata
cost. For stable campaign runs, fold proven dev packages back into
`containers/selfupdate-python-deps-cu128.sqsh` with `mksquashfs`.

Do not copy a venv, and do not install a second torch by accident. Use
`--no-deps` or explicit constraints so the dev/overlay layers keep using the
base image's torch 2.11.0+cu128.

## Training Runtime & Certification (2026-07-10 refactor)

Execution concerns live in `src/selfupdate/train/runtime.py`
(TrainingRuntime: loading/placement/teacher/tripwire/save; OptimizerPlan:
`lora_fused` / `full_resident` / `full_offload` with streamed pinned-CPU
paging). Schedule loops in `layerwise.py` never construct models or
optimizers. Since the 2026-07-11 factorisation the trainer package is one
module per concern — schedules (`layerwise.py`), step primitives
(`steps.py`), knob validation (`validate.py`), telemetry (`telemetry.py`),
teacher states (`teacher_source.py`), anchor (`anchor.py`) — module map in
docs/runtime.md; `layerwise.py` re-exports the historical names.
One batched walk: `batching: item` is a B=1 padded batch,
bit-exact vs the historical item loop. Read `docs/runtime.md` before
touching execution machinery, and note the measured NEGATIVE results in
issues.md (async target prefetch; PP2 throughput) before "optimizing".

There is NO stored test or certification gate (owner decision 2026-07-11:
tests and stored fingerprints act as specifications that agents ossify
around; both were deleted — `tests/` in fd7138d, `certs/pre`+`certs/pp2`
in this pass; git history keeps them). The specification is the prose laws
in this file plus the RUNTIME enforcement in the code: `_validate_knob_schedule`
raises at dispatch, the frozen-vocab fingerprint tripwire at save, the
graph-leak/MoE tripwires in the walk, `scripts/audit_configs.py`.

For a trainer change that is INTENDED to be numerics-preserving, use
`scripts/train_certify.py` as an on-demand A/B instrument — record fresh
fingerprints on current HEAD, apply the change, compare, discard:

```bash
python scripts/train_certify.py --all --out-dir /tmp/$USER/certify_head
# ... apply the change ...
python scripts/train_certify.py --all --reference-dir /tmp/$USER/certify_head
```

References are always minted from HEAD, never stored in the repo — there
is no frozen numerics doctrine, only a per-diff "did this change anything?"
measurement (~minutes; 13 tiny variants; semantic config hash excludes
placement knobs, so `--override model.pipeline_split=14` compares PP
against the same single-device fingerprints; calibration in
`certs/README.md`). `scripts/memory_plan.py` (meta-device + one measured
block) recommends micro-batch/window/optimizer-placement/splits BEFORE
loading weights — advisory only.

## Branch Focus

This branch is for layerwise forward distillation only. Active training methods
are in `src/selfupdate/train/layerwise.py`:

- `summed`
- `sequential`
- `teacher_censored`
- `mixed`
- bounded sliding connected windows (`conn_window` + `conn_stride: 1`)
- teacher-sourced readout (`readout_source: teacher_kl`) only when attached
  to the sanctioned sliding window

Do not reintroduce non-layerwise training configs, queues, docs, or dispatch.

## Publication-Critical Constraints (owner directives, 2026-07-04)

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
- **Training-target law (owner, 2026-07-05): it is INCORRECT to train
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
- **Naming contract (owner-refined 2026-07-04):** "auxiliary" = ANY
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

- The trainer hot loop is SYNC-bound: `.item()` per block = a GPU
  round-trip per block per item purely for logging (measured 1.46x
  recoverable; see issues.md "sync-bound"). Never add `.item()`/`.cpu()`
  /`print` inside the block walk; accumulate on GPU, flush per accum
  boundary.

- Strict hidden matching stores signal but weakly recites; the readout is the
  hard part.
- Sliding uniform windows are the lever (C2-26/30): hidden matching stores,
  uniform k=8 credit + a mimicry-free top window reads out clean. (The C1-era
  "tail-CE is the best lever" reading is superseded — tail windows are banned
  on this branch.)
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
- No nvcc on PATH by default; CUDA modules exist but pip wheels normally bundle
  runtime libraries.
- Native CPU thread pools are uncapped by default and can oversubscribe the
  64-CPU host. `OMP_NUM_THREADS=64` is not a process-wide 64-thread cap here:
  PyTorch/OpenMP/MKL can still create multiple pools. The train entry point and
  launch scripts default to `SELFUPDATE_CPU_THREADS=8` and clamp it at 22, which
  measured near 64 OS threads after CPU matmul. The trainer already uses
  `num_workers=0`, so extra CPU load comes from runtime pools, not DataLoader
  workers.
- Scheduler pattern:

```bash
GPUS="0 1 2 3" MAX_PER_GPU=3 nohup setsid bash scripts/gpu_scheduler.sh >> runs/pipeline_sched_main.log 2>&1 &
```

## Bootstrap

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e .
.venv/bin/python scripts/fetch_poem.py
.venv/bin/python scripts/build_dataset.py
.venv/bin/python scripts/build_teacher_cache.py
.venv/bin/python scripts/audit_configs.py
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
- Re-run tests after changes touching masking, aligned spans, cache layer-index
  conventions, or detach discipline in `train/layerwise.py`.

## Hardware Ladder

- 0.6B: mechanics, locality tests, strict-vs-tail ablations.
- 1.7B/4B/8B: readout-window scaling and memory curve.
- 14B/32B: online-teacher LoRA, sharding where needed.
- MoE/120B-class: streamed blocks, post-combine hidden matching, Don Quijote.

## Current Pointer

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
