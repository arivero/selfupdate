# Agent Guide — selfupdate_kd

Operational guide for this branch. Read `README.md` for the science framing
and `EXPERIMENTS.md` for the live status board.

## Branch Scope

This branch is **classical KL-based distillation only**. The research question
is where classical KD modifies the model: layer delta norms, adapter norms,
logit-lens readability, and graft/ablate causal effects.

## Source Of Truth

Everything needed should live in the repo, not host-local memory.

- `EXPERIMENTS.md` — current plan, active axes, and run list.
- `runs/results.md`, `runs/report.pdf`, `runs/curves.png` — generated outputs.
- `docs/scaling.md` — large-model KD plan.
- `runs/*/metrics.jsonl`, `runs/*/eval/*`, `runs/*/checkpoint` — raw run
  artifacts; `runs/` is gitignored and must be copied separately if moving.

## Hard-Won Lessons

- KL saturation is not recitation. Pure top-k KL reached low loss while
  free-run text remained broken; gold answer CE is the critical recitation
  signal.
- LoRA KD needs lr around `1e-4`; `1e-5` produced misleading plateaus.
- Evaluate on the full corpus. The 8-example training subset is front-biased
  and can hide severe coverage failure.
- Full fine-tune KD freezes embeddings, final norm, and lm_head so layer
  localization is about transformer blocks.
- Two concurrent GPU jobs need a VRAM guard with random stagger.
- `pkill -f pattern` can kill its own shell if the pattern appears in the
  command line; use bracketed patterns and keep the target filename out of the
  command text.
- VRAM checks at launch underestimate peak because AdamW state lands at the
  first optimizer step. Leave headroom.
- Do not abuse filesystem search on Lustre. Search only inside this repo, and
  prefer `git ls-files`/known paths plus `rg`.

## Agustina Cluster Notes

- `$HOME` is on Lustre. Use `~/...`, not `/home/...`.
- No usable system Python: `/usr/bin/python3` is 3.6.8.
- Working interpreter:
  `/opt/ohpc/pub/apps/anaconda/anaconda-2025/bin/python3` (3.13.5).
- Driver 560.35 = CUDA 12.6. Install torch with cu128 wheels:
  `--index-url https://download.pytorch.org/whl/cu128`.
- No nvcc on PATH. Pip wheels normally bundle the needed CUDA runtime.
- Scheduler example:
  `GPUS="0 1 2 3" MAX_PER_GPU=3 nohup setsid bash scripts/gpu_scheduler.sh >> runs/pipeline_sched_main.log 2>&1 &`

## Bootstrap

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e . && .venv/bin/pip install pytest
.venv/bin/python scripts/fetch_poem.py
.venv/bin/python scripts/build_dataset.py
.venv/bin/python scripts/build_teacher_cache.py
.venv/bin/python -m pytest tests/ -q
```

Deps: torch >= 2.10, transformers >= 5.3. Always export
`PYTORCH_ALLOC_CONF=expandable_segments:True` before training.

## Multi-Node Conventions

- This cluster does not constrain GPU devices by cgroup. Treat
  `CUDA_VISIBLE_DEVICES` entries as physical GPU ids and pass them verbatim:
  `GPUS="$(tr ',' ' ' <<<"$CUDA_VISIBLE_DEVICES")"`.
- Use one queue file per node/lane. Two schedulers must never share a queue.
- Use per-node scheduler state, e.g. `SCHED=runs/.sched-$(hostname -s)`.
- Use per-node job logs, e.g. `JOBLOG=runs/pipeline_sched_<host>.log`.
- `evaluate.py --base` needs a distinct `--out` per lane.
- Run `results_refresher` and `report_shipper` from one node only.

## Operational Rules

- Never abort a training run before it has seen at least 12,000 training items
  unless it is clearly broken.
- Configs live in `configs/base.yaml` plus one YAML per run under
  `configs/experiments/`.
- Outputs land under `runs/<run_name>/`.
- Long work runs detached:
  `nohup setsid bash ... >> runs/pipeline*.log 2>&1 &`.
- Rebuild reporting with `scripts/analyze.py` and `scripts/report.py` at wave
  boundaries.
- Keep `scripts/queue.tsv`, `scripts/queue_h100.tsv`, and
  `scripts/watchdog_backlog.tsv` KD-only.

## Current State Pointers

Best early full-FT recitation: `runs/kd_ce_0p6b_rag`, KD + `answer_ce_weight:
0.5`, 20 epochs, full-corpus CER about 0.596 in prior artifacts.

Active next steps:

- Compare KD recipes by per-layer delta profile, not only recitation score.
- Run logit lens and graft/ablate on successful KD checkpoints.
- Continue the model-size ladder only after premise gates pass.
