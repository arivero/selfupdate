# Agent guide — selfupdate

Orientation for a fresh agent/session, especially after moving the repo to a
new machine. Read README.md for the science; this file is operational.

## Source of truth

Everything an agent needs lives IN THIS REPO — do not depend on host-local
state (`~/.claude/plans`, agent memory dirs): those do not travel with clones.

- `EXPERIMENTS.md` — the experiment plan + live status board + headline
  results. Update it at every wave boundary.
- `runs/results.md`, `runs/report.pdf`, `runs/curves.png` — auto-generated
  metrics/report (rebuild with `scripts/analyze.py`, `scripts/report.py`).
- `docs/hidden_loss.md`, `docs/scaling.md`, `docs/memory.md` — loss math and
  locality proofs; big-model plan; memory-vs-params accounting.
- `runs/*/metrics.jsonl` + `runs/pipeline_*.log` — raw training dynamics and
  pipeline history. Checkpoints are in `runs/*/checkpoint` (gitignored — copy
  separately if they must move).

## Hard-won lessons (do not relearn these on GPU time)

- **KL saturation ≠ recitation**: pure distillation reaches KL ~0.03 while
  free-run recitation stays broken; a gold-CE auxiliary is what makes
  recitation click. Exposure bias, not optimization failure.
- **LoRA needs lr ~1e-4**, not the full-FT 1e-5 (rank-16 KL plateaued at 2.2
  with the wrong lr and poisoned a whole round of conclusions).
- **Eval on the full corpus**: the 8-example training-eval subset covers the
  poem's opening and masked severe front-of-poem bias (subset CER 0.002 vs
  full-corpus 0.596 on the same checkpoint).
- **Layerwise trains hidden states, not behavior**: no block-local variant
  recites yet; last-block-CE hybrid failed at lr 1e-5. Current bets: proper
  lr, per-block lens-CE, or layerwise-as-preconditioner + short KD polish.
- **teacher_censored (variant b) dominates student-stream (a)** on both
  memorization and forgetting; its per-layer increment profile doubles as a
  localization readout (context integration peaks at layer ~7 in Qwen3-0.6B).
- Two concurrent GPU jobs need a VRAM guard WITH a random stagger — two
  processes checking free memory in the same second both pass and collide.
- `pkill -f pattern` kills your own shell if the pattern appears in your own
  command line; use `[.]`-style patterns — and never reference the target
  filename elsewhere in the same command.
- VRAM checks at launch time underestimate peak (AdamW state lands at the
  first step): full-FT jobs need launch-requirement ≈ peak + 1.5 GB.
- Greedy small-job packing starves big-VRAM queue items; give the scheduler
  a drain/priority mode before running mixed grids on the L40S.
- **Do not abuse filesystem search**: home/repo live on Lustre — recursive
  `find`/`grep -r` over big trees hammers the metadata servers for everyone.
  Search only inside the repo (it is small), never sweep `/fs/...`, and prefer
  `git ls-files`/known paths over crawling.

## L40S cluster environment (Agustina, 2026-07)

- `$HOME` is on Lustre, not under /home. **Always write paths as `~/...`,
  never `/home/...`** (and avoid hardcoding the absolute Lustre prefix).
  `~/.cache/huggingface` (model cache), `~/.local/bin` (`hf` CLI) and this
  repo are all on Lustre — big sequential reads are fine; metadata-heavy
  crawls are what must be avoided.
- OpenHPC + Lmod. **No usable system python**: `/usr/bin/python3` is 3.6.8.
  The conda modules only APPEND to PATH so `module load` does not change
  `python3` — use absolute paths. Working interpreter for the venv:
  `/opt/ohpc/pub/apps/anaconda/anaconda-2025/bin/python3` (3.13.5).
  `python-math/3.11.4` worked historically but pyproject now needs ≥3.12.
- Driver 560.35 = CUDA 12.6: default pip torch (≥2.12) ships cu130 wheels
  that FAIL on this driver. Install with
  `--index-url https://download.pytorch.org/whl/cu128`.
- No nvcc on PATH; `cudatoolkit/12.9` + `cudnn/9.10` modules exist if a build
  ever needs them (pip wheels bundle their own CUDA runtime — normally not).
- No vllm module; LLM-adjacent modules are llama.cpp/b4706,
  llama-cpp-python/0.3.1, ollama/0.15.1 (irrelevant to training; pip-install
  vllm into the venv if trace harvesting needs it).
- Scheduler invocation here: `GPUS="0 1 2 3" MAX_PER_GPU=3
  nohup setsid bash scripts/gpu_scheduler.sh >> runs/pipeline_sched_main.log
  2>&1 &` plus `scripts/results_refresher.sh` alongside.

## What this repo is (one paragraph)

Self-distillation of context: the same model is teacher (sees a RAG passage or
its own <think> trace) and student (context hidden, must behave identically).
Regimes: logit KD vs block-local layerwise hidden matching (schedules:
summed / sequential / teacher_censored), each × {full-FT, LoRA}, with gold-CE
auxiliaries and an "online teacher" (LoRA adapters off = frozen teacher, no
cache). First corpus: *La tierra de Alvargonzález* (Machado). Endgame
("Pierre Menard"): 120B-class models memorizing Don Quijote.

## Bootstrap on a new machine

```bash
python3 -m venv --system-site-packages .venv   # reuse system torch if recent
.venv/bin/pip install -e . && .venv/bin/pip install pytest
.venv/bin/python scripts/fetch_poem.py         # data/poem/raw.txt
.venv/bin/python scripts/build_dataset.py      # examples.jsonl (RAG mode)
.venv/bin/python scripts/build_teacher_cache.py  # per-model cache + premise check
.venv/bin/python -m pytest tests/ -q           # MUST be green before training
```

- Deps: torch ≥ 2.10, transformers ≥ 5.3 (v5 API: `dtype=`, Cache objects).
- Always `export PYTORCH_ALLOC_CONF=expandable_segments:True` for training.
- Online-teacher runs (`train.online_teacher: true`, LoRA only) need **no**
  teacher cache — preferred on new machines and for big models.
- The premise check printed by build_teacher_cache must show a large gap
  (teacher CE with context ≪ without); if not, the model already knows the
  corpus — pick another corpus/model.

## Operational conventions

- Configs: `configs/base.yaml` + one small YAML per run in
  `configs/experiments/`; run outputs land in `runs/<run_name>/`
  (config.yaml, metrics.jsonl, checkpoint/, eval/).
- Long work runs DETACHED so SSH/session death cannot kill it:
  `nohup setsid bash scripts/overnight*.sh >> runs/pipeline*.log 2>&1 &`
  Pipelines are idempotent (done-file guards) — rerun to resume.
  Two-lane pattern (train lane + VRAM-guarded eval lane) overlaps GPU use.
- Reporting: `scripts/analyze.py` (results.md, curves.png, delta profiles,
  convergence), `scripts/report.py` (runs/report.pdf, verbose).
- Never change without re-verifying tests: masking segment conventions,
  aligned-span definition, cache layer-index convention (h{L} =
  output_hidden_states[L]; last is post-final-norm), detach discipline in
  train/layerwise.py.
- Established results and env quirks live in the agent memory and in
  runs/results.md; docs/hidden_loss.md and docs/scaling.md explain the loss
  and the big-model plan.

## Hardware ladder

### Tier 0 — 1× RTX 3060 12 GB (origin)
0.6B full-FT KD (9.4 GB, embed/head frozen); 0.6B–1.7B layerwise + LoRA
(3–8 GB). Everything in runs/ up to 2026-07 came from here.

### Tier 1 — 2× RTX 4090 (2×24 GB, PCIe)
- The experiment grid is embarrassingly parallel: run one experiment per GPU
  (`CUDA_VISIBLE_DEVICES=0/1` with two detached pipelines) — the biggest win.
- Full-FT KD at 1.7B: fp32 AdamW needs ~27 GB → use bitsandbytes 8-bit Adam
  or bf16 weights + fp32 master offload; layerwise sequential fits easily.
- 4B: LoRA + online teacher comfortably on one card; full-FT layerwise
  sequential also fits (one block ~0.4 GB + optimizer).
- teacher_censored can split blocks across both GPUs (layers are independent).
- Batched eval (task noted in repo) matters once two cards multiply runs.

### Tier 2 — 4× L40S (4×48 GB)
- Qwen3-14B/32B with LoRA + online teacher (bf16 base 28/64 GB — 32B needs
  2-GPU sharding via accelerate/FSDP2).
- First MoE work: Qwen3-30B-A3B — post-combine hidden matching only, log
  teacher/student routing agreement, per-expert delta norms (docs/scaling.md).
- Full-FT KD via FSDP2 to ~8B; beyond that KD stays LoRA-only.
- Move teacher-cache builds to layer-streamed forwards if caching (or stay
  online-teacher).

### Tier 3 — 4× H100 (4×80 GB, NVLink)
- Pierre Menard stage: corpus switches to Don Quijote (chapter-chunked tasks
  through the same masking abstraction; expect ~500k answer tokens).
- 120B-class dense / DeepSeek-GLM-class MoE:
  - layerwise sequential = one block (2–4 GB bf16) + its optimizer per GPU;
    pipeline stages across cards (advance / train / prefetch).
  - teacher_censored = 4 blocks training concurrently, zero communication.
  - KD = LoRA-only, FSDP2-sharded bf16 base, online teacher mandatory
    (a hidden-state cache would be ~450 GB).
- vLLM/sglang only for teacher trace harvesting and KD prompt_logprobs —
  no engine exposes per-layer hidden states (docs/scaling.md).

## Current state pointers (2026-07-03)

Best recitation: runs/kd_ce_0p6b_rag (KD + answer_ce 0.5, 20 ep, CER 0.596
full / 0.875 line-exact on eval subset). Pure hidden matching does not recite
yet; local last-block CE (`last_block_ce_weight`) is the live hybrid lever.
Convergence finding: methods share per-layer magnitude profiles (Spearman
~0.75) with orthogonal delta directions (cos ~0.02). Next planned: hybrid
verdicts, compaction axis (remove/stub/stub_gap), thinking-mode arm, 1.7B
replication, then Tier-1 parallel grid.
