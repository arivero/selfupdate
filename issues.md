# Code Review Issues

Scope: static read-only review of the repo code surface: `src/`, `scripts/`,
`tests/`, experiment configs, and shell orchestration files. No tests or
training commands were completed after the stop instruction; this is file
inspection only. This file is the only change.

## Findings

### P2 - Multi-GPU scheduler does not actually reserve multi-GPU jobs exclusively

`scripts/gpu_scheduler.sh` documents `n_gpus > 1` jobs as reserving that many
devices exclusively (`scripts/gpu_scheduler.sh:10-12`). The multi-GPU candidate
selection does require every selected device to have zero scheduler jobs
(`scripts/gpu_scheduler.sh:63-74`), but after launching a multi-GPU job the
outer loop continues to later devices and only checks whether
`running_on "$dev"` is below `MAX_PER_GPU` (`scripts/gpu_scheduler.sh:52-54`).
With the default `MAX_PER_GPU=3`, a device already occupied by a multi-GPU job
still has count 1, so a single-GPU job can be launched onto the same card in
the same or next cycle.

Impact: the scheduler can collide supposedly exclusive tensor/FSDP jobs with
other jobs, exactly the kind of VRAM race the operational docs warn about.
Multi-GPU locks need to mark their devices as exclusive for all scheduling
decisions, not just for selecting other multi-GPU jobs.

### P2 - Teacher-cache identity omits payload-shaping parameters

`src/selfupdate/teacher/cache.py` hashes model name, mask mode, compaction, and
examples SHA (`src/selfupdate/teacher/cache.py:33-50`). It does not include
`cfg.cache.topk`, hidden target dtype, tokenizer/model revision, or a full cache
schema. The cache writer stores top-k logits using `cfg.cache.topk`
(`scripts/build_teacher_cache.py:74-77`), while disk-teacher KD simply consumes
whatever tensor width is present (`src/selfupdate/train/kd.py:101-104`).
Online-teacher KD, however, uses the current config value
(`src/selfupdate/train/kd.py:91-100`).

Impact: changing `cache.topk` or rebuilding expectations can silently reuse an
old cache at the same directory. That makes disk-teacher and online-teacher KD
incomparable, or lets a run claim one top-k setting while training on another.
The index should record and validate payload shape, at minimum `topk`, hidden
dtype/schema version, tokenizer identity, and model revision.

### P2 - Sequential layerwise ignores `last_block_ce_weight`

`src/selfupdate/train/layerwise.py` implements `last_block_step(...)` for the
hidden+gold-CE hybrid (`src/selfupdate/train/layerwise.py:61-74`), and both
`_train_summed(...)` and `_train_teacher_censored(...)` use it for the final
block (`src/selfupdate/train/layerwise.py:234-240`,
`src/selfupdate/train/layerwise.py:294-300`). `_train_sequential(...)` always
calls `local_block_step(...)`, including `L == n`
(`src/selfupdate/train/layerwise.py:401-405`), so
`train.last_block_ce_weight` is a silent no-op under the sequential schedule.

Impact: a sequential hybrid experiment can be configured and logged as if the
final-block CE lever is active, but the model only receives hidden matching.
That invalidates comparisons against summed or teacher-censored hybrid runs.

### P2 - Logit-lens analysis cannot load LoRA checkpoints

`scripts/logit_lens.py` loads both base and trained sources with
`AutoModelForCausalLM.from_pretrained(...)`
(`scripts/logit_lens.py:36-44`). LoRA runs save adapter-only checkpoints and
are handled specially in `scripts/evaluate.py`
(`scripts/evaluate.py:37-45`), but `logit_lens.py` has no equivalent
`PeftModel.from_pretrained(...)` path.

Impact: LoRA and online-teacher runs are first-class in the experiment grid, but
the depth-localization tool fails on them unless adapters are manually merged.
The script should detect adapter checkpoints and load them the same way
evaluation does.

### P3 - Rerunning a failed run appends incompatible metrics to the same log

`setup_run_dir(...)` overwrites `config.yaml` but opens `metrics.jsonl` in append
mode (`src/selfupdate/utils/runlog.py:17`, `src/selfupdate/utils/runlog.py:31-36`).
`scripts/analyze.py` then reads all historical lines for that run and derives
losses, steps, evals, time, and VRAM from the combined stream
(`scripts/analyze.py:33-66`).

Impact: if a training run fails partway and is rerun with the same `run_name`
or a changed config, analysis mixes old and new metrics under the latest
config. That can corrupt `runs/results.md`, curves, and report appendices. A
new run should either start with a fresh metrics file, rotate the old one, or
record a run attempt id and have analysis select one attempt.

### P3 - Base evaluation output path does not match analysis input

`scripts/evaluate.py --base` writes to `runs/base-eval/recite.json`
(`scripts/evaluate.py:56-58`). `scripts/analyze.py` looks for the baseline
forgetting probe at `runs/base-eval-full/recite.json`
(`scripts/analyze.py:22-27`).

Impact: after following the documented base-eval command, `scripts/analyze.py`
leaves `forgetting_dCE` empty, or uses a stale manually copied
`base-eval-full` result if one exists. The two scripts should agree on one
baseline path or accept an explicit baseline argument.

### P3 - Generated report contains hardcoded conclusions

`scripts/report.py` derives some numbers from artifacts, but `summary_text()`
still hardcodes corpus/task counts, method conclusions, convergence values,
memory figures, and the named checkpoint `kd_ce_0p6b_rag`
(`scripts/report.py:59-102`). Other report pages are also tied to that same
checkpoint (`scripts/report.py:113-127`, `scripts/report.py:181-186`).

Impact: `runs/report.pdf` is presented as an auto-generated report, but after
new experiments it can publish stale conclusions even when `runs/results.md`
says something different. The report should compute these claims from current
artifacts or label them as a fixed narrative snapshot with a source date.

## Test Gap

The test suite was not completed because code execution was stopped. The review
above is from file inspection, not from a fresh green test run. The existing
tests cover several important invariants, but the findings above are mostly in
orchestration, reporting, rerun hygiene, and schedule-specific branches that are
not directly covered by the current tests.

## Resolutions (2026-07-03, all fixed in commit "address issues.md")

1. **Scheduler multi-GPU exclusivity** — FIXED: `exclusive_on()` marks every
   device of a live multi-GPU lock as unschedulable for all decisions, not
   just multi-GPU selection. Applies on next scheduler restart (single-GPU
   tonight, so no live exposure).
2. **Cache identity** — FIXED: hash now covers topk, hidden dtype, and a
   schema version (schema=2). Existing v1 cache migrated to its new identity
   (dir renamed, index config_hash rewritten, reopen-validated).
3. **Sequential ignores last_block_ce_weight** — FIXED: `_train_sequential`
   uses `last_block_step` for the final block, same as the other schedules.
4. **Logit lens on LoRA checkpoints** — FIXED: adapter checkpoints are
   detected and merged via PeftModel before profiling.
5. **Metrics append on rerun** — FIXED: `setup_run_dir` rotates a non-empty
   metrics.jsonl to metrics.prev-<timestamp>.jsonl; analysis reads only the
   current attempt.
6. **Base-eval path mismatch** — FIXED: `evaluate.py --base` writes to
   `runs/base-eval-full/` (the path analysis reads).
7. **Report hardcoded conclusions** — MITIGATED: headline recitation/
   forgetting numbers computed from current artifacts (best run auto-
   selected); graft/ablate and logit-lens pages iterate all runs; the prose
   findings are explicitly labeled a dated narrative snapshot. Full auto-
   generation of narrative text is intentionally out of scope.

Test-gap note: the suite could not run green during review because the GPU
was fully packed by the scheduler (all failures were CUDA OOM at fixture
load); a full-suite run is queued behind a 9 GB free-VRAM requirement
(`runs/.tests_green`).

## Tokenization is Qwen-family-only by construction (audited 2026-07-03; GENERALIZED same day — see src/selfupdate/chatfmt.py)

Verified on all five ladder models (0.6B/1.7B/4B/8B/14B): identical chat
template (manual rendering in masking.py == apply_chat_template, student and
teacher views), same special tokens (<|im_end|> = 151645 = eos), same vocab,
no BOS. So the current grid is sound. But three Qwen-isms will break silently
on other families (Llama/SmolLM/Gemma are on the ladder):
- masking.py hardcodes <|im_start|>/<|im_end|> and the empty-<think> block;
  Llama-style templates also prepend a BOS that segment-encoding would drop.
- eval/recite.py line ~70 hardcodes convert_tokens_to_ids("<|im_end|>").
- teacher/generate.py assumes </think> exists.
FIXED via src/selfupdate/chatfmt.py: segment pieces are derived from any
tokenizer's own chat template (sentinel splitting), records re-render at load
time from their raw question/answer_text fields (exact identity on Qwen —
tests/test_chatfmt.py), and generation stops use stop_token_id(). Thinking
mode and rag_tool stay Qwen-only and raise explicit errors elsewhere.
Remaining for a new family: run build_teacher_cache premise check and eyeball
one adapted record; Gemma additionally needs sliding-window mask support.
Qualitative forgetting probe: scripts/sanity_chat.py (trivial questions incl.
a Quijote control we never train on; queue writes eval/sanity.json per run).
