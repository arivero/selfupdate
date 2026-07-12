# The v5 Training

Companion source for `docs/v5_training.pdf`, generated with:

```bash
.venv/bin/python scripts/build_v5_training_pdf.py
```

This is an implementation note for the active v5 RAG layerwise runs.  It
documents the frozen-teacher cache, the `train.py` execution path, exact cache
storage arithmetic, and the full-vocabulary loss peak.  Every implementation
claim in the PDF is accompanied by a gray source excerpt and file/line
reference.  The numbers are regenerated from the active v5 cache indexes and
model configs, not inferred from a failure log.

The document intentionally distinguishes:

- persistent cache storage on disk;
- resident model/optimizer memory on the GPU; and
- temporary vocabulary-loss tensors created during a local backward step.

The last category is the relevant one for the current 1.7B OOM investigation.

## Recovered comparison snapshot

The PDF also compares the current implementation with a detached temporary
worktree at `/tmp/selfupdate_lw_pre_v5`, commit `3fb5305` (the parent of the
commit that taught the cache builder to generate v5 answers).  The comparison
corrects an important historical shorthand: that snapshot already has a
hidden-state `TeacherCacheWriter`; what it lacks is the current v5
teacher-generation / answer-id cache payload needed for question-only records.

## Critique and online alternative

The PDF also criticizes v5's extra teacher-generation/cache pass and sketches
an on-the-fly autoregressive alternative. The latter is a design proposal, not
an implemented mode: an optimizer update after every generated token makes
student KV caches stale, so an exact design must replay the student prefix.

## Future KV and teacher-vector directions

The article distinguishes three directions: censored teacher-KV reuse and
stale student-KV reuse are not implemented; `teacher_censored` is implemented
as a distinct teacher-hidden-vector schedule. It is not KV reuse: the frozen
teacher first computes full-context hidden states, then the student block sees
only the non-privileged teacher rows at its input.

## Review correction, 2026-07-12

The initial v5 summed full-finetuning path had a serious execution error.
`frozen_teacher_copy: true` was needed only to precompute anchor targets, but
its mere presence also selected the online-teacher path. Every student batch
therefore retained a second 1.7B teacher in VRAM and recomputed all teacher
hidden targets instead of using the cache. New launches release that frozen
copy immediately after anchor materialization and select online targets only
when `train.online_teacher` is explicitly true. Results produced by processes
started before this correction are legacy results, not measurements of the
intended cached method.

The cache identity now includes `cache.generation_extra_tokens`; changing the
answer budget cannot silently reuse a cache with different generated answer
ids. Future queues also run `scripts/cache_generation_gate.py` after the cache
build, checking the hard-cut rate in the actual `generation_report.json` before
an arm can start. This complements, rather than replaces, the RAG retrieval
gate.
