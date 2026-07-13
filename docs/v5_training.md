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

## Question-suite source coverage

The active window-RAG suite
`data/combined/examples_v5rs_window.jsonl` contains 2,071 questions: 1,490
for Machado (716 source lines) and 581 for Quijote (280 source lines).  A
2026-07-13 audit counted each record's half-open `target_lines` span against
the source named by `examples_v5rs_window_coverage.json`.  “Direct” below
means only `next` and `prev` questions.  Cloze records are reported separately
because their metadata identifies the target block, not the exact deleted
words; treating every word in that block as a cloze answer would overclaim.

| corpus | source lines | normalized word types | direct hits per line (min / median / mean / max) | hits including cloze block | uncovered direct lines | uncovered direct word types |
|---|---:|---:|---:|---:|---:|---:|
| Machado | 716 | 1,245 | 1 / 3 / 3.327 / 4 | 2 / 4 / 4.327 / 5 | 0 | 0 |
| Quijote | 280 | 2,295 | 1 / 3 / 3.311 / 4 | 2 / 4 / 4.311 / 5 | 0 | 0 |

Thus every supplied source line, every normalized word occurrence, and every
normalized vocabulary type appears in at least one direct answer target; the
suite's full-text coverage does not depend on the cloze questions.  The
least-repeated targets are boundary cases: the first and last Machado lines,
the first Quijote line, and the last few long Quijote lines.  The generated
coverage manifest's coarser invariant (`covered_lines == n_lines`) agrees with
this audit for both corpora.

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

## Single-card cache construction, 2026-07-13

The teacher cache has two independent phases.  A graph-capable continuous
generation backend produces exact response token IDs with one per-record
allowance and stop ID; `build_teacher_cache.py --generation-responses` then
uses those IDs directly for the teacher-forced hidden-state pass.  There is no
decode/re-encode round trip.  The response file content hash is part of cache
identity.  `--generation-only --generation-responses` validates and scores a
response artifact without loading teacher weights; the full 2,071-row Gemma
artifact imported in 0.029 s and its complete CPU audit took 1.66 s.

This division does not introduce a student dependency.  Cache construction
loads only the frozen teacher, and its forward input is the teacher prompt plus
the teacher's own generated answer.  No student premise/contrast forward is
part of this phase.

The dense Qwen3.5-4B single-card measurement used batch 64 for both generation
and requested hidden forwards.  Generation took 35.22 s for 87,306 tokens
(2,478.57 token/s).  The original B=1 hidden walk took 623.64 s; length-aligned
randomized hidden batches with persistent OOM backoff reduced it to 86.20 s
(effective batches 5–64, no OOM).  The 36.63 GB cache copied to CPU in 0.924 s
at 36.92 GiB/s.  Thus D2H was only 1.34% of teacher compute; after batching,
storage/backpressure, not PCIe, is the secondary bottleneck.

The batching refactor preserves `teacher_batch: 1` bit-exactly.  Against the
completed B=1 reference, a 64-example all-layer certification was bit-exact
for all 2,048 tensors, and a full-cache audit confirmed identical semantic
spans for all 2,071 examples plus bit-exact tensors for 32 evenly spaced
examples.  Cache schema 9 includes `teacher_batch`, because future model
kernels are not assumed to share this exactness result.

The same requested B=64 path completed 64-example one-card capacity probes for
both large MoEs without OOM: Gemma-4-26B-A4B-it took 11.29 s total for 30-layer
cache payloads (effective B 1–42), and Qwen3.6-35B-A3B took 12.23 s for 40
layers (effective B 1–44).  The maxima reflect sparse length buckets in the
small evenly spaced sample, not memory backoff.  This verifies the batched
decoder-body path for dense Qwen, MoE Qwen, and Gemma's wrapped language model.

Operationally, generation progress is logged per engine batch and hidden-pass
progress is appended to `teacher_progress.jsonl` every 100 examples.  The
hidden ledger records only wall/queue counters: it adds no `.item()`, CPU tensor
copy, CUDA-event wait, or stream synchronization to the teacher walk.
