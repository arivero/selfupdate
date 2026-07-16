# Trainer numerical fingerprint regression instrument (no stored references)

`scripts/train_certify.py` runs the real `train_layerwise` on a matrix of
tiny variants (every schedule / batching / window / optimizer path) and
fingerprints each run: per-step losses, a per-tensor checkpoint signature
(fp64 sum, abs-sum, 64-point sample), and peak allocated/reserved VRAM.

**No references are stored in this repo** (owner decision 2026-07-11:
stored fingerprints — like stored tests — act as a frozen specification
that agents ossify around; the historical `pre/` and `pp2/` reference sets
live in git history). The tool is an on-demand numerical-regression instrument for changes
INTENDED to be numerics-preserving:

```bash
python scripts/train_certify.py --all --out-dir /tmp/$USER/certify_head
# ... apply the trainer change ...
python scripts/train_certify.py --all --reference-dir /tmp/$USER/certify_head
```

Record on HEAD, apply, compare, discard. The recording always encodes what
HEAD does now — never doctrine about what the trainer must forever do.

Certification is separate from throughput benchmarking by design:
it answers "is this the same experiment?", `speed_check.py` /
`parallel_bench.py` answer "how fast?". Comparisons key on a SEMANTIC config
hash that excludes placement-only knobs (`model.device`, `model.device_map`,
`model.pipeline_split(s)`, `run_name`), so one single-device recording also
certifies a pipeline-parallel run of the same experiment
(`--override model.pipeline_split=14`).

- `examples_subset16.jsonl` — 16-example slice used by the variants whose
  schedules lack `max_steps` support (teacher_censored / mixed / sequential),
  so the budget stays small without touching trainer code.

## Tolerance calibration (measured 2026-07-10, L40S agpul04, torch 2.11.0+cu128)

Defaults: loss rtol 5e-3, sampled-weight rtol 5e-2 — set above the measured
same-code run-to-run drift. Bit-exact equality is NOT expected: SDPA
backward and batched GEMMs are nondeterministic at bf16. Evidence (full
record in git history: `certs/pre/SELF_DRIFT.md`):

- Four full 13-variant sweeps passed against fixed references without one
  tolerance violation, across process restarts and concurrent GPU load.
- Drift sources pinned: a B=1 collated batch is BIT-EXACT vs the historical
  item path (same kernel shapes); B>1 drifts only through bf16
  kernel-shape rounding, compounding with depth to ~3e-2 max-relative at
  h21/28; the streamed optimizer offload is bitwise-equal to the resident
  step. (The measuring tests were deleted with `tests/` in fd7138d; the
  measurements stand in git history.)
- PP2 (`pipeline_split=14`) matched single-device fingerprints inside the
  same bounds — placement does not move losses or final weights.

If a comparison fails marginally, first rerun the variant twice on one GPU:
only a violation exceeding the same-code spread is a finding.
