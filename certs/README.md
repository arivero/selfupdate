# Trainer certification artifacts

`scripts/train_certify.py` runs the real `train_layerwise` on a matrix of
tiny variants (every schedule / batching / window / optimizer path) and
fingerprints each run: per-step losses, a per-tensor checkpoint signature
(fp64 sum, abs-sum, 64-point sample), and peak allocated/reserved VRAM.

Certification is separate from throughput benchmarking by design:
certification answers "is this the same experiment?", `speed_check.py` /
`parallel_bench.py` answer "how fast?". Comparisons key on a SEMANTIC config
hash that excludes placement-only knobs (`model.device`, `model.device_map`,
`model.pipeline_split(s)`, `run_name`), so one single-device reference also
certifies a pipeline-parallel run of the same experiment.

- `pre/` — reference artifacts captured at the pre-refactor revision (see
  `git_rev` inside each JSON). Any trainer refactor must re-run
  `--all --reference-dir certs/pre` and pass before merging.
- `pp2/` — the same experiments run under `--override
  model.pipeline_split=14` on 2xL40S, certified against the SINGLE-DEVICE
  references above (the matched PP artifacts issues.md required). Their
  `done.vram_per_device_gb` fields carry the per-card peaks.
- `examples_subset16.jsonl` — 16-example slice used by the variants whose
  schedules lack `max_steps` support (teacher_censored / mixed / sequential),
  so the certification budget stays small without touching trainer code.

Tolerances: loss rtol 5e-3, sampled-weight rtol 5e-2 by default — set above
the measured same-code self-drift (two identical runs on one L40S: see
`pre/SELF_DRIFT.md`). Bit-exact equality is NOT expected: SDPA backward and
batched GEMMs are nondeterministic at bf16.
