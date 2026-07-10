# Same-code drift evidence (tolerance calibration)

Tolerances (loss rtol 5e-3, sampled-weight rtol 5e-2) must sit above
same-code run-to-run drift. Evidence collected 2026-07-10 on L40S
(`agpul04`), torch 2.11.0+cu128:

- Four full 13-variant sweeps of the refactored trainer passed against
  these references without a single tolerance violation (post-Task-3,
  post-Task-4, post-Task-7/8, final), spanning process restarts and
  concurrent GPU load — the observed same-code drift is comfortably inside
  the bounds.
- Controls that pin down the drift sources: a B=1 collated batch is
  BIT-EXACT against the historical item path on all layers (same kernel
  shapes); B>1 batching drifts only through bf16 kernel-shape rounding,
  compounding with depth to ~3e-2 max-relative at h21 of 28
  (tests/test_online_teacher.py documents the measurement). The streamed
  optimizer offload is bitwise-equal to the resident step
  (tests/test_offload_adam.py).
- PP2 (pipeline_split=14) certified against these SINGLE-DEVICE references
  in certs/pp2/ — placement does not move losses or final weights outside
  the same bounds.

If a future comparison fails marginally, first rerun the variant twice on
one GPU: only a violation that exceeds the same-code spread is a finding.
