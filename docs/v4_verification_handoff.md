# v4 verification handoff — runs to execute and what to accept

Written 2026-07-17. The pipeline-v4 code is complete and committed (see
`docs/training_pipeline_v4.md` for the protocol). M1 (single-process) and M2
(4-process shard) are ALREADY verified on `agpuh01` — do not re-litigate:

- M1: full epoch in 6.8 s, 34,413 token-events/s, 56 writes (28 blocks × 2
  cohorts), locality passed with cross-block and frozen-vocab gradients
  exactly 0.0, teacher-forced CE 0.136 / KL 0.0001 over 8,334/8,334 tokens,
  recall + standard damage + parameter deltas per epoch.
- M2: per-layer epoch losses BIT-IDENTICAL to M1 (worst relative delta
  0.000e+00, `scripts/compare_v4_shard_numerics.py`); merged adapter vs
  single-process max |Δ| 3.0e-11 (CUDA atomic jitter, not systematic);
  locality exact zeros on all four stages.

## Environment (fresh session bootstrap)

```bash
cd /fs/agustina/arivero/supercomplex/selfup_teacher
scripts/venv_setup.sh && scripts/venv_check.sh       # ~30 s if not present
PY=/tmp/$USER/selfupdate-venv/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1 \
       HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error \
       SELFUPDATE_CPU_THREADS=8
```

The v4 smoke cache (identity `16b7ed657df99d67`, 3.1 GB) lives at
`/dev/shm/$USER/selfupdate-teacher-cache-v4-smoke` on `agpuh01`. If the node
was cleaned, rebuild (~10 min):

```bash
$PY scripts/build_teacher_cache.py \
  --config configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml \
  --experiment configs/experiments/h100_smoke/qwen3_0p6b_v4_1proc.yaml \
  --coordinated-node-cache
```

## Pending verification legs (in order; A and B may run concurrently on
different cards, C needs all four)

**A — adam + aligned positions + single-process relay eval (cuda:0):**
```bash
$PY scripts/train.py \
  --config configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml \
  --experiment configs/experiments/h100_smoke/qwen3_0p6b_v4_adam_aligned_e2.yaml
```
Accept when: exit 0; TWO `v4_epoch` rows; `student_trajectory_eval` rows
present (one per epoch) with `trajectory=student_censored_flow_full_walk`;
`teacher_output_eval` per epoch; `locality_certification.passed=true`;
checkpoint published. Loss positions per item are larger (aligned span), so
`token_events` per epoch exceeds leg-B's.

**B — student-refresh KV (cuda:1):**
```bash
$PY scripts/train.py \
  --config configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml \
  --experiment configs/experiments/h100_smoke/qwen3_0p6b_v4_refresh_e2.yaml
```
Accept when: exit 0; two epochs; epoch 2's per-layer losses differ from a
teacher_frozen run's epoch 2 (the KV moved with the adapters — compare
against leg-A only qualitatively, different optimizer); locality passed.
Epoch 2 should be SLOWER than epoch 1's steady state (KV rebuilt once).

**C — staged relay + merged battery (all four cards):**
```bash
scripts/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml \
  configs/experiments/h100_smoke/qwen3_0p6b_v4_4stage_e2.yaml
```
Accept when: all four stages exit 0; `runs/<run>/relay/e0001/` and `e0002/`
contain `stage0..2.st` boundary files and `adapters_stage0..3.st`;
stage3's metrics carry `student_trajectory_eval` with
`trajectory=student_censored_flow_staged_relay` per epoch; stage0's metrics
carry the per-epoch recall (`eval`) and `standard_eval` rows AFTER epoch 1
(the merged battery) plus epoch zero; every stage's locality passed;
`scripts/merge_v4_adapters.py runs/h100_smoke_qwen3_0p6b_v4_4stage_e2`
succeeds.

Watch every leg with compact greps (kinds via
`rg -o '"kind": "[a-z_0-9]+"' <run>/metrics.jsonl | sort | uniq -c`), never
raw tails. Verify artifacts, not exit codes (an `echo EXIT=$?` after a
command reports the echo's own status — this bit us once already).

## Known rough edges to watch

- The staged relay waits up to 3600 s per boundary; a dead stage surfaces as
  a relay timeout naming the missing file — inspect that stage's log first.
- Stage 0's battery grafts foreign-block adapters onto its model per epoch;
  this is by design harmless to training (v4 never reads foreign blocks).
- `student_trajectory_eval` (single-process form) has NOT run yet: the resident leg (m1a) is
  its first exercise. If it fails, the fault is likely in
  `_relay_segment`'s flow_keep walk (run_block builds the causal mask from
  `flow_keep.shape[1]` when q_len == kv_len — full-sequence calls are the
  intended shape).
- After the legs pass, fold the results into `docs/training_pipeline_v4.md`
  (a Measured Results section), update `EXPERIMENTS.md`, and consider the
  12k-item floor before any scientific claim — these are mechanics runs.
