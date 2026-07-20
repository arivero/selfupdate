# Pipeline v4.5 migration

Pipeline v4.5 moves live student evaluation into the trainer. The scientific
training law is unchanged: every optimizer update remains block-local against
the frozen zero-run teacher. The revision change names the new synchronized
evaluation and provenance protocol.

## Removed user-facing scripts

### `scripts/v4_battery.py`

Do not launch an epoch battery separately. For supported resident Qwen3,
Qwen3.5-dense, and Gemma4 configurations, select the live evaluator:

```yaml
train:
  pipeline_version: 4
  pipeline_revision: "4.5"
  v4_battery_mode: distributed
  v4_relay_every_cohorts: 1  # enables synchronized b at battery boundaries
```

The same trainer invocation now runs epoch-zero and post-epoch recall,
standard scoring, a, b, c, and optional a′:

```bash
scripts/launch_v4_stages.sh configs/base.yaml configs/experiments/arm.yaml
```

The direct Python form remains valid for a one-stage/non-PPP diagnostic whose
configuration uses the resident `graft` evaluator (or the reconstructed
`subprocess` fallback):

```bash
PY=/tmp/$USER/selfupdate-venv/bin/python
$PY scripts/train.py --config configs/base.yaml \
  --experiment configs/experiments/arm.yaml
```

On L40S the equivalent command uses the repository shell wrapper, which is
itself the Python launcher:

```bash
scripts/l40s_exec.sh scripts/train.py --config configs/base.yaml \
  --experiment configs/experiments/arm.yaml
```

Unsupported/shared-KV/rotary configurations may keep
`v4_battery_mode: subprocess`. The trainer automatically self-invokes a private
reconstructed-model worker after coordinating publication and GPU release.
There is no supported manual worker command.

### `scripts/verify_vllm_teacher_forced.py`

Use test a emitted by a v4.5 distributed epoch-zero battery. It forwards the
complete uncensored prompt plus vLLM answer through the live PP owners and
records both token acceptance and exact-answer acceptance in
`vllm_teacher_forced_reproduction_eval`.

For the synchronized censored counterpart, read `student_trajectory_eval`.
For free generation controls, enable a′:

```yaml
eval:
  vllm_uncensored_generation_limit: 8
  vllm_uncensored_max_extra_tokens: 48
```

That writes `vllm_uncensored_autoregressive_control`. It is distinct from the
teacher-forced a row and from censored recall c.

## Scripts intentionally retained

- `scripts/merge_v4_adapters.py`: still required to assemble a portable full
  LoRA checkpoint from disjoint stage checkpoints. Native evaluation does not
  materialize foreign blocks and therefore cannot replace publication merge.
- `scripts/evaluate.py`: full checkpoint recitation/CER evaluation, broader
  than the trainer's fixed epoch recall battery.
- `scripts/standard_destruction_eval.py`: standalone historical-checkpoint and
  WikiText-perplexity evaluation; the trainer covers only the vendored
  multiple-choice battery.
- `scripts/teacher_ceiling.py`: independent programmatic-vLLM RAG controls,
  including wrong/random context conditions.
- `scripts/vllm_prefill_verify.py`: verifies the vLLM engine against its own
  earlier greedy output; the trainer cannot substitute for a vLLM-side check.
- `scripts/check_distributed_eval_cpu.py`, `scripts/compare_v4_shard_numerics.py`,
  and other `check_*`/comparison tools: independent certification must not be
  replaced by the implementation it certifies.

Historical experiment overlays pinned to `pipeline_revision: "4.0"` remain
loadable for reproduction. New configurations inherit 4.5 from
`configs/base.yaml`. `v4_battery_mode: distributed` requires revision 4.5 so a
run cannot silently claim the old protocol identity while using live PP
evaluation.
