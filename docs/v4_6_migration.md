# Pipeline v4.6 migration

Pipeline v4.6 makes live-owner evaluation mandatory and deletes the complete-
model reconstruction path. The block-local training law is unchanged.

## Configuration

Remove `train.v4_battery_mode`; it no longer exists. Set revision 4.6 and use
the ordinary trainer/launcher:

```yaml
train:
  pipeline_version: 4
  pipeline_revision: "4.6"
  v4_relay_every_cohorts: 1
eval:
  every_epochs: 1
  vllm_uncensored_generation_limit: 8  # optional a′
```

```bash
PY=/tmp/$USER/selfupdate-venv/bin/python
$PY scripts/train.py --config configs/base.yaml --experiment EXP.yaml

# Any PPP1/PPP2/PPPn, resident or rotary; host placement is data, not a
# model-specific wrapper:
SELFUPDATE_V4_STAGE_HOSTS="$(hostname -s) agpuh02" \
  scripts/launch_v4_stages.sh configs/base.yaml EXP.yaml
```

For a single-node PPPn launch omit `SELFUPDATE_V4_STAGE_HOSTS`. Physical GPU
IDs and stage cuts remain in the YAML. The same launcher covers the deleted
Qwen, Gemma, DeepSeek, Adam, eval-in, and campaign-specific launch wrappers.

The trainer emits a, b, c, a′, standard scoring, parameter deltas, and the
typed teacher-output diagnostic at the configured boundaries. Do not invoke a
separate epoch-battery process.

## Removed legacy scripts

The following categories were deleted because they encoded one dated queue,
model, host map, optimizer choice, or report refresh rather than a reusable
operation:

- `launch_{dsflash,g31b,q0p6b,q122b,q397b}*`,
  `chain_ppp8_when_ready.sh`, `overnight_27b_online.sh`, `run_m1_legs.sh`,
  `wait_m1_then_g26b_e500.sh`, and `stop_g31b_at25.sh`;
- campaign-specific standard-eval/report refresh shell wrappers;
- the one-off coverage-queue builder;
- the v3 delta comparator;
- the old cross-node reconstructed-battery publication check;
- the private reconstructed evaluator and its hidden trainer-worker CLI.

Use YAML overlays plus `launch_v4_stages.sh` for launches, the trainer's
durable epoch telemetry for recall/standard rows, and
`compare_v4_shard_numerics.py` plus `check_distributed_eval_cpu.py` for current
certification. Shell sequencing belongs in the scheduler/queue configuration,
not a model-named script checked into the source tree.

Historical docs may still name deleted wrappers as provenance for old runs;
that is not current execution guidance.

## Utilities intentionally retained

- `merge_v4_adapters.py` produces an ordinary PEFT adapter from disjoint stage
  checkpoints. It is deployment/publication, not evaluation.
- `fuse_experts_snapshot.py` produces an ordinary Hugging Face sharded
  checkpoint (`model-*.safetensors`, `model.safetensors.index.json`, copied
  `config.json` and tokenizer assets). It is retained for vLLM-compatible
  deployment; it does not invent a selfupdate checkpoint format.
- `evaluate.py` and `standard_destruction_eval.py` evaluate arbitrary
  historical/foreign checkpoints and include surfaces outside an active
  trainer boundary.
- `teacher_ceiling.py`, `vllm_prefill_verify.py`, and the vLLM benchmark
  drivers test the external teacher engine, which the trainer cannot certify
  from inside itself.
- `check_*`, `compare_v4_shard_numerics.py`, report builders, cache builders,
  staging tools, and the generic launcher remain independent utilities.

## Adoption gate

The CPU protocol check is necessary but not fleet certification. Before
adopting v4.6 for a campaign, use disposable copies—not live jobs—to compare
epoch-zero and nonzero-LoRA Qwen/Gemma artifacts, configured batching and EOS,
rotary PPP1/PPP2 large models, standard/recall telemetry, exact mode and byte
restoration, GPU ownership, and injected-rank failure. Run the v4 numerics
comparison and config audit for this masking/cache change.
