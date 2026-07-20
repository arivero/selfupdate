# Native pipeline-parallel evaluation

`train.v4_battery_mode: distributed` evaluates the live stage-owned student.
It never reconstructs a complete model, materializes a foreign block, or sends
token traffic through `/dev/shm`. Every rank enters a dedicated NCCL group at
one epoch boundary, after the ordinary relay has been drained. Rank 0
tokenizes and embeds, each owner executes its contiguous blocks, and the final
rank alone applies final norm and the frozen vocabulary head.

This mode is opt-in until disposable-checkpoint multi-GPU parity is complete.
Unsupported configurations fall back explicitly to the reconstructed-model
subprocess battery and write `distributed_battery_fallback`.

## Evaluation taxonomy

The telemetry names four distinct experiments.

| Test / row | Student path | Reference | Cache/KV provenance | Autoregressive |
|---|---|---|---|---|
| a: `vllm_teacher_forced_reproduction_eval` | Epoch-zero, uncensored full PP forward over the stored vLLM input and answer | vLLM answer token IDs | Frozen adapters-disabled zero-run teacher state, constructed as one cached prefill | No |
| b: native `student_trajectory_eval` | Current adapters enabled; full censored student trajectory through every PP block | Frozen adapters-disabled, uncensored zero-run teacher trajectory | Student retains only current student state for its owned layers; teacher reference retains only zero-run state for its owned layers | No |
| c: `eval` recall battery | Current adapters enabled; prompt omits privileged/RAG context | Reference answer text | Fresh prefill once, then per-owned-layer cache retained across incremental greedy tokens | Yes |
| a′: `vllm_uncensored_autoregressive_control` | Current adapters enabled; full privileged/RAG prompt | vLLM answer text and token sequence | Same live per-owned-layer cached decode as c | Yes |

Test a is recorded only at epoch zero. At zero-initialized LoRA, whether the
adapter switch is enabled is numerically irrelevant; the implementation makes
the provenance explicit by disabling adapters before constructing all teacher
hidden and K/V state. Test b recomputes that frozen reference at the boundary
and compares the adapters-enabled censored student with it.

`teacher_output_eval` is neither a nor b. It is a streaming, block-local
final-block diagnostic: the trained final block consumes zero-run teacher
`h[n-1]` and frozen teacher K/V. It reports answer-token and exact-answer
vLLM agreement, cross-entropy against the vLLM token IDs, and teacher-to-
student KL, but it is not a complete student trajectory.

The legacy asynchronous staged `student_trajectory_eval` is a complete
censored student forward over fixed sequences, but stages may service the
requested epoch after training farther. Its rows now state
`may_combine_different_adapter_epochs=true`, `adapter_epoch_exact=false`, and
the final stage's service epoch. Only the native epoch-boundary row claims a
synchronized adapter epoch.

Standard evaluation remains teacher-forced normalized continuation log
likelihood. CE/KL/acceptance rows are also teacher-forced. Only c and a′ are
autoregressive rollouts.

## Audit of the pre-native paths

- The reconstructed subprocess recall row is genuine autoregressive inference
  from a complete reconstructed student. Each stage publishes its trainable
  tensors under an epoch and launch envelope before the subprocess grafts
  them. The old reader checked launch and epoch but did not require the
  envelope producer to equal the requested stage, and the child did not prove
  that each stage's complete expected adapter key set arrived. Both checks are
  now mandatory. With them, every reported token either uses every stage's
  requested-epoch adapter or the battery fails; there is no partial graft.
- Graft/subprocess epoch snapshots cannot mix named epochs because paths and
  envelopes carry the requested epoch and launch identity. They can only have
  been silently incomplete before the new key-set check. The live native path
  additionally exchanges epoch, launch hash, ownership, and adapter
  fingerprints before inference.
- `teacher_output_eval` uses uncensored zero-run teacher `h[n-1]` after the
  first block. It is intentionally a block-local surrogate. Native and relay
  `student_trajectory_eval` start from the student embedding and never inject
  teacher hidden after it; teacher states are scoring targets only.
- CE-eval-loss, KL-eval-loss, both argmax-acceptance rates, exact-sequence
  rates, and standard option NLL are teacher-forced. None is produced from
  generated answers and none enters backward or an optimizer.
- Epoch-zero and post-epoch recall/standard calls use the same task builders,
  item order, prompts, censorship state, token budgets, padding, position-ID
  rules, and scoring functions. Epoch zero is not the uncensored RAG teacher
  control. a′ is named separately for that condition.
- The asynchronous relay can combine stage weights serviced at different
  training frontiers. Its requested epoch labels the sequence payload, not an
  exact multi-stage weight snapshot. Native evaluation flushes that traffic,
  barriers all ranks, and therefore cannot race asynchronous training.
- Runtime freezes embedding, final norm, and LM head on every rank. Native
  entry compares their fingerprints across ranks and evaluation exit compares
  them with their pre-evaluation values.

## Why student K/V is rebuilt at the epoch boundary

A current student trajectory must construct K/V from current adapters-enabled
student states. Reusing the training teacher cache would create a hybrid
surrogate and must not be called student inference. The native b path creates
a fresh cache for each validation cohort and proves that every foreign cache
slot is empty.

The certified schedule is the epoch boundary. This is the natural coherent
frontier at which all blocks and stages have completed the same epoch.
Per-cohort or per-batch recomputation is mechanically possible for
`item_major`: all ranks would need a barrier after every N cohorts (N=1 for
every batch) and a fresh student prefill. It is not certified or enabled.
There is no coherent mid-epoch frontier in `layer_major`, because blocks have
processed different corpus prefixes. Carrying a student cache between
training cohorts is invalid because examples are independent sequences.

## Supported architectures and fallback

| Configuration | Native standard | Native cached generation | Result |
|---|---:|---:|---|
| Qwen3 dense full attention | yes | yes | supported |
| Qwen3.5 dense text with full + linear attention | yes | yes | supported at protocol/unit level; fleet checkpoint parity still required |
| Qwen3.5 MoE text | no | no | subprocess fallback until separately certified |
| Gemma4 text with sliding + full attention, `num_kv_shared_layers=0`, `hidden_size_per_layer_input=0` | yes | yes | supported at protocol/unit level; fleet checkpoint parity still required |
| Gemma shared K/V (`num_kv_shared_layers>0`) | no | no | loud subprocess fallback; shared-KV side channel is not transported |
| Gemma per-layer input embedding | no | no | loud subprocess fallback |
| DeepSeek compressed/mHC boundaries | no | no | subprocess fallback |
| Rotary/nonresident block paging | no | no | subprocess fallback |
| Non-LoRA training | no | no | subprocess fallback; a/b need an adapters-disabled zero-run reference |

Current locally resolved Gemma 26B and 31B text configurations report
`num_kv_shared_layers=0` and `hidden_size_per_layer_input=0`. Runtime support
still checks the loaded config instead of relying on model names.

## Protocol and invariants

Before evaluation, ranks barrier and exchange launch hash, requested epoch,
and exact contiguous ownership bounds. Trainable surfaces are fingerprinted
per stage; embedding, final norm, and LM head fingerprints must match across
ranks. Relevant modules enter eval under `torch.inference_mode()`. Exact
heterogeneous train/eval flags are restored afterward, and adapter plus frozen
vocabulary fingerprints must be unchanged.

Every forward broadcasts the boundary after each owner. This deliberately
gives every rank the same collective sequence. Local compute/tokenizer/head
errors are reduced before the next payload collective. An execution counter
must show every layer exactly once. Dynamic-cache tensor state must be present
for every owned layer and absent for every foreign layer. Stage 0 alone writes
durable telemetry.

## Timing rows

`distributed_battery` records fixed-sequence validation, standard scoring,
recall generation, optional uncensored vLLM generation, and total boundary
seconds. Offload, model load, and adapter graft are exactly zero.

The fallback records parent offload/restore/total boundary time and child
model-load, adapter-attach, adapter-graft, recall, standard, and evaluation
time. Existing historical rows lack some components; before/after claims need
fresh disposable runs under both backends on the same checkpoint.

## Certification

CPU-safe protocol coverage:

```bash
TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
TRANSFORMERS_VERBOSITY=error \
/tmp/$USER/selfupdate-venv/bin/python scripts/check_distributed_eval_cpu.py
```

It compares complete-model versus stage-cut logits, normalized option NLL,
prefill logits, and five cached greedy tokens for B=1 and a variable-left-
padding batch. It uses independent stage caches for Qwen3, hybrid Qwen3.5,
and Gemma4; checks parameter/mode preservation; and proves a rank-local Gloo
failure does not strand its peer.

This is not final fleet parity. Adoption still requires disposable artifacts
for epoch zero and nonzero LoRA, B=1 and configured batching/EOS budgets,
standard and recall telemetry row parity, Qwen and Gemma checkpoints, no
foreign CUDA context, and injected-rank failure. Do not run that certification
on live campaign processes or GPUs.
