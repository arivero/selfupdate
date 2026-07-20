# Pipeline v4.6 native pipeline-parallel evaluation

Pipeline v4.6 has one staged evaluation path: the live stage owners evaluate
the current student together. There is no reconstructed-model child, adapter
publication/graft protocol, CPU evaluator, or architecture fallback.

Every rank enters a dedicated NCCL evaluation group at the same epoch. The
trainer first drains the asynchronous fixed-sequence relay, barriers, verifies
launch/epoch/ownership, and then executes an identical collective sequence.
Rank 0 alone tokenizes and embeds. Each owner executes its contiguous blocks
exactly once. The final rank alone applies final norm and the frozen vocabulary
head; rank 0 alone writes durable telemetry.

## Evaluation taxonomy

| Row | Student computation | Reference | Autoregressive |
|---|---|---|---:|
| `teacher_output_eval` | One trained final block on detached zero-run teacher `h[n-1]` and frozen teacher context | vLLM answer IDs / zero-run teacher output | no |
| a: `vllm_teacher_forced_reproduction_eval` | Epoch-zero uncensored full PP trajectory over vLLM input+answer | vLLM answer token IDs | no |
| b: `student_trajectory_eval` | Current adapters-enabled censored full student trajectory | adapters-disabled uncensored zero-run teacher trajectory | no |
| c: recall `eval` rows | Current adapters-enabled censored/no-RAG prompt rollout | reference answer text | yes |
| a′: `vllm_uncensored_autoregressive_control` | Current adapters-enabled full-RAG prompt rollout | vLLM answer text/token sequence | yes |

CE-eval-loss, KL-eval-loss, argmax acceptance, exact-answer acceptance,
standard option scores, a, and b are teacher-forced. They do not become
generation metrics merely because every student block participates. Only c
and a′ are autoregressive.

At epoch zero, zero-initialized LoRA is numerically the base model whether the
adapter switch is enabled or disabled. The implementation disables adapters
for the reference anyway, making provenance explicit. At later boundaries b
compares the current censored student with that adapters-disabled zero-run
trajectory. The frozen teacher K/V used by local training is likewise typed
separately from the current student cache.

The asynchronous staged b relay remains useful between batteries: it is a
complete fixed-sequence student trajectory, but its telemetry explicitly says
that stages may have serviced the requested row at different training
frontiers. The synchronous boundary b row is epoch-exact. Neither uses teacher
hidden as a student input after the embedding.

## Architecture state carried natively

There is no model-name allowlist. The block adapter reflects the loaded
Transformers contract:

- dense/full and sliding attention use the owner-local dynamic cache;
- Qwen3.5 linear/recurrent layers retain recurrent state only on their owner;
- rotary residency pages only the active owner block, records evaluation H2D
  cost, then restores a quiescent rotator and its counters;
- mHC boundaries preserve the `[B,T,hc_mult,H]` tail and the final owner uses
  the frozen hyper-head before final norm;
- Gemma per-layer token inputs are computed by the frozen input stack and the
  relevant owner slice is transported;
- Gemma shared-KV producers retain their ordinary prefix cache. Their
  full-length `(K,V)` mapping is a transient NCCL side channel at every stage
  cut and token step. Shared consumers never pretend hidden-only transport is
  sufficient;
- mixed full/sliding caches derive physical key length from the actual cache
  class or transported producer tensor, not from a family label;
- DeepSeek compressed/mHC block inputs and token-id routing remain the loaded
  block adapter's responsibility; no foreign block is materialized.

The same shared-KV envelope is used by the asynchronous b relay and teacher
store-fill. Frozen local training of a shared consumer uses `_FrozenSharedKV`,
which supplies the adapters-disabled producer state through the explicit
`shared_kv_states` argument while preserving query-side-only gradients.

Current Gemma 26B/31B snapshots report `num_kv_shared_layers=0` and
`hidden_size_per_layer_input=0`, but v4.6 tests nonzero synthetic instances so
support does not depend on those current values.

## Cached generation

Recall and a′ begin with one left-padded prefill. Each rank retains cache state
only for its owned non-shared layers. The final rank chooses the greedy token;
the token is broadcast to rank 0, embedded, and decoded as a one-token chunk.
Finished rows emit pad, variable per-row budgets are honored, EOS is counted
with the same current battery semantics, and the complete prefix is never
recomputed per token.

Standard multiple-choice scoring preserves the vendored task order, prompt and
continuation boundaries, right padding, masks, position IDs, normalized
continuation log likelihood, per-option rows, predictions, and aggregate
schema. Only the logit backend changed.

## Entry, failure, and restoration protocol

Before the first payload collective ranks exchange launch hash, epoch, exact
ownership bounds, complete live trainable key sets, and byte-exact SHA-256
fingerprints. Embedding, final norm, LM head, mHC head, and frozen per-layer
input modules must match across ranks and remain unchanged.

Every tokenizer, allocation, owner-compute, head, decode, postprocess,
tokenizer-restoration, and durable-log phase reduces a local failure before a
sibling enters the next payload collective. Injected rank-0 failures therefore
terminate all peers instead of stranding one in NCCL. Evaluation runs under
`torch.inference_mode()`, switches the entire model to eval, and restores each
module's exact prior `training` flag rather than recursively imposing one mode.
Trainable bytes, optimizer-inaccessible weights, frozen vocabulary bytes,
cache ownership, and rotary state are checked on exit.

Foreign-GPU inspection is tri-state. `verified_no_foreign_context` is emitted
only when the process table proves the process opened exactly its configured
physical device. Unavailable inspection records null/unverified, never a
fabricated pass.

## Timing

`distributed_battery` records fixed-sequence validation, standard scoring,
recall generation, optional uncensored generation, rotary page stalls/H2D,
and total boundary seconds. Reconstructed-evaluator fields are retained at
zero for row compatibility: `offload_seconds=0`, `model_load_seconds=0`, and
`adapter_graft_seconds=0`.

Historical subprocess timings remain historical evidence only. Fresh
before/after timing requires a disposable old-revision checkout and a v4.6
checkout on the same copied checkpoint; never disturb a live campaign.

## Certification

CPU-safe protocol coverage:

```bash
TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
TRANSFORMERS_VERBOSITY=error \
/tmp/$USER/selfupdate-venv/bin/python scripts/check_distributed_eval_cpu.py
```

It compares full-model and stage-cut logits, normalized option NLL, task rows,
prefill logits, and five cached greedy tokens for B=1 and variable left
padding. It covers Qwen3, hybrid Qwen3.5, ordinary Gemma4, Gemma shared-KV
across a producer/consumer cut, and Gemma per-layer inputs; checks frozen
producer-KV local-query parity and side-channel file encoding; proves no
parameter mutation and exact mode restoration; and injects asymmetric decode,
postprocess, and logging failures through a two-rank Gloo protocol.

This unit coverage does not replace disposable multi-GPU checkpoint parity.
Certification still compares epoch zero and nonzero LoRA, configured batches
and budgets, complete standard/recall telemetry, Qwen/Gemma fleet checkpoints,
rotary PPP1/PPP2 large-model runs, no foreign CUDA context, and injected-rank
failure. A narrow smoke test is not a parity claim.
