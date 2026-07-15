# Pareto pipeline-v3.1 training progress

Live operational/scientific ledger begun 2026-07-15. This document is updated
as work progresses; it is independent of the individual `report.md` and
`report.pdf` generated inside each completed training run. Dataset identity is
dataset v5 throughout. The training pipeline is v3.1. Qwen3.5-0.8B is the
mechanics/metaparameter model, Qwen3.5-4B is the first promoted flagship, and
successful recipes then continue along the declared Pareto frontier.

## Timing contract

Every launch records node, physical GPU, source commit, exact config/cache
identity, launcher wall time, and exit state. Stage timings are kept separate:
runtime/model load, compiler work, vLLM answer generation, hidden-state
teacher compute, cache storage/finalization, training-only epoch time,
evaluation time, and report/PDF time. Throughput never folds a cold model load
into generated-token or training-token rates.

## Qwen3.5-0.8B epoch zero

Node `agpul05`, physical L40S GPU0. Exact answers:
`runs/vllm_benchmark_l40s/qwen35_0p8b_fixed4096_exactids_agpul05/responses_bs256.jsonl`.
Published node-local hidden cache:
`/dev/shm/arivero/selfupdate-teacher-cache-v3/Qwen3.5-0.8B-rag_system-remove-b632054c01558f61`.

| stage | commit/runtime | result | measured time and rate |
|---|---|---|---|
| fixed-ceiling answer generation | `710c32e`; vLLM 0.25.0, torch 2.11.0+cu129 | 2,071/2,071; 947,644 tokens; 6.33% hard cuts; mean task score 0.6270 | model/runtime load 318.8 s; `torch.compile` 92.73 s within startup; generation 197.3 s at 4,802 tok/s; launcher wall 636 s |
| exact-token L40S hidden pass | `a40adb6`; torch 2.7.1+cu126 | 24 bfloat16 target layers; 50 GiB; cache hash `b632054c01558f61`; atomic ready publication | teacher forward 259.6 s; D2H 2.61 s; storage 23.12 s; cache write accounting 44.32 s; measured total 291.9 s; launcher wall 334 s |

The earlier mixed short-ceiling cache had 58.52% hard cuts. Restoring the
fixed 4,096-token allowance recovered the expected completion regime and
therefore changed the cache identity. The old cache is not a training target.
Compiler artifacts accidentally used Lustre on this first launch; subsequent
vLLM/Triton/Inductor caches default to node-local `/tmp` (commit `a40adb6`).

## Hybrid B×K certification

Qwen3.5-0.8B has 24 blocks, alternating three `linear_attention` Gated
DeltaNet blocks with one full-attention block. Commit `486f961` separates the
current-chunk `[B,K]` recurrent mask from the full-attention
`[B,1,K,prefix+K]` causal/flow mask and excludes finished cells from the loss
sum. It also prevents intact probes from accidentally masking privileged RAG.

| time | probe | node/GPU | status | timing/result |
|---|---|---|---|---|
| 22:07 | flow B256K1 | `agpul05`/GPU1 | failed before GPU work | 9 s; new base omitted `generation_budget_bucket: 32`, resolving cache `848655…` instead of ready `b63205…` |
| 22:07 | flow B256K16 | `agpul05`/GPU2 | failed before GPU work | same fail-fast identity defect, 9 s |
| 22:09 | flow B256K1 retry | `agpul05`/GPU1 | invalid empty success | cache identity restored, but a misplaced helper made the tile body unreachable; wrapper exited 0 after 10 s without a result |
| 22:09 | flow B256K16 retry | `agpul05`/GPU2 | invalid empty success | same control-flow defect; no tile and no weight update occurred |
| 22:14 | flow B256K1 retry 2 | `agpul05`/GPU1 | passed | commit `87729c7`; 256 events; tile 2.251 s / 113.7 events/s; end-to-end 5.40 events/s; 14.86 GiB peak; 24 physical block writes; launcher 58 s |
| 22:14 | flow B256K16 retry 2 | `agpul05`/GPU2 | passed | commit `87729c7`; 4,096 events; tile 11.569 s / 354.1 events/s; end-to-end 72.4 events/s; 15.68 GiB peak; 24 physical block writes; launcher 67 s |
| 22:17 | intact B256K1 | `agpul05`/GPU1 | running | numerical-noise and maximum-compute control |
| 22:17 | intact B256K16 | `agpul05`/GPU2 | running | numerical-noise and maximum-compute control |

The first failures and invalid empty exits are retained because launch/retry
time is part of the operational result. They did not perform a training tile
or modify weights. Commit `87729c7` moves the helper out of `main()` and makes
a missing passed-result payload fail loudly.
After the flow probes pass, intact B256K1/B256K16 establish the numerical-noise
and maximum-compute timing controls before the scientific Wave-A queue opens.

## Overnight progression rule

Each scientific 0.8B arm runs six complete dataset-v5 epochs (12,426 answer
visits), publishes its checkpoint, locality certificate, individual Markdown
report, PDF, and completion-ordered report symlink, then becomes eligible for
Pareto selection. Promoted Qwen3.5-4B arms run six epochs and extend to 12 when
measured throughput makes that practical. Selection uses recall, standard
damage, intrusion, layerwise loss/delta dynamics, locality, and elapsed time;
loss alone does not promote a run.
