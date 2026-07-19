# Next-lease runbook — finish the vLLM-reproduction goal

Standing goal: trainer-native `teacher_argmax_acceptance` for ALL models;
large models on 4/8 cards. Done so far: 0.8B PPP1/2/4 bit-identical
(RESULTS.md); torch baselines for 6 models.

## 1. Trainer-native rows, remaining single-node models (~15 min total)
One command per row (subset+cache+PPP1+extract; add `[gpu]` to spread):
    scripts/spec_verify_matrix.sh 27b 0
    scripts/spec_verify_matrix.sh 35b 1
    scripts/spec_verify_matrix.sh 26b 2   # gemma: verify budget check passes; responses were l40s campaign
    scripts/spec_verify_matrix.sh 31b 3
    scripts/spec_verify_matrix.sh 122b    # PPP1 rotary or use a PPP4 overlay (244GB > 1 card resident)
NOTE 122b: single-card PPP1 needs v4_weight_residency rotate (stage-scoped);
simplest correct row = PPP4 overlay (copy 0p8b ppp4 overlay, adjust splits
to 48 layers: [12,24,36]).

## 2. PPP8 cross-node trainer row (0.8B vehicle, then 122B)
Prereqs on agpuh02: scripts/venv_setup.sh; stage_hf_cache.sh --shm <model>;
build_teacher_cache --index-only --coordinated-node-cache (same config).
    SELFUPDATE_V4_STAGE_HOSTS="agpuh01 agpuh01 agpuh01 agpuh01 agpuh02 agpuh02 agpuh02 agpuh02" \
    scripts/launch_v4_stages.sh configs/experiments/spec_verify/base_qwen35_0p8b_v4_spec.yaml \
                                configs/experiments/spec_verify/qwen35_0p8b_v4_spec_ppp8x.yaml
Goal row appears on the LAST stage (stage7, agpuh02). Expect
0.9854211663066955 bit-exact.

## 3. 397B: genuine reference REQUIRED first (finding 2026-07-19 17:47)
runs/vllm_h100/qwen35_397b_a17b_fp8/responses_bs256.jsonl is BYTE-IDENTICAL
to the 122B answers (verified first-row; provenance = the reuse note). It is
valid as training data (same tokenizer) but INVALID as a 397B reproduction
reference. Sequence:
  a. Two-node vLLM deploy of 397B-FP8 (~400GB): vLLM multi-node (ray),
     TP4 x PP2 over agpuh01+agpuh02, greedy, same prompts
     (benchmark_vllm_generation.py needs a --distributed path or a ray
     cluster + pipeline_parallel_size=2).
  b. Trainer-native row: index-only cache from (a) + v4 PPP8 store+rotate
     (the proven 397B training config) with the acceptance metric.
  c. The torch-baseline equivalent needs the 2-stage relay verify (752GB
     bf16 dequant cannot single-process even across 4 cards).

## 4. DeepSeek: regenerate responses WITH prompt_token_ids
Its vLLM answer-gen was blocked on driver-565 fp4 kernels (notebook
2026-07-18 16:05). If still blocked, DeepSeek's row waits for a driver/vLLM
path; do not fake it from the bf16 dequant via transformers generate (not
vLLM, wrong reference).

## 5. Length-matched retest (science hygiene)
0.8B is the only long-answer row (~115 tok vs ~8-16). Re-run 27b/35b/122b
with vLLM budgets forced to ~115 tokens to separate "small model" from
"long answer" in the divergence-depth effect.

## 6. 0.8B divergence — owner hypothesis tested (2026-07-19 17:50)
"Wrong weights/fine-tune" RULED OUT: single snapshot 2fc06364..., refs/main
-> it, Lustre cache and shm stage symlink the SAME blob 04b1c301... — vLLM
and our stack loaded identical bytes. Refined suspect: linear-attention
RUNTIME POLICY (state dtype fp32-vs-bf16 / chunked-scan order) — consistent
with confident depth-growing flips and with fp32 halving the gap (5.09->2.14).
Decisive test queued: vLLM dtype=float32 regen of the 0.8B answers + our
fp32 teacher-force; convergence => precision policy (pin dtype to get
"exact"); non-convergence => structural scan order. Also read vLLM's
qwen3_5 linear-attn kernel for its state dtype (code inspection, no GPU).

## 7. Root-cause narrowed (code inspection, 2026-07-19 17:53)
transformers qwen3_5 linear attention = CHUNKED TORCH implementation,
upcasts q/k/v/beta/g to float32 (modeling_qwen3_5.py:263) and runs the
recurrence in fp32. vLLM = fused Triton FLA kernels (vllm/model_executor/
layers/fla/ops/, v1 GDN backend) — a different algorithm implementation
with its own precision policy. The 0.8B confident divergences are the
accumulated difference between these two implementations over long answers.
Also check: with kernels==0.12.0 installed, does transformers swap in a hub
GDN kernel instead of the torch path? (would change which impl our trainer
actually ran). The fp32-both-sides test (item 6) remains decisive for
whether pinning precision closes it.
