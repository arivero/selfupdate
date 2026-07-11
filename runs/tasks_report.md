# Checkpoint re-evaluation: recall and model damage

Recall artifacts exist for 41/41 checkpoints; 41/41 cover every corpus in the declared training scope. Paired standard-capability results exist for 41/41 checkpoints. There are 19 corpus-specific recall base references.

Recall word accuracy is the fraction of reference words recovered in order, averaged over next, previous, and cloze prompts. Each Δ uses the epoch-zero model on the same corpus. A dash means that corpus/reference has not been evaluated; it is never imputed from the other author.

## Trained for Machado

| run | model | Machado | epoch 0 | Δ |
|---|---|---:|---:|---:|
| lw_b_mixed_strict_0p6b_rag | Qwen3-0.6B | 0.10 | 0.07 | +0.03 |
| lw_b_tc_frozen_0p6b_rag | Qwen3-0.6B | 0.10 | 0.07 | +0.03 |
| lw_b_tc_lora_0p6b_rag | Qwen3-0.6B | 0.09 | 0.07 | +0.03 |
| lw_i_cosine_0p6b_rag | Qwen3-0.6B | 0.23 | 0.07 | +0.16 |
| lw_i_huber_0p6b_rag | Qwen3-0.6B | 0.19 | 0.07 | +0.13 |
| lw_i_nmse_s43_0p6b_rag | Qwen3-0.6B | 0.18 | 0.07 | +0.11 |
| lw_i_nmse_strict_0p6b_rag | Qwen3-0.6B | 0.16 | 0.07 | +0.09 |
| lw_k_tailonly_0p6b_rag | Qwen3-0.6B | 0.15 | 0.07 | +0.08 |
| lw_k_tailonly_anchor_0p6b_rag | Qwen3-0.6B | 0.13 | 0.07 | +0.06 |
| lw_k_tailonly_anchorkl_0p6b_rag | Qwen3-0.6B | 0.14 | 0.07 | +0.07 |
| lw_k_v4strict_0p6b_rag | Qwen3-0.6B | 0.11 | 0.07 | +0.04 |
| lw_o_fisher_0p6b_rag | Qwen3-0.6B | 0.21 | 0.07 | +0.14 |
| lw_r_lensdeep2_0p6b_rag | Qwen3-0.6B | 0.19 | 0.07 | +0.12 |
| lw_r_lensonly_0p6b_rag | Qwen3-0.6B | 0.12 | 0.07 | +0.05 |
| lw_s_lenskluni_0p6b_rag | Qwen3-0.6B | 0.21 | 0.07 | +0.14 |
| lw_s_tcmodern_0p6b_rag | Qwen3-0.6B | 0.10 | 0.07 | +0.03 |
| lw_l_tailonly_4b_rag | Qwen3-4B | 0.15 | 0.10 | +0.05 |

## Trained for Machado + Quijote ch1

| run | model | Machado | epoch 0 | Δ | Quijote ch1 | epoch 0 | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| a_lossgrid_1p7b_combined_slide1_cosine | Qwen3-1.7B | 0.14 | 0.11 | +0.03 | 0.17 | 0.14 | +0.04 |
| a_lossgrid_1p7b_combined_slide1_delta_cosine | Qwen3-1.7B | 0.15 | 0.11 | +0.05 | 0.15 | 0.14 | +0.02 |
| a_lossgrid_1p7b_combined_slide1_delta_nmse | Qwen3-1.7B | 0.14 | 0.11 | +0.04 | 0.14 | 0.14 | +0.00 |
| a_lossgrid_1p7b_combined_slide1_delta_vocab_cos | Qwen3-1.7B | 0.16 | 0.11 | +0.05 | 0.14 | 0.14 | -0.00 |
| a_lossgrid_1p7b_combined_slide1_huber | Qwen3-1.7B | 0.17 | 0.11 | +0.06 | 0.16 | 0.14 | +0.02 |
| a_lossgrid_1p7b_combined_slide1_jacobian_lens_kl | Qwen3-1.7B | 0.25 | 0.11 | +0.14 | 0.23 | 0.14 | +0.10 |
| a_lossgrid_1p7b_combined_slide1_jacobian_vocab_mse | Qwen3-1.7B | 0.13 | 0.11 | +0.03 | 0.13 | 0.14 | -0.00 |
| a_lossgrid_1p7b_combined_slide1_l2mse | Qwen3-1.7B | 0.11 | 0.11 | +0.01 | 0.14 | 0.14 | +0.00 |
| a_lossgrid_1p7b_combined_slide1_lenskl | Qwen3-1.7B | 0.22 | 0.11 | +0.11 | 0.22 | 0.14 | +0.08 |
| a_lossgrid_1p7b_combined_slide1_nmse | Qwen3-1.7B | 0.16 | 0.11 | +0.05 | 0.18 | 0.14 | +0.04 |
| a_lossgrid_1p7b_combined_slide1_vocab | Qwen3-1.7B | 0.12 | 0.11 | +0.02 | 0.13 | 0.14 | -0.01 |
| a_lossgrid_1p7b_combined_slide1_vocabfisher | Qwen3-1.7B | 0.15 | 0.11 | +0.05 | 0.16 | 0.14 | +0.02 |
| a_lossgrid_1p7b_combined_slide2_cosine | Qwen3-1.7B | 0.22 | 0.11 | +0.12 | 0.16 | 0.14 | +0.02 |
| a_lossgrid_1p7b_combined_slide2_delta_cosine | Qwen3-1.7B | 0.14 | 0.11 | +0.04 | 0.17 | 0.14 | +0.03 |
| a_lossgrid_1p7b_combined_slide2_delta_nmse | Qwen3-1.7B | 0.19 | 0.11 | +0.09 | 0.18 | 0.14 | +0.04 |
| a_lossgrid_1p7b_combined_slide2_delta_vocab_cos | Qwen3-1.7B | 0.21 | 0.11 | +0.10 | 0.17 | 0.14 | +0.04 |
| a_lossgrid_1p7b_combined_slide2_huber | Qwen3-1.7B | 0.24 | 0.11 | +0.14 | 0.18 | 0.14 | +0.04 |
| a_lossgrid_1p7b_combined_slide2_jacobian_lens_kl | Qwen3-1.7B | 0.18 | 0.11 | +0.08 | 0.19 | 0.14 | +0.05 |
| a_lossgrid_1p7b_combined_slide2_jacobian_vocab_mse | Qwen3-1.7B | 0.14 | 0.11 | +0.03 | 0.15 | 0.14 | +0.02 |
| a_lossgrid_1p7b_combined_slide2_l2mse | Qwen3-1.7B | 0.15 | 0.11 | +0.05 | 0.17 | 0.14 | +0.03 |
| a_lossgrid_1p7b_combined_slide2_lenskl | Qwen3-1.7B | 0.19 | 0.11 | +0.09 | 0.22 | 0.14 | +0.08 |
| a_lossgrid_1p7b_combined_slide2_nmse | Qwen3-1.7B | 0.21 | 0.11 | +0.10 | 0.18 | 0.14 | +0.04 |
| a_lossgrid_1p7b_combined_slide2_vocab | Qwen3-1.7B | 0.16 | 0.11 | +0.05 | 0.16 | 0.14 | +0.02 |
| a_lossgrid_1p7b_combined_slide2_vocabfisher | Qwen3-1.7B | 0.14 | 0.11 | +0.04 | 0.19 | 0.14 | +0.05 |

## Model damage: fixed standard benchmark subsets

This is the capability check. Accuracy and Δ are paired means over the standard tasks available in both checkpoint and epoch-zero artifacts — the primary suite is ARC-Easy, ARC-Challenge, and HellaSwag (n=100 fixed subsets); legacy destruction.json fallbacks may add HellaSwag/MMLU/ARC-Challenge/WinoGrande/MMLU-Pro at n=200. Negative Δ means lost general knowledge/skill. The old custom prose-loss task is no longer part of checkpoint re-evaluation.

| run | model | paired/base tasks | accuracy | epoch 0 | Δ | worst loss |
|---|---|---:|---:|---:|---:|---|
| lw_b_mixed_strict_0p6b_rag | Qwen3-0.6B | 3/3 | 0.39 | 0.46 | -0.06 | arc_easy -0.09 |
| lw_b_tc_frozen_0p6b_rag | Qwen3-0.6B | 3/3 | 0.40 | 0.46 | -0.06 | arc_easy -0.12 |
| lw_b_tc_lora_0p6b_rag | Qwen3-0.6B | 3/3 | 0.36 | 0.46 | -0.10 | arc_easy -0.14 |
| lw_i_cosine_0p6b_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.05 | arc_easy -0.13 |
| lw_i_huber_0p6b_rag | Qwen3-0.6B | 3/3 | 0.39 | 0.46 | -0.06 | arc_easy -0.17 |
| lw_i_nmse_s43_0p6b_rag | Qwen3-0.6B | 3/3 | 0.39 | 0.46 | -0.07 | arc_easy -0.16 |
| lw_i_nmse_strict_0p6b_rag | Qwen3-0.6B | 3/3 | 0.40 | 0.46 | -0.06 | arc_easy -0.12 |
| lw_k_tailonly_0p6b_rag | Qwen3-0.6B | 3/3 | 0.37 | 0.46 | -0.09 | arc_easy -0.18 |
| lw_k_tailonly_anchor_0p6b_rag | Qwen3-0.6B | 3/3 | 0.37 | 0.46 | -0.09 | arc_easy -0.20 |
| lw_k_tailonly_anchorkl_0p6b_rag | Qwen3-0.6B | 3/3 | 0.36 | 0.46 | -0.10 | arc_easy -0.19 |
| lw_k_v4strict_0p6b_rag | Qwen3-0.6B | 3/3 | 0.36 | 0.46 | -0.09 | arc_easy -0.18 |
| lw_o_fisher_0p6b_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.05 | arc_easy -0.09 |
| lw_r_lensdeep2_0p6b_rag | Qwen3-0.6B | 3/3 | 0.38 | 0.46 | -0.08 | arc_easy -0.14 |
| lw_r_lensonly_0p6b_rag | Qwen3-0.6B | 3/3 | 0.29 | 0.46 | -0.17 | arc_easy -0.24 |
| lw_s_lenskluni_0p6b_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.05 | arc_easy -0.09 |
| lw_s_tcmodern_0p6b_rag | Qwen3-0.6B | 3/3 | 0.38 | 0.46 | -0.08 | arc_easy -0.16 |
| lw_l_tailonly_4b_rag | Qwen3-4B | 3/3 | 0.58 | 0.58 | +0.00 | arc_easy -0.01 |
| a_lossgrid_1p7b_combined_slide1_cosine | Qwen3-1.7B | 3/3 | 0.49 | 0.52 | -0.04 | hellaswag -0.04 |
| a_lossgrid_1p7b_combined_slide1_delta_cosine | Qwen3-1.7B | 3/3 | 0.51 | 0.52 | -0.01 | hellaswag -0.02 |
| a_lossgrid_1p7b_combined_slide1_delta_nmse | Qwen3-1.7B | 3/3 | 0.52 | 0.52 | -0.01 | arc_easy -0.02 |
| a_lossgrid_1p7b_combined_slide1_delta_vocab_cos | Qwen3-1.7B | 3/3 | 0.50 | 0.52 | -0.02 | arc_challenge -0.05 |
| a_lossgrid_1p7b_combined_slide1_huber | Qwen3-1.7B | 3/3 | 0.49 | 0.52 | -0.03 | hellaswag -0.04 |
| a_lossgrid_1p7b_combined_slide1_jacobian_lens_kl | Qwen3-1.7B | 3/3 | 0.52 | 0.52 | -0.01 | arc_easy -0.06 |
| a_lossgrid_1p7b_combined_slide1_jacobian_vocab_mse | Qwen3-1.7B | 3/3 | 0.50 | 0.52 | -0.03 | arc_challenge -0.07 |
| a_lossgrid_1p7b_combined_slide1_l2mse | Qwen3-1.7B | 3/3 | 0.52 | 0.52 | -0.01 | arc_easy -0.03 |
| a_lossgrid_1p7b_combined_slide1_lenskl | Qwen3-1.7B | 3/3 | 0.52 | 0.52 | -0.00 | arc_challenge -0.04 |
| a_lossgrid_1p7b_combined_slide1_nmse | Qwen3-1.7B | 3/3 | 0.50 | 0.52 | -0.02 | hellaswag -0.03 |
| a_lossgrid_1p7b_combined_slide1_vocab | Qwen3-1.7B | 3/3 | 0.52 | 0.52 | +0.00 | arc_easy -0.04 |
| a_lossgrid_1p7b_combined_slide1_vocabfisher | Qwen3-1.7B | 3/3 | 0.47 | 0.52 | -0.05 | arc_easy -0.15 |
| a_lossgrid_1p7b_combined_slide2_cosine | Qwen3-1.7B | 3/3 | 0.53 | 0.52 | +0.00 | arc_easy -0.04 |
| a_lossgrid_1p7b_combined_slide2_delta_cosine | Qwen3-1.7B | 3/3 | 0.43 | 0.52 | -0.09 | arc_easy -0.28 |
| a_lossgrid_1p7b_combined_slide2_delta_nmse | Qwen3-1.7B | 3/3 | 0.49 | 0.52 | -0.03 | hellaswag -0.04 |
| a_lossgrid_1p7b_combined_slide2_delta_vocab_cos | Qwen3-1.7B | 3/3 | 0.50 | 0.52 | -0.03 | arc_easy -0.09 |
| a_lossgrid_1p7b_combined_slide2_huber | Qwen3-1.7B | 3/3 | 0.52 | 0.52 | -0.01 | arc_easy -0.03 |
| a_lossgrid_1p7b_combined_slide2_jacobian_lens_kl | Qwen3-1.7B | 3/3 | 0.53 | 0.52 | +0.00 | arc_challenge -0.02 |
| a_lossgrid_1p7b_combined_slide2_jacobian_vocab_mse | Qwen3-1.7B | 3/3 | 0.53 | 0.52 | +0.01 | arc_challenge -0.03 |
| a_lossgrid_1p7b_combined_slide2_l2mse | Qwen3-1.7B | 3/3 | 0.54 | 0.52 | +0.02 | arc_easy +0.00 |
| a_lossgrid_1p7b_combined_slide2_lenskl | Qwen3-1.7B | 3/3 | 0.54 | 0.52 | +0.02 | arc_challenge -0.05 |
| a_lossgrid_1p7b_combined_slide2_nmse | Qwen3-1.7B | 3/3 | 0.52 | 0.52 | +0.00 | arc_easy -0.01 |
| a_lossgrid_1p7b_combined_slide2_vocab | Qwen3-1.7B | 3/3 | 0.54 | 0.52 | +0.02 | arc_easy -0.02 |
| a_lossgrid_1p7b_combined_slide2_vocabfisher | Qwen3-1.7B | 3/3 | 0.52 | 0.52 | -0.01 | arc_challenge -0.06 |

## Corpus-specific base references

- BSC-LT/ALIA-40b-fc-2606 — Machado: 0.13
- BSC-LT/ALIA-40b-fc-2606 — Quijote ch1: 0.20
- Qwen/Qwen3-0.6B — Machado: 0.07
- Qwen/Qwen3-0.6B — Quijote ch1: 0.09
- Qwen/Qwen3-0.6B — Quijote ch16: 0.08
- Qwen/Qwen3-0.6B — Quijote ch4: 0.09
- Qwen/Qwen3-0.6B — Quijote ch8: 0.08
- Qwen/Qwen3-1.7B — Machado: 0.11
- Qwen/Qwen3-1.7B — Quijote ch1: 0.14
- Qwen/Qwen3-14B — Machado: 0.13
- Qwen/Qwen3-14B — Quijote ch1: 0.14
- Qwen/Qwen3-4B — Machado: 0.10
- Qwen/Qwen3-4B — Quijote ch1: 0.10
- Qwen/Qwen3-8B — Machado: 0.14
- Qwen/Qwen3-8B — Quijote ch1: 0.13
- meta-llama/Llama-3.1-8B-Instruct — Machado: 0.13
- microsoft/Phi-4-mini-reasoning — Machado: 0.00
- mistralai/Mistral-7B-Instruct-v0.1 — Machado: 0.09
- openai/gpt-oss-20b — Machado: 0.00
