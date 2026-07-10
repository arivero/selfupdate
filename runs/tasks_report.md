# Checkpoint re-evaluation: recall and model damage

Recall artifacts exist for 160/160 checkpoints; 160/160 cover every corpus in the declared training scope. Paired standard-capability results exist for 159/160 checkpoints. There are 19 corpus-specific recall base references.

Recall word accuracy is the fraction of reference words recovered in order, averaged over next, previous, and cloze prompts. Each Δ uses the epoch-zero model on the same corpus. A dash means that corpus/reference has not been evaluated; it is never imputed from the other author.

## Trained for Machado

| run | model | Machado | epoch 0 | Δ |
|---|---|---:|---:|---:|
| clean_machado_slide8_vocab_alia40b_lora | ALIA-40b-fc-2606 | 0.24 | 0.13 | +0.11 |
| certify_anchor_readout | Qwen3-0.6B | 0.07 | 0.07 | -0.00 |
| certify_censored_frozen | Qwen3-0.6B | 0.06 | 0.07 | -0.01 |
| certify_lora_online | Qwen3-0.6B | 0.07 | 0.07 | +0.00 |
| certify_mixed_frozen | Qwen3-0.6B | 0.07 | 0.07 | +0.00 |
| certify_offload_adam | Qwen3-0.6B | 0.07 | 0.07 | +0.00 |
| certify_sequential_subset | Qwen3-0.6B | 0.07 | 0.07 | -0.00 |
| certify_slide4_dedup | Qwen3-0.6B | 0.07 | 0.07 | +0.00 |
| certify_slide4_item | Qwen3-0.6B | 0.07 | 0.07 | -0.00 |
| certify_slide4_readout | Qwen3-0.6B | 0.07 | 0.07 | -0.00 |
| certify_slide4_readout_padded | Qwen3-0.6B | 0.07 | 0.07 | -0.00 |
| certify_summed_bucketed | Qwen3-0.6B | 0.07 | 0.07 | -0.00 |
| certify_summed_item | Qwen3-0.6B | 0.07 | 0.07 | -0.00 |
| certify_summed_padded | Qwen3-0.6B | 0.07 | 0.07 | -0.00 |
| lw_b_mixed_0p6b_rag | Qwen3-0.6B | 0.14 | 0.07 | +0.07 |
| lw_b_mixed_strict_0p6b_rag | Qwen3-0.6B | 0.10 | 0.07 | +0.03 |
| lw_b_tc_frozen_0p6b_rag | Qwen3-0.6B | 0.10 | 0.07 | +0.03 |
| lw_b_tc_lora_0p6b_rag | Qwen3-0.6B | 0.09 | 0.07 | +0.03 |
| lw_i_cosine_0p6b_rag | Qwen3-0.6B | 0.23 | 0.07 | +0.16 |
| lw_i_huber_0p6b_rag | Qwen3-0.6B | 0.19 | 0.07 | +0.13 |
| lw_i_l2mse_0p6b_rag | Qwen3-0.6B | 0.23 | 0.07 | +0.16 |
| lw_i_lenskl_0p6b_rag | Qwen3-0.6B | 0.21 | 0.07 | +0.14 |
| lw_i_lenskl_strict_0p6b_rag | Qwen3-0.6B | 0.17 | 0.07 | +0.10 |
| lw_i_nmse_s43_0p6b_rag | Qwen3-0.6B | 0.18 | 0.07 | +0.11 |
| lw_i_nmse_strict_0p6b_rag | Qwen3-0.6B | 0.16 | 0.07 | +0.09 |
| lw_i_vocab_0p6b_rag | Qwen3-0.6B | 0.19 | 0.07 | +0.12 |
| lw_i_vocab_strict_0p6b_rag | Qwen3-0.6B | 0.12 | 0.07 | +0.06 |
| lw_j_vocab_v3_0p6b_rag | Qwen3-0.6B | 0.14 | 0.07 | +0.07 |
| lw_k_anchor_0p6b_rag | Qwen3-0.6B | 0.18 | 0.07 | +0.11 |
| lw_k_anchorkl_0p6b_rag | Qwen3-0.6B | 0.17 | 0.07 | +0.10 |
| lw_k_final2p_0p6b_rag | Qwen3-0.6B | 0.16 | 0.07 | +0.09 |
| lw_k_final_0p6b_rag | Qwen3-0.6B | 0.18 | 0.07 | +0.11 |
| lw_k_final_k8_0p6b_rag | Qwen3-0.6B | 0.23 | 0.07 | +0.16 |
| lw_k_final_s43_0p6b_rag | Qwen3-0.6B | 0.17 | 0.07 | +0.10 |
| lw_k_maieutic_0p6b_rag | Qwen3-0.6B | 0.21 | 0.07 | +0.15 |
| lw_k_tailonly_0p6b_rag | Qwen3-0.6B | 0.15 | 0.07 | +0.08 |
| lw_k_tailonly_anchor_0p6b_rag | Qwen3-0.6B | 0.13 | 0.07 | +0.06 |
| lw_k_tailonly_anchorkl_0p6b_rag | Qwen3-0.6B | 0.14 | 0.07 | +0.07 |
| lw_k_v4strict_0p6b_rag | Qwen3-0.6B | 0.11 | 0.07 | +0.04 |
| lw_k_vocab_s43_0p6b_rag | Qwen3-0.6B | 0.15 | 0.07 | +0.08 |
| lw_lens_ce_0p6b_rag | Qwen3-0.6B | 0.13 | 0.07 | +0.06 |
| lw_lens_ce_deep_0p6b_rag | Qwen3-0.6B | 0.23 | 0.07 | +0.16 |
| lw_lens_ce_lora_0p6b_rag | Qwen3-0.6B | 0.15 | 0.07 | +0.08 |
| lw_lens_kl_0p6b_rag | Qwen3-0.6B | 0.16 | 0.07 | +0.10 |
| lw_lens_kl_deep_0p6b_rag | Qwen3-0.6B | 0.25 | 0.07 | +0.18 |
| lw_m_anchordiv_0p6b_rag | Qwen3-0.6B | 0.21 | 0.07 | +0.14 |
| lw_m_anchordiv_s43_0p6b_rag | Qwen3-0.6B | 0.19 | 0.07 | +0.12 |
| lw_m_anchordiv_w1_0p6b_rag | Qwen3-0.6B | 0.17 | 0.07 | +0.10 |
| lw_n_thinksel_0p6b_rag | Qwen3-0.6B | 0.13 | 0.07 | +0.06 |
| lw_n_thinksel_s43_0p6b_rag | Qwen3-0.6B | 0.13 | 0.07 | +0.06 |
| lw_n_thinkselkl_0p6b_rag | Qwen3-0.6B | 0.10 | 0.07 | +0.03 |
| lw_n_thinkslide_0p6b_rag | Qwen3-0.6B | 0.13 | 0.07 | +0.07 |
| lw_n_thinkwhole_0p6b_rag | Qwen3-0.6B | 0.09 | 0.07 | +0.02 |
| lw_o_fisher_0p6b_rag | Qwen3-0.6B | 0.21 | 0.07 | +0.14 |
| lw_q_pp2_0p6b_rag | Qwen3-0.6B | 0.12 | 0.07 | +0.05 |
| lw_q_pp2fix_0p6b_rag | Qwen3-0.6B | 0.20 | 0.07 | +0.13 |
| lw_r_disj_pinned | Qwen3-0.6B | 0.19 | 0.07 | +0.12 |
| lw_r_lensdeep2_0p6b_rag | Qwen3-0.6B | 0.19 | 0.07 | +0.12 |
| lw_r_lensonly_0p6b_rag | Qwen3-0.6B | 0.12 | 0.07 | +0.05 |
| lw_r_s43_pinned | Qwen3-0.6B | 0.20 | 0.07 | +0.13 |
| lw_r_slide2_0p6b_rag | Qwen3-0.6B | 0.17 | 0.07 | +0.11 |
| lw_r_slide4_0p6b_rag | Qwen3-0.6B | 0.20 | 0.07 | +0.13 |
| lw_r_slide6pure_0p6b_rag | Qwen3-0.6B | 0.15 | 0.07 | +0.08 |
| lw_r_slide8_0p6b_rag | Qwen3-0.6B | 0.19 | 0.07 | +0.12 |
| lw_r_slide8disj_0p6b_rag | Qwen3-0.6B | 0.16 | 0.07 | +0.09 |
| lw_r_slide8kl_0p6b_rag | Qwen3-0.6B | 0.21 | 0.07 | +0.14 |
| lw_r_slide8pure_0p6b_rag | Qwen3-0.6B | 0.19 | 0.07 | +0.12 |
| lw_r_slide8pure_s43_0p6b_rag | Qwen3-0.6B | 0.19 | 0.07 | +0.12 |
| lw_r_tailpure_0p6b_rag | Qwen3-0.6B | 0.19 | 0.07 | +0.12 |
| lw_s_lenskluni_0p6b_rag | Qwen3-0.6B | 0.21 | 0.07 | +0.14 |
| lw_s_slide8lenskl_0p6b_rag | Qwen3-0.6B | 0.25 | 0.07 | +0.18 |
| lw_s_slide8nmse_0p6b_rag | Qwen3-0.6B | 0.20 | 0.07 | +0.13 |
| lw_s_tcmodern_0p6b_rag | Qwen3-0.6B | 0.10 | 0.07 | +0.03 |
| lw_seq_0p6b_rag | Qwen3-0.6B | 0.09 | 0.07 | +0.03 |
| lw_seq_bf16_0p6b_rag | Qwen3-0.6B | 0.10 | 0.07 | +0.03 |
| lw_summed_0p6b_rag | Qwen3-0.6B | 0.12 | 0.07 | +0.05 |
| lw_summed_ce_0p6b_rag | Qwen3-0.6B | 0.11 | 0.07 | +0.04 |
| lw_summed_ce_e40_0p6b_rag | Qwen3-0.6B | 0.17 | 0.07 | +0.10 |
| lw_summed_e40_0p6b_rag | Qwen3-0.6B | 0.14 | 0.07 | +0.07 |
| lw_summed_l2mse_e40_0p6b_rag | Qwen3-0.6B | 0.12 | 0.07 | +0.05 |
| lw_t_ragchannel_0p6b_rag | Qwen3-0.6B | 0.20 | 0.07 | +0.13 |
| lw_tail_ce_0p6b_rag | Qwen3-0.6B | 0.12 | 0.07 | +0.05 |
| lw_tail_ce_e40_0p6b_rag | Qwen3-0.6B | 0.18 | 0.07 | +0.11 |
| lw_tail_ce_e40_s2_0p6b_rag | Qwen3-0.6B | 0.20 | 0.07 | +0.13 |
| lw_tail_ce_e40_v2_0p6b_rag | Qwen3-0.6B | 0.22 | 0.07 | +0.15 |
| lw_tail_ce_k2_e40_0p6b_rag | Qwen3-0.6B | 0.19 | 0.07 | +0.13 |
| lw_tail_ce_lora_0p6b_rag | Qwen3-0.6B | 0.10 | 0.07 | +0.04 |
| lw_tc_lora_e40_0p6b_rag | Qwen3-0.6B | 0.10 | 0.07 | +0.03 |
| pp2diag_lora_single_0p6b | Qwen3-0.6B | 0.09 | 0.07 | +0.03 |
| pp2diag_noanchor_single_0p6b | Qwen3-0.6B | 0.21 | 0.07 | +0.14 |
| tl_i_tunedlenskl_strict_0p6b_rag | Qwen3-0.6B | 0.11 | 0.07 | +0.05 |
| dev | Qwen3-1.7B | 0.10 | 0.11 | -0.01 |
| lw_j_l2mse_k4_1p7b_rag | Qwen3-1.7B | 0.16 | 0.11 | +0.05 |
| lw_j_vocab_k2_1p7b_rag | Qwen3-1.7B | 0.16 | 0.11 | +0.05 |
| lw_j_vocab_k4_1p7b_rag | Qwen3-1.7B | 0.19 | 0.11 | +0.08 |
| lw_j_vocab_k8_1p7b_rag | Qwen3-1.7B | 0.13 | 0.11 | +0.02 |
| lw_k_final_1p7b_rag | Qwen3-1.7B | 0.17 | 0.11 | +0.06 |
| lw_m_anchordiv_1p7b_rag | Qwen3-1.7B | 0.11 | 0.11 | +0.01 |
| lw_n_thinksel_1p7b_rag | Qwen3-1.7B | 0.12 | 0.11 | +0.02 |
| lw_r_slide8pure_1p7b_rag | Qwen3-1.7B | 0.22 | 0.11 | +0.12 |
| lw_seq_1p7b_rag | Qwen3-1.7B | 0.11 | 0.11 | +0.00 |
| xs_fisher_1p7b_rag | Qwen3-1.7B | 0.18 | 0.11 | +0.07 |
| xs_slide2_1p7b_rag | Qwen3-1.7B | 0.19 | 0.11 | +0.08 |
| xs_slide4_1p7b_rag | Qwen3-1.7B | 0.21 | 0.11 | +0.11 |
| clean_machado_slide8_vocab_14b_lora | Qwen3-14B | 0.18 | 0.13 | +0.05 |
| lw_l_final_14b_rag | Qwen3-14B | 0.15 | 0.13 | +0.03 |
| lw_tc_lora_14b_rag | Qwen3-14B | 0.10 | 0.13 | -0.02 |
| clean_machado_slide8_vocab_4b_lora | Qwen3-4B | 0.09 | 0.10 | -0.01 |
| lw_j_qwen4b_rag | Qwen3-4B | 0.15 | 0.10 | +0.05 |
| lw_l_final_4b_av2_rag | Qwen3-4B | 0.15 | 0.10 | +0.05 |
| lw_l_final_4b_av2lr_rag | Qwen3-4B | 0.13 | 0.10 | +0.03 |
| lw_l_final_4b_r64_rag | Qwen3-4B | 0.24 | 0.10 | +0.15 |
| lw_l_final_4b_rag | Qwen3-4B | 0.17 | 0.10 | +0.07 |
| lw_l_tailonly_4b_rag | Qwen3-4B | 0.15 | 0.10 | +0.05 |
| lw_p_ft4b_rag | Qwen3-4B | 0.26 | 0.10 | +0.16 |
| clean_machado_slide8_vocab_8b_lora | Qwen3-8B | 0.15 | 0.14 | +0.01 |
| lw_j_qwen8b_rag | Qwen3-8B | 0.18 | 0.14 | +0.04 |
| lw_l_final_8b_av2_rag | Qwen3-8B | 0.15 | 0.14 | +0.00 |
| lw_l_final_8b_rag | Qwen3-8B | 0.17 | 0.14 | +0.03 |
| lw_j_llama8b_rag | Llama-3.1-8B-Instruct | 0.17 | 0.13 | +0.04 |
| lw_j_phi4mini_rag | Phi-4-mini-reasoning | 0.09 | 0.00 | +0.09 |
| lw_j_mistral7b_rag | Mistral-7B-Instruct-v0.1 | 0.18 | 0.09 | +0.09 |
| lw_k_gptoss_rag | gpt-oss-20b | 0.00 | 0.00 | +0.00 |

## Trained for Quijote ch1

| run | model | Quijote ch1 | epoch 0 | Δ |
|---|---|---:|---:|---:|
| clean_q_ch1_slide4_vocab_0p6b_e320_rag | Qwen3-0.6B | 0.20 | 0.09 | +0.11 |
| clean_q_ch1_slide4_vocab_0p6b_rag | Qwen3-0.6B | 0.13 | 0.09 | +0.04 |
| clean_q_ch1_slide8_nmse_0p6b_e320_rag | Qwen3-0.6B | 0.18 | 0.09 | +0.09 |
| clean_q_ch1_slide8_nmse_0p6b_rag | Qwen3-0.6B | 0.14 | 0.09 | +0.05 |
| clean_q_ch1_slide8_vocab_0p6b_e320_rag | Qwen3-0.6B | 0.14 | 0.09 | +0.05 |
| clean_q_ch1_slide8_vocab_0p6b_rag | Qwen3-0.6B | 0.18 | 0.09 | +0.09 |
| q_ch1_0p6b_rag | Qwen3-0.6B | 0.14 | 0.09 | +0.05 |
| q_ch1_ext_0p6b_rag | Qwen3-0.6B | 0.16 | 0.09 | +0.07 |
| clean_q_ch1_slide2_vocab_1p7b_rag | Qwen3-1.7B | 0.18 | 0.14 | +0.04 |
| clean_q_ch1_slide4_vocab_1p7b_rag | Qwen3-1.7B | 0.16 | 0.14 | +0.02 |
| clean_q_ch1_slide8_vocab_1p7b_e320_rag | Qwen3-1.7B | 0.15 | 0.14 | +0.01 |
| clean_q_ch1_slide8_vocab_1p7b_rag | Qwen3-1.7B | 0.19 | 0.14 | +0.05 |
| clean_q_ch1_slide2_vocab_14b_lora | Qwen3-14B | 0.14 | 0.14 | +0.00 |
| clean_q_ch1_slide4_vocab_14b_lora | Qwen3-14B | 0.20 | 0.14 | +0.06 |
| clean_q_ch1_slide8_lenskl_14b_lora | Qwen3-14B | 0.21 | 0.14 | +0.07 |
| clean_q_ch1_slide8_nmse_14b_lora | Qwen3-14B | 0.28 | 0.14 | +0.14 |
| clean_q_ch1_slide8_vocab_14b_lora | Qwen3-14B | 0.25 | 0.14 | +0.12 |
| clean_q_ch1_slide8_vocab_4b_lora | Qwen3-4B | 0.13 | 0.10 | +0.03 |
| clean_q_ch1_slide8_vocab_8b_lora | Qwen3-8B | 0.15 | 0.13 | +0.02 |

## Trained for Quijote ch4

| run | model | Quijote ch4 | epoch 0 | Δ |
|---|---|---:|---:|---:|
| q_ch4_0p6b_rag | Qwen3-0.6B | 0.14 | 0.09 | +0.05 |
| q_ch4_av2_0p6b_rag | Qwen3-0.6B | 0.16 | 0.09 | +0.07 |
| q_ch4_av2_s43_0p6b_rag | Qwen3-0.6B | 0.15 | 0.09 | +0.06 |

## Trained for Quijote ch8

| run | model | Quijote ch8 | epoch 0 | Δ |
|---|---|---:|---:|---:|
| q_ch8_0p6b_rag | Qwen3-0.6B | 0.14 | 0.08 | +0.06 |
| q_ch8_av2_0p6b_rag | Qwen3-0.6B | 0.15 | 0.08 | +0.07 |
| q_ch8_slide8pure_0p6b_rag | Qwen3-0.6B | 0.12 | 0.08 | +0.04 |

## Trained for Quijote ch16

| run | model | Quijote ch16 | epoch 0 | Δ |
|---|---|---:|---:|---:|
| q_ch16_0p6b_rag | Qwen3-0.6B | 0.15 | 0.08 | +0.06 |
| q_ch16_av2_0p6b_rag | Qwen3-0.6B | 0.17 | 0.08 | +0.09 |
| q_ch16_ext_0p6b_rag | Qwen3-0.6B | 0.14 | 0.08 | +0.06 |

## Trained for Machado + Quijote ch1

| run | model | Machado | epoch 0 | Δ | Quijote ch1 | epoch 0 | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| clean_combined_slide8_nmse_0p6b_rag | Qwen3-0.6B | 0.14 | 0.07 | +0.07 | 0.15 | 0.09 | +0.06 |
| clean_combined_slide8_vocab_0p6b_rag | Qwen3-0.6B | 0.14 | 0.07 | +0.08 | 0.15 | 0.09 | +0.06 |
| lw_m_combined_0p6b_rag | Qwen3-0.6B | 0.19 | 0.07 | +0.12 | 0.17 | 0.09 | +0.08 |
| lw_m_combined_s43_0p6b_rag | Qwen3-0.6B | 0.20 | 0.07 | +0.13 | 0.13 | 0.09 | +0.04 |
| clean_combined_slide8_vocab_1p7b_rag | Qwen3-1.7B | 0.20 | 0.11 | +0.10 | 0.18 | 0.14 | +0.04 |
| lw_m_combined_1p7b_rag | Qwen3-1.7B | 0.15 | 0.11 | +0.04 | 0.17 | 0.14 | +0.04 |
| clean_combined_slide8_vocab_14b_lora | Qwen3-14B | 0.16 | 0.13 | +0.03 | 0.13 | 0.14 | -0.01 |
| clean_combined_slide8_vocab_4b_lora | Qwen3-4B | 0.11 | 0.10 | +0.01 | 0.10 | 0.10 | +0.00 |
| clean_combined_slide8_vocab_8b_lora | Qwen3-8B | 0.12 | 0.14 | -0.02 | 0.16 | 0.13 | +0.03 |

## Model damage: fixed standard benchmark subsets

This is the capability check. Accuracy and Δ are paired means over the standard tasks available in both checkpoint and epoch-zero artifacts — the primary suite is ARC-Easy, ARC-Challenge, and HellaSwag (n=100 fixed subsets); legacy destruction.json fallbacks may add HellaSwag/MMLU/ARC-Challenge/WinoGrande/MMLU-Pro at n=200. Negative Δ means lost general knowledge/skill. The old custom prose-loss task is no longer part of checkpoint re-evaluation.

| run | model | paired/base tasks | accuracy | epoch 0 | Δ | worst loss |
|---|---|---:|---:|---:|---:|---|
| clean_machado_slide8_vocab_alia40b_lora | ALIA-40b-fc-2606 | 3/3 | 0.37 | 0.67 | -0.31 | arc_easy -0.41 |
| certify_anchor_readout | Qwen3-0.6B | 3/3 | 0.45 | 0.46 | -0.01 | hellaswag -0.02 |
| certify_censored_frozen | Qwen3-0.6B | 3/3 | 0.45 | 0.46 | -0.01 | arc_easy -0.04 |
| certify_lora_online | Qwen3-0.6B | 3/3 | 0.45 | 0.46 | -0.01 | arc_challenge -0.01 |
| certify_mixed_frozen | Qwen3-0.6B | 3/3 | 0.42 | 0.46 | -0.03 | arc_easy -0.06 |
| certify_offload_adam | Qwen3-0.6B | 3/3 | 0.44 | 0.46 | -0.02 | hellaswag -0.02 |
| certify_sequential_subset | Qwen3-0.6B | 3/3 | 0.44 | 0.46 | -0.02 | hellaswag -0.03 |
| certify_slide4_dedup | Qwen3-0.6B | 3/3 | 0.44 | 0.46 | -0.01 | arc_easy -0.03 |
| certify_slide4_item | Qwen3-0.6B | 3/3 | 0.44 | 0.46 | -0.02 | hellaswag -0.03 |
| certify_slide4_readout | Qwen3-0.6B | 3/3 | 0.44 | 0.46 | -0.02 | hellaswag -0.03 |
| certify_slide4_readout_padded | Qwen3-0.6B | 3/3 | 0.45 | 0.46 | -0.01 | hellaswag -0.02 |
| certify_summed_bucketed | Qwen3-0.6B | 3/3 | 0.45 | 0.46 | -0.00 | hellaswag -0.01 |
| certify_summed_item | Qwen3-0.6B | 3/3 | 0.44 | 0.46 | -0.02 | arc_easy -0.04 |
| certify_summed_padded | Qwen3-0.6B | 3/3 | 0.44 | 0.46 | -0.01 | arc_easy -0.03 |
| lw_b_mixed_0p6b_rag | Qwen3-0.6B | 3/3 | 0.40 | 0.46 | -0.06 | arc_easy -0.09 |
| lw_b_mixed_strict_0p6b_rag | Qwen3-0.6B | 3/3 | 0.39 | 0.46 | -0.06 | arc_easy -0.09 |
| lw_b_tc_frozen_0p6b_rag | Qwen3-0.6B | 3/3 | 0.40 | 0.46 | -0.06 | arc_easy -0.12 |
| lw_b_tc_lora_0p6b_rag | Qwen3-0.6B | 3/3 | 0.36 | 0.46 | -0.10 | arc_easy -0.14 |
| lw_i_cosine_0p6b_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.05 | arc_easy -0.13 |
| lw_i_huber_0p6b_rag | Qwen3-0.6B | 3/3 | 0.39 | 0.46 | -0.06 | arc_easy -0.17 |
| lw_i_l2mse_0p6b_rag | Qwen3-0.6B | 3/3 | 0.39 | 0.46 | -0.07 | arc_easy -0.13 |
| lw_i_lenskl_0p6b_rag | Qwen3-0.6B | 3/3 | 0.42 | 0.46 | -0.04 | arc_easy -0.08 |
| lw_i_lenskl_strict_0p6b_rag | Qwen3-0.6B | 3/3 | 0.42 | 0.46 | -0.03 | arc_easy -0.07 |
| lw_i_nmse_s43_0p6b_rag | Qwen3-0.6B | 3/3 | 0.39 | 0.46 | -0.07 | arc_easy -0.16 |
| lw_i_nmse_strict_0p6b_rag | Qwen3-0.6B | 3/3 | 0.40 | 0.46 | -0.06 | arc_easy -0.12 |
| lw_i_vocab_0p6b_rag | Qwen3-0.6B | 3/3 | 0.37 | 0.46 | -0.08 | arc_easy -0.16 |
| lw_i_vocab_strict_0p6b_rag | Qwen3-0.6B | 3/3 | 0.36 | 0.46 | -0.10 | arc_easy -0.13 |
| lw_j_vocab_v3_0p6b_rag | Qwen3-0.6B | 3/3 | 0.36 | 0.46 | -0.09 | arc_easy -0.20 |
| lw_k_anchor_0p6b_rag | Qwen3-0.6B | 3/3 | 0.38 | 0.46 | -0.08 | arc_easy -0.13 |
| lw_k_anchorkl_0p6b_rag | Qwen3-0.6B | 3/3 | 0.38 | 0.46 | -0.08 | arc_easy -0.12 |
| lw_k_final2p_0p6b_rag | Qwen3-0.6B | 3/3 | 0.37 | 0.46 | -0.09 | arc_easy -0.15 |
| lw_k_final_0p6b_rag | Qwen3-0.6B | 3/3 | 0.36 | 0.46 | -0.10 | arc_easy -0.18 |
| lw_k_final_k8_0p6b_rag | Qwen3-0.6B | 3/3 | 0.37 | 0.46 | -0.09 | arc_easy -0.16 |
| lw_k_final_s43_0p6b_rag | Qwen3-0.6B | 3/3 | 0.36 | 0.46 | -0.10 | arc_easy -0.17 |
| lw_k_maieutic_0p6b_rag | Qwen3-0.6B | 3/3 | 0.37 | 0.46 | -0.09 | arc_easy -0.19 |
| lw_k_tailonly_0p6b_rag | Qwen3-0.6B | 3/3 | 0.37 | 0.46 | -0.09 | arc_easy -0.18 |
| lw_k_tailonly_anchor_0p6b_rag | Qwen3-0.6B | 3/3 | 0.37 | 0.46 | -0.09 | arc_easy -0.20 |
| lw_k_tailonly_anchorkl_0p6b_rag | Qwen3-0.6B | 3/3 | 0.36 | 0.46 | -0.10 | arc_easy -0.19 |
| lw_k_v4strict_0p6b_rag | Qwen3-0.6B | 3/3 | 0.36 | 0.46 | -0.09 | arc_easy -0.18 |
| lw_k_vocab_s43_0p6b_rag | Qwen3-0.6B | 3/3 | 0.37 | 0.46 | -0.08 | arc_easy -0.14 |
| lw_lens_ce_0p6b_rag | Qwen3-0.6B | 3/3 | 0.35 | 0.46 | -0.11 | arc_easy -0.24 |
| lw_lens_ce_deep_0p6b_rag | Qwen3-0.6B | 3/3 | 0.38 | 0.46 | -0.07 | arc_easy -0.13 |
| lw_lens_ce_lora_0p6b_rag | Qwen3-0.6B | 3/3 | 0.26 | 0.46 | -0.20 | arc_easy -0.31 |
| lw_lens_kl_0p6b_rag | Qwen3-0.6B | 3/3 | 0.42 | 0.46 | -0.04 | hellaswag -0.05 |
| lw_lens_kl_deep_0p6b_rag | Qwen3-0.6B | 3/3 | 0.42 | 0.46 | -0.03 | arc_easy -0.09 |
| lw_m_anchordiv_0p6b_rag | Qwen3-0.6B | 3/3 | 0.38 | 0.46 | -0.08 | arc_easy -0.14 |
| lw_m_anchordiv_s43_0p6b_rag | Qwen3-0.6B | 3/3 | 0.37 | 0.46 | -0.09 | arc_easy -0.17 |
| lw_m_anchordiv_w1_0p6b_rag | Qwen3-0.6B | 3/3 | 0.36 | 0.46 | -0.09 | arc_easy -0.16 |
| lw_n_thinksel_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.02 | hellaswag -0.03 |
| lw_n_thinksel_s43_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.02 | arc_challenge -0.03 |
| lw_n_thinkselkl_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.02 | arc_challenge -0.03 |
| lw_n_thinkslide_0p6b_rag | Qwen3-0.6B | 3/3 | 0.44 | 0.46 | -0.02 | hellaswag -0.03 |
| lw_n_thinkwhole_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.02 | arc_easy -0.08 |
| lw_o_fisher_0p6b_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.05 | arc_easy -0.09 |
| lw_q_pp2_0p6b_rag | Qwen3-0.6B | 3/3 | 0.37 | 0.46 | -0.09 | arc_easy -0.16 |
| lw_q_pp2fix_0p6b_rag | Qwen3-0.6B | 3/3 | 0.37 | 0.46 | -0.09 | arc_easy -0.16 |
| lw_r_disj_pinned | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.03 | arc_easy -0.09 |
| lw_r_lensdeep2_0p6b_rag | Qwen3-0.6B | 3/3 | 0.38 | 0.46 | -0.08 | arc_easy -0.14 |
| lw_r_lensonly_0p6b_rag | Qwen3-0.6B | 3/3 | 0.29 | 0.46 | -0.17 | arc_easy -0.24 |
| lw_r_s43_pinned | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.02 | arc_easy -0.05 |
| lw_r_slide2_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.03 | arc_easy -0.06 |
| lw_r_slide4_0p6b_rag | Qwen3-0.6B | 3/3 | 0.39 | 0.46 | -0.07 | arc_easy -0.09 |
| lw_r_slide6pure_0p6b_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.04 | arc_challenge -0.05 |
| lw_r_slide8_0p6b_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.05 | arc_easy -0.13 |
| lw_r_slide8disj_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.03 | arc_easy -0.08 |
| lw_r_slide8kl_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.03 | arc_easy -0.09 |
| lw_r_slide8pure_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.02 | arc_easy -0.06 |
| lw_r_slide8pure_s43_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.03 | arc_easy -0.09 |
| lw_r_tailpure_0p6b_rag | Qwen3-0.6B | 3/3 | 0.38 | 0.46 | -0.08 | arc_easy -0.13 |
| lw_s_lenskluni_0p6b_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.05 | arc_easy -0.09 |
| lw_s_slide8lenskl_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.03 | arc_easy -0.08 |
| lw_s_slide8nmse_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.03 | arc_easy -0.11 |
| lw_s_tcmodern_0p6b_rag | Qwen3-0.6B | 3/3 | 0.38 | 0.46 | -0.08 | arc_easy -0.16 |
| lw_seq_0p6b_rag | Qwen3-0.6B | 3/3 | 0.45 | 0.46 | -0.00 | arc_easy -0.06 |
| lw_seq_bf16_0p6b_rag | Qwen3-0.6B | 3/3 | 0.44 | 0.46 | -0.01 | arc_easy -0.07 |
| lw_summed_0p6b_rag | Qwen3-0.6B | 3/3 | 0.42 | 0.46 | -0.04 | hellaswag -0.07 |
| lw_summed_ce_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.02 | arc_easy -0.06 |
| lw_summed_ce_e40_0p6b_rag | Qwen3-0.6B | 3/3 | 0.42 | 0.46 | -0.03 | arc_easy -0.09 |
| lw_summed_e40_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.03 | arc_easy -0.07 |
| lw_summed_l2mse_e40_0p6b_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.05 | arc_challenge -0.05 |
| lw_t_ragchannel_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.02 | arc_easy -0.05 |
| lw_tail_ce_0p6b_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.05 | arc_easy -0.10 |
| lw_tail_ce_e40_0p6b_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.05 | arc_easy -0.11 |
| lw_tail_ce_e40_s2_0p6b_rag | Qwen3-0.6B | 3/3 | 0.42 | 0.46 | -0.04 | arc_easy -0.11 |
| lw_tail_ce_e40_v2_0p6b_rag | Qwen3-0.6B | 3/3 | 0.38 | 0.46 | -0.08 | arc_easy -0.16 |
| lw_tail_ce_k2_e40_0p6b_rag | Qwen3-0.6B | 3/3 | 0.42 | 0.46 | -0.04 | arc_easy -0.10 |
| lw_tail_ce_lora_0p6b_rag | Qwen3-0.6B | 3/3 | 0.38 | 0.46 | -0.08 | arc_easy -0.13 |
| lw_tc_lora_e40_0p6b_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.04 | hellaswag -0.06 |
| pp2diag_lora_single_0p6b | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.05 | arc_easy -0.10 |
| pp2diag_noanchor_single_0p6b | Qwen3-0.6B | 3/3 | 0.42 | 0.46 | -0.04 | arc_easy -0.10 |
| tl_i_tunedlenskl_strict_0p6b_rag | Qwen3-0.6B | 3/3 | 0.44 | 0.46 | -0.01 | arc_easy -0.04 |
| dev | Qwen3-1.7B | 3/3 | 0.53 | 0.52 | +0.00 | arc_challenge -0.03 |
| lw_j_l2mse_k4_1p7b_rag | Qwen3-1.7B | 3/3 | 0.52 | 0.52 | -0.01 | arc_easy -0.06 |
| lw_j_vocab_k2_1p7b_rag | Qwen3-1.7B | 3/3 | 0.48 | 0.52 | -0.05 | arc_easy -0.09 |
| lw_j_vocab_k4_1p7b_rag | Qwen3-1.7B | 3/3 | 0.50 | 0.52 | -0.02 | arc_challenge -0.06 |
| lw_j_vocab_k8_1p7b_rag | Qwen3-1.7B | 3/3 | 0.50 | 0.52 | -0.02 | arc_easy -0.07 |
| lw_k_final_1p7b_rag | Qwen3-1.7B | 3/3 | 0.48 | 0.52 | -0.04 | arc_challenge -0.08 |
| lw_m_anchordiv_1p7b_rag | Qwen3-1.7B | 3/3 | 0.51 | 0.52 | -0.01 | arc_challenge -0.04 |
| lw_n_thinksel_1p7b_rag | Qwen3-1.7B | 3/3 | 0.53 | 0.52 | +0.00 | arc_challenge -0.01 |
| lw_r_slide8pure_1p7b_rag | Qwen3-1.7B | 3/3 | 0.51 | 0.52 | -0.02 | arc_easy -0.04 |
| lw_seq_1p7b_rag | Qwen3-1.7B | 3/3 | 0.53 | 0.52 | +0.01 | arc_challenge -0.03 |
| xs_fisher_1p7b_rag | Qwen3-1.7B | 3/3 | 0.44 | 0.52 | -0.08 | arc_easy -0.18 |
| xs_slide2_1p7b_rag | Qwen3-1.7B | 3/3 | 0.51 | 0.52 | -0.01 | arc_easy -0.05 |
| xs_slide4_1p7b_rag | Qwen3-1.7B | 3/3 | 0.51 | 0.52 | -0.01 | arc_challenge -0.04 |
| clean_machado_slide8_vocab_14b_lora | Qwen3-14B | 3/3 | 0.66 | 0.70 | -0.04 | hellaswag -0.05 |
| lw_l_final_14b_rag | Qwen3-14B | 3/3 | 0.63 | 0.70 | -0.07 | hellaswag -0.11 |
| lw_tc_lora_14b_rag | Qwen3-14B | 3/3 | 0.66 | 0.70 | -0.04 | arc_challenge -0.05 |
| clean_machado_slide8_vocab_4b_lora | Qwen3-4B | 3/3 | 0.56 | 0.58 | -0.02 | arc_easy -0.05 |
| lw_j_qwen4b_rag | Qwen3-4B | 3/3 | 0.55 | 0.58 | -0.02 | hellaswag -0.05 |
| lw_l_final_4b_av2_rag | Qwen3-4B | 3/3 | 0.56 | 0.58 | -0.01 | arc_easy -0.04 |
| lw_l_final_4b_av2lr_rag | Qwen3-4B | 3/3 | 0.55 | 0.58 | -0.02 | arc_challenge -0.04 |
| lw_l_final_4b_r64_rag | Qwen3-4B | 3/3 | 0.53 | 0.58 | -0.05 | arc_challenge -0.08 |
| lw_l_final_4b_rag | Qwen3-4B | 3/3 | 0.55 | 0.58 | -0.03 | arc_challenge -0.06 |
| lw_l_tailonly_4b_rag | Qwen3-4B | 3/3 | 0.58 | 0.58 | +0.00 | arc_easy -0.01 |
| lw_p_ft4b_rag | Qwen3-4B | 3/3 | 0.53 | 0.58 | -0.05 | arc_easy -0.07 |
| clean_machado_slide8_vocab_8b_lora | Qwen3-8B | 3/3 | 0.60 | 0.63 | -0.03 | hellaswag -0.09 |
| lw_j_qwen8b_rag | Qwen3-8B | 3/3 | 0.57 | 0.63 | -0.06 | hellaswag -0.16 |
| lw_l_final_8b_av2_rag | Qwen3-8B | 3/3 | 0.61 | 0.63 | -0.02 | hellaswag -0.06 |
| lw_l_final_8b_rag | Qwen3-8B | 3/3 | 0.59 | 0.63 | -0.04 | hellaswag -0.10 |
| lw_j_llama8b_rag | Llama-3.1-8B-Instruct | 3/3 | 0.55 | 0.65 | -0.10 | arc_challenge -0.15 |
| lw_j_phi4mini_rag | Phi-4-mini-reasoning | 3/3 | 0.53 | 0.51 | +0.02 | hellaswag -0.02 |
| lw_j_mistral7b_rag | Mistral-7B-Instruct-v0.1 | 0/3 | — | 0.60 | — | — |
| lw_k_gptoss_rag | gpt-oss-20b | 3/3 | 0.42 | 0.56 | -0.14 | arc_easy -0.22 |
| clean_combined_slide8_nmse_0p6b_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.05 | arc_easy -0.11 |
| clean_combined_slide8_vocab_0p6b_rag | Qwen3-0.6B | 3/3 | 0.40 | 0.46 | -0.06 | arc_easy -0.11 |
| lw_m_combined_0p6b_rag | Qwen3-0.6B | 3/3 | 0.38 | 0.46 | -0.08 | arc_easy -0.16 |
| lw_m_combined_s43_0p6b_rag | Qwen3-0.6B | 3/3 | 0.38 | 0.46 | -0.08 | arc_easy -0.16 |
| clean_combined_slide8_vocab_1p7b_rag | Qwen3-1.7B | 3/3 | 0.51 | 0.52 | -0.02 | arc_easy -0.05 |
| lw_m_combined_1p7b_rag | Qwen3-1.7B | 3/3 | 0.51 | 0.52 | -0.01 | arc_challenge -0.04 |
| clean_combined_slide8_vocab_14b_lora | Qwen3-14B | 3/3 | 0.66 | 0.70 | -0.04 | hellaswag -0.05 |
| clean_combined_slide8_vocab_4b_lora | Qwen3-4B | 3/3 | 0.54 | 0.58 | -0.04 | arc_easy -0.08 |
| clean_combined_slide8_vocab_8b_lora | Qwen3-8B | 3/3 | 0.60 | 0.63 | -0.03 | hellaswag -0.07 |
| clean_q_ch1_slide4_vocab_0p6b_e320_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.04 | arc_easy -0.08 |
| clean_q_ch1_slide4_vocab_0p6b_rag | Qwen3-0.6B | 3/3 | 0.42 | 0.46 | -0.04 | hellaswag -0.07 |
| clean_q_ch1_slide8_nmse_0p6b_e320_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.05 | hellaswag -0.08 |
| clean_q_ch1_slide8_nmse_0p6b_rag | Qwen3-0.6B | 3/3 | 0.42 | 0.46 | -0.04 | arc_easy -0.08 |
| clean_q_ch1_slide8_vocab_0p6b_e320_rag | Qwen3-0.6B | 3/3 | 0.45 | 0.46 | -0.01 | hellaswag -0.05 |
| clean_q_ch1_slide8_vocab_0p6b_rag | Qwen3-0.6B | 3/3 | 0.42 | 0.46 | -0.04 | hellaswag -0.05 |
| q_ch1_0p6b_rag | Qwen3-0.6B | 3/3 | 0.44 | 0.46 | -0.02 | arc_easy -0.05 |
| q_ch1_ext_0p6b_rag | Qwen3-0.6B | 3/3 | 0.42 | 0.46 | -0.03 | arc_easy -0.07 |
| clean_q_ch1_slide2_vocab_1p7b_rag | Qwen3-1.7B | 3/3 | 0.53 | 0.52 | +0.00 | arc_challenge -0.02 |
| clean_q_ch1_slide4_vocab_1p7b_rag | Qwen3-1.7B | 3/3 | 0.54 | 0.52 | +0.02 | arc_challenge +0.00 |
| clean_q_ch1_slide8_vocab_1p7b_e320_rag | Qwen3-1.7B | 3/3 | 0.51 | 0.52 | -0.02 | hellaswag -0.03 |
| clean_q_ch1_slide8_vocab_1p7b_rag | Qwen3-1.7B | 3/3 | 0.51 | 0.52 | -0.01 | hellaswag -0.02 |
| clean_q_ch1_slide2_vocab_14b_lora | Qwen3-14B | 3/3 | 0.67 | 0.70 | -0.03 | arc_challenge -0.04 |
| clean_q_ch1_slide4_vocab_14b_lora | Qwen3-14B | 3/3 | 0.66 | 0.70 | -0.03 | hellaswag -0.05 |
| clean_q_ch1_slide8_lenskl_14b_lora | Qwen3-14B | 3/3 | 0.65 | 0.70 | -0.05 | hellaswag -0.07 |
| clean_q_ch1_slide8_nmse_14b_lora | Qwen3-14B | 3/3 | 0.57 | 0.70 | -0.12 | arc_easy -0.15 |
| clean_q_ch1_slide8_vocab_14b_lora | Qwen3-14B | 3/3 | 0.68 | 0.70 | -0.02 | arc_challenge -0.02 |
| clean_q_ch1_slide8_vocab_4b_lora | Qwen3-4B | 3/3 | 0.57 | 0.58 | -0.01 | hellaswag -0.03 |
| clean_q_ch1_slide8_vocab_8b_lora | Qwen3-8B | 3/3 | 0.62 | 0.63 | -0.01 | hellaswag -0.03 |
| q_ch16_0p6b_rag | Qwen3-0.6B | 3/3 | 0.40 | 0.46 | -0.05 | arc_easy -0.07 |
| q_ch16_av2_0p6b_rag | Qwen3-0.6B | 3/3 | 0.40 | 0.46 | -0.06 | arc_easy -0.07 |
| q_ch16_ext_0p6b_rag | Qwen3-0.6B | 3/3 | 0.39 | 0.46 | -0.06 | arc_easy -0.08 |
| q_ch4_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.03 | arc_easy -0.07 |
| q_ch4_av2_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.02 | arc_easy -0.05 |
| q_ch4_av2_s43_0p6b_rag | Qwen3-0.6B | 3/3 | 0.45 | 0.46 | -0.01 | hellaswag -0.03 |
| q_ch8_0p6b_rag | Qwen3-0.6B | 3/3 | 0.41 | 0.46 | -0.04 | arc_easy -0.12 |
| q_ch8_av2_0p6b_rag | Qwen3-0.6B | 3/3 | 0.40 | 0.46 | -0.05 | arc_easy -0.08 |
| q_ch8_slide8pure_0p6b_rag | Qwen3-0.6B | 3/3 | 0.43 | 0.46 | -0.03 | arc_easy -0.06 |

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
