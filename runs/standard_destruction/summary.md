# Standard Destruction Summary

Primary metric: WikiText-2 validation perplexity ratio vs epoch-zero teacher.

| run | model | PPL | teacher PPL | ratio | CE delta | tokens |
|---|---:|---:|---:|---:|---:|---:|
| clean_q_ch1_slide8_nmse_0p6b_e320_rag | Qwen3-0.6B | 41.62 | 18.96 | 2.19 | +0.786 | 57344 |
| clean_combined_slide8_nmse_0p6b_rag | Qwen3-0.6B | 43.41 | 18.96 | 2.29 | +0.828 | 57344 |
| clean_q_ch1_slide8_vocab_1p7b_e320_rag | Qwen3-1.7B | 20.43 | 14.89 | 1.37 | +0.317 | 57344 |
| clean_combined_slide8_vocab_1p7b_rag | Qwen3-1.7B | 23.50 | 14.89 | 1.58 | +0.456 | 57344 |
| clean_machado_slide8_vocab_14b_lora | Qwen3-14B | 10.83 | 7.93 | 1.37 | +0.311 | 57344 |
| clean_combined_slide8_vocab_14b_lora | Qwen3-14B | 13.27 | 7.93 | 1.67 | +0.514 | 57344 |
| clean_machado_slide8_vocab_4b_lora | Qwen3-4B | 21.06 | 12.40 | 1.70 | +0.530 | 57344 |
| clean_q_ch1_slide8_vocab_4b_lora | Qwen3-4B | 21.59 | 12.40 | 1.74 | +0.555 | 57344 |
| clean_combined_slide8_vocab_4b_lora | Qwen3-4B | 21.90 | 12.40 | 1.77 | +0.569 | 57344 |
| clean_machado_slide8_vocab_8b_lora | Qwen3-8B | 12.91 | 9.01 | 1.43 | +0.359 | 57344 |
| clean_q_ch1_slide8_vocab_8b_lora | Qwen3-8B | 14.07 | 9.01 | 1.56 | +0.446 | 57344 |
| clean_combined_slide8_vocab_8b_lora | Qwen3-8B | 14.53 | 9.01 | 1.61 | +0.478 | 57344 |
