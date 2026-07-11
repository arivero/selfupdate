# 1.7B Loss-Grid Live Report

Generated 2026-07-11T22:03:16+00:00.
Recall columns are deliberately corpus-separated. A `fast epoch probe` uses the fixed in-training subset; `full checkpoint eval` is the post-training evaluation. Standard deltas are paired within their stated source: fast epoch-0 subset or full pinned Qwen3-1.7B reference on ARC-Easy, ARC-Challenge, and HellaSwag.

Deliberately unqueued: `lens_js` slide1/slide2 configs exist but were never run — a bounded symmetric-divergence control, not a sweep candidate (issues.md low-priority item 13); absence is by design, not a missing artifact.

| run | loss | slide | status | items | epoch | source | epoch-0 M/Q1/Q4 | final M/Q1/Q4 | e0 mean | final mean | e0 standard | standard Δ | worst Δ |
|---|---|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| a_lossgrid_1p7b_combined_slide1_jacobian_lens_kl | jacobian_lens_kl | 1 | complete | 12312 | 24 | full checkpoint eval | 0.106/0.157/0.205 | 0.246/0.233/0.142 | 0.156 | 0.207 | 0.562 | -0.007 | -0.060 |
| a_lossgrid_1p7b_combined_slide1_jacobian_lens_kl_anthropic | jacobian_lens_kl | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.246/0.183/0.155 | 0.132 | 0.194 | 0.562 | -0.027 | -0.090 |
| a_lossgrid_1p7b_combined_slide2_huber | huber | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.244/0.177/0.150 | 0.132 | 0.190 | 0.562 | -0.007 | -0.030 |
| a_lossgrid_1p7b_combined_slide1_lenskl | lens_kl | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.217/0.218/0.114 | 0.132 | 0.183 | 0.562 | -0.003 | -0.040 |
| a_lossgrid_1p7b_combined_slide2_delta_nmse | delta_nmse | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.191/0.177/0.174 | 0.132 | 0.180 | 0.562 | -0.033 | -0.040 |
| a_lossgrid_1p7b_combined_slide2_lenskl | lens_kl | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.193/0.221/0.127 | 0.132 | 0.180 | 0.562 | 0.017 | -0.050 |
| a_lossgrid_1p7b_combined_slide2_nmse | nmse | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.210/0.178/0.148 | 0.132 | 0.179 | 0.562 | 0.000 | -0.010 |
| a_lossgrid_1p7b_combined_slide2_anchor_trajectory | nmse | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.201/0.196/0.138 | 0.132 | 0.178 | 0.562 | 0.000 | -0.010 |
| a_lossgrid_1p7b_combined_slide2_jacobian_lens_kl | jacobian_lens_kl | 2 | complete | 12312 | 24 | full checkpoint eval | 0.106/0.157/0.205 | 0.182/0.186/0.160 | 0.156 | 0.176 | 0.562 | 0.003 | -0.020 |
| a_lossgrid_1p7b_combined_slide2_jacobian_lens_kl_anthropic | jacobian_lens_kl | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.206/0.164/0.153 | 0.132 | 0.174 | 0.562 | 0.000 | -0.040 |
| a_lossgrid_1p7b_combined_slide2_cosine | cosine | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.222/0.158/0.137 | 0.132 | 0.172 | 0.562 | 0.003 | -0.040 |
| a_lossgrid_1p7b_combined_slide2_delta_vocab_cos | delta_vocab_cos | 2 | complete | 12312 | 24 | full checkpoint eval | 0.106/0.157/0.205 | 0.208/0.174/0.132 | 0.156 | 0.172 | 0.562 | -0.027 | -0.090 |
| a_lossgrid_1p7b_combined_slide2_mahalanobis | mahalanobis | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.195/0.162/0.145 | 0.132 | 0.167 | 0.562 | -0.007 | -0.020 |
| a_lossgrid_1p7b_combined_slide2_relational_state | relational_state | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.206/0.164/0.132 | 0.132 | 0.167 | 0.562 | -0.007 | -0.040 |
| a_lossgrid_1p7b_combined_slide2_vocabfisher | vocab_fisher | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.142/0.187/0.162 | 0.132 | 0.164 | 0.562 | -0.007 | -0.060 |
| a_lossgrid_1p7b_combined_slide2_l2mse | l2mse | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.153/0.168/0.142 | 0.132 | 0.155 | 0.562 | 0.020 | 0.000 |
| a_lossgrid_1p7b_combined_slide1_huber | huber | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.166/0.160/0.133 | 0.132 | 0.153 | 0.562 | -0.033 | -0.040 |
| a_lossgrid_1p7b_combined_slide2_flow | flow_nmse | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.162/0.142/0.152 | 0.132 | 0.152 | 0.562 | -0.043 | -0.080 |
| a_lossgrid_1p7b_combined_slide1_nmse | nmse | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.156/0.176/0.123 | 0.132 | 0.152 | 0.562 | -0.020 | -0.030 |
| a_lossgrid_1p7b_combined_slide1_vocabfisher | vocab_fisher | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.151/0.158/0.145 | 0.132 | 0.151 | 0.562 | -0.050 | -0.150 |
| a_lossgrid_1p7b_combined_slide2_vocab | vocab_mse | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/— | 0.156/0.158/0.139 | 0.115 | 0.151 | 0.562 | 0.017 | -0.020 |
| a_lossgrid_1p7b_combined_slide1_cosine | cosine | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.137/0.172/0.144 | 0.132 | 0.151 | 0.562 | -0.037 | -0.040 |
| a_lossgrid_1p7b_combined_slide2_delta_cosine | delta_cosine | 2 | complete | 12312 | 24 | full checkpoint eval | 0.106/0.157/0.205 | 0.141/0.166/0.141 | 0.156 | 0.150 | 0.562 | -0.090 | -0.280 |
| a_lossgrid_1p7b_combined_slide1_delta_cosine | delta_cosine | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.154/0.152/0.143 | 0.132 | 0.150 | 0.562 | -0.013 | -0.020 |
| a_lossgrid_1p7b_combined_slide1_jacobian_nmse | jacobian_nmse | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.149/0.151/0.148 | 0.132 | 0.149 | 0.562 | -0.013 | -0.040 |
| a_lossgrid_1p7b_combined_slide2_jacobian_cosine_anthropic | jacobian_cosine | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.156/0.145/0.146 | 0.132 | 0.149 | 0.562 | 0.013 | 0.000 |
| a_lossgrid_1p7b_combined_slide2_embedding | embedding_mse | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.144/0.157/0.138 | 0.132 | 0.146 | 0.562 | 0.010 | -0.020 |
| a_lossgrid_1p7b_combined_slide2_jacobian_nmse_anthropic | jacobian_nmse | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.153/0.144/0.138 | 0.132 | 0.145 | 0.562 | 0.010 | -0.020 |
| a_lossgrid_1p7b_combined_slide1_delta_nmse | delta_nmse | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.143/0.137/0.153 | 0.132 | 0.145 | 0.562 | -0.007 | -0.020 |
| a_lossgrid_1p7b_combined_slide1_jacobian_cosine_anthropic | jacobian_cosine | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.150/0.149/0.129 | 0.132 | 0.142 | 0.562 | -0.027 | -0.080 |
| a_lossgrid_1p7b_combined_slide2_robust | state_delta_charbonnier | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.133/0.162/0.129 | 0.132 | 0.142 | 0.562 | 0.003 | -0.020 |
| a_lossgrid_1p7b_combined_slide1_jacobian_cosine | jacobian_cosine | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.130/0.170/0.122 | 0.132 | 0.141 | 0.562 | -0.020 | -0.060 |
| a_lossgrid_1p7b_combined_slide1_jacobian_nmse_anthropic | jacobian_nmse | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.133/0.156/0.128 | 0.132 | 0.139 | 0.562 | -0.010 | -0.050 |
| a_lossgrid_1p7b_combined_slide2_jacobian_vocab_mse_anthropic | jacobian_vocab_mse | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.137/0.152/0.122 | 0.132 | 0.137 | 0.562 | 0.013 | -0.030 |
| a_lossgrid_1p7b_combined_slide1_delta_vocab_cos | delta_vocab_cos | 1 | complete | 12312 | 24 | full checkpoint eval | 0.106/0.157/0.205 | 0.156/0.136/0.116 | 0.156 | 0.136 | 0.562 | -0.020 | -0.050 |
| a_lossgrid_1p7b_combined_slide2_contrastive | contrastive | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.147/0.134/0.126 | 0.132 | 0.135 | 0.562 | -0.230 | -0.370 |
| a_lossgrid_1p7b_combined_slide2_jacobian_vocab_mse | jacobian_vocab_mse | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.136/0.154/0.116 | 0.132 | 0.135 | 0.562 | 0.010 | -0.030 |
| a_lossgrid_1p7b_combined_slide2_jacobian_nmse | jacobian_nmse | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.139/0.146/0.121 | 0.132 | 0.135 | 0.562 | 0.007 | -0.020 |
| a_lossgrid_1p7b_combined_slide1_jacobian_vocab_mse | jacobian_vocab_mse | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.132/0.135/0.129 | 0.132 | 0.132 | 0.562 | -0.027 | -0.070 |
| a_lossgrid_1p7b_combined_slide1_l2mse | l2mse | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.113/0.141/0.132 | 0.132 | 0.129 | 0.562 | -0.007 | -0.030 |
| a_lossgrid_1p7b_combined_slide1_vocab | vocab_mse | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/— | 0.125/0.129/0.127 | 0.115 | 0.127 | 0.562 | 0.000 | -0.040 |
| a_lossgrid_1p7b_combined_slide1_jacobian_vocab_mse_anthropic | jacobian_vocab_mse | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.111/0.148/0.118 | 0.132 | 0.126 | 0.562 | -0.037 | -0.090 |
| a_lossgrid_1p7b_combined_slide2_jacobian_cosine | jacobian_cosine | 2 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.119/0.130/0.118 | 0.132 | 0.122 | 0.562 | 0.010 | -0.010 |
| a_lossgrid_1p7b_combined_slide1_component | component_nmse | 1 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.122/0.129/0.107 | 0.132 | 0.119 | 0.562 | -0.017 | -0.070 |
| a_lossgrid_1p7b_combined_slide4_multidelta | multi_delta_nmse | 4 | complete | 12312 | 24 | full checkpoint eval | 0.081/0.150/0.166 | 0.135/0.117/0.087 | 0.132 | 0.113 | 0.562 | -0.007 | -0.010 |
