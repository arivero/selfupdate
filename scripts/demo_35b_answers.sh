#!/usr/bin/env bash
# 35B-A3B vLLM answer generation on ONE H100 (TP1, GPU1) — the front gate
# for its speed-table row. Same params as the other envelope demos.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export SSL_CERT_FILE=/fs/agustina/arivero/supercomplex/.local/lib/python3.11/site-packages/certifi/cacert.pem
export HF_HUB_OFFLINE=1 TQDM_DISABLE=1 HF_HOME=/dev/shm/arivero/selfupdate-hf-cache
export CUDA_VISIBLE_DEVICES=1
export VLLM_CACHE_ROOT=/tmp/$USER/selfupdate-vllm-cache
export TRITON_CACHE_DIR=/tmp/$USER/selfupdate-vllm-triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/selfupdate-vllm-inductor
exec /fs/agustina/arivero/supercomplex/venvs/vllm025/bin/python \
  scripts/benchmark_vllm_generation.py \
  --config configs/experiments/h100_smoke/base_qwen36_35b_v4_full.yaml \
  --experiment configs/experiments/h100_smoke/qwen36_35b_v4_ppp1_rotate.yaml \
  --model Qwen/Qwen3.6-35B-A3B \
  --examples data/combined/examples_v5rs_window.jsonl \
  --batch-sizes 256 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.88 --max-model-len 6144 \
  --prompt-format native --generation-extra-tokens 96 \
  --generation-max-tokens 4096 \
  --out runs/vllm_h100/qwen36_35b_a3b
