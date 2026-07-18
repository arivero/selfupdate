#!/usr/bin/env bash
# DeepSeek-V4-Flash answer generation on the BF16 dequant snapshot (TP4).
# The fp8 hub snapshot failed vLLM on the fp4-expert Marlin PTX (driver
# 565); the bf16 snapshot has NO fp4/fp8 tensors, so that kernel path is
# never taken. Same dequant that makes DeepSeek trainable makes it
# generatable — one artifact, both gates (2026-07-18).
set -u
cd /fs/agustina/arivero/supercomplex/selfup_teacher || exit 1
export SSL_CERT_FILE=/fs/agustina/arivero/supercomplex/.local/lib/python3.11/site-packages/certifi/cacert.pem
export HF_HUB_OFFLINE=1 TQDM_DISABLE=1
export VLLM_CACHE_ROOT=/tmp/$USER/selfupdate-vllm-cache
export TRITON_CACHE_DIR=/tmp/$USER/selfupdate-vllm-triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/selfupdate-vllm-inductor
exec /fs/agustina/arivero/supercomplex/venvs/vllm025/bin/python \
  scripts/benchmark_vllm_generation.py \
  --config configs/experiments/h100_smoke/base_deepseek_v4_flash_v4_full.yaml \
  --experiment configs/experiments/h100_smoke/base_deepseek_v4_flash_v4_full.yaml \
  --model /fs/agustina/arivero/supercomplex/snapshots/deepseek-v4-flash-bf16 \
  --examples data/combined/examples_v5rs_window.jsonl \
  --batch-sizes 256 --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 --max-model-len 6144 \
  --prompt-format native --generation-extra-tokens 96 \
  --generation-max-tokens 4096 \
  --out runs/vllm_h100/deepseek_v4_flash
