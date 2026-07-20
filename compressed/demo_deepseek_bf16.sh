#!/usr/bin/env bash
# DeepSeek-V4-Flash answer generation on the BF16 snapshot (TP4).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export SSL_CERT_FILE=/fs/agustina/arivero/supercomplex/.local/lib/python3.11/site-packages/certifi/cacert.pem
export HF_HUB_OFFLINE=1 TQDM_DISABLE=1
export VLLM_CACHE_ROOT=/tmp/$USER/selfupdate-vllm-cache
export TRITON_CACHE_DIR=/tmp/$USER/selfupdate-vllm-triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/selfupdate-vllm-inductor
exec /fs/agustina/arivero/supercomplex/venvs/vllm025/bin/python \
  compressed/benchmark_vllm_generation.py \
  --config configs/experiments/h100_smoke/base_deepseek_v4_flash_v4_full.yaml \
  --experiment configs/experiments/h100_smoke/base_deepseek_v4_flash_v4_full.yaml \
  --model /fs/agustina/arivero/supercomplex/snapshots/deepseek-v4-flash-bf16 \
  --examples data/combined/examples_v5rs_window.jsonl \
  --batch-sizes 256 --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 --max-model-len 6144 \
  --prompt-format native --generation-extra-tokens 96 \
  --generation-max-tokens 4096 \
  --out runs/vllm_h100/deepseek_v4_flash
