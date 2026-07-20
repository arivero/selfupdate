#!/usr/bin/env bash
# One-shot DeepSeek-V4-Flash vLLM demo retry (chatfmt fallback in 4958e67).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export SSL_CERT_FILE=/fs/agustina/arivero/supercomplex/.local/lib/python3.11/site-packages/certifi/cacert.pem
export HF_HUB_OFFLINE=1 TQDM_DISABLE=1
export VLLM_CACHE_ROOT=/tmp/$USER/selfupdate-vllm-cache
export TRITON_CACHE_DIR=/tmp/$USER/selfupdate-vllm-triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/selfupdate-vllm-inductor
exec /fs/agustina/arivero/supercomplex/venvs/vllm025/bin/python \
  defactorised/benchmark_vllm_generation.py \
  --config configs/experiments/h100_smoke/base_qwen36_27b_v4_full.yaml \
  --experiment configs/experiments/h100_smoke/qwen36_27b_v4_ppp4_einf.yaml \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --examples data/combined/examples_v5rs_window.jsonl \
  --batch-sizes 256 --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 --moe-backend triton \
  --gpu-memory-utilization 0.85 --max-model-len 6144 \
  --prompt-format native --generation-extra-tokens 96 \
  --generation-max-tokens 4096 --reasoning-effort low \
  --out runs/vllm_h100/deepseek_v4_flash
