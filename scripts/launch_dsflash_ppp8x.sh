#!/usr/bin/env bash
# DeepSeek-V4-Flash PPP8 cross-node launch (agpuh01 GPUs 0-3 + agpuh02 GPUs
# 0-3), from agpuh01. bf16 dequant snapshot on shared Lustre; answers are the
# re-tokenized 122B set (runs/vllm_h100/deepseek_v4_flash/PROVENANCE.md);
# chat template installed locally (snapshot CHAT_TEMPLATE_PROVENANCE.md).
# 43 blocks -> splits [5,11,16,22,27,32,38]; residency auto (~40 GB owned ->
# resident). 3-epoch SPEED test.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
unset SELFUPDATE_V4_RELAY_ROOT
export SELFUPDATE_V4_STAGE_HOSTS="local local local local agpuh02 agpuh02 agpuh02 agpuh02"
exec scripts/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_deepseek_v4_flash_v4_full.yaml \
  configs/experiments/h100_smoke/deepseek_v4_flash_ppp8_xnode.yaml
