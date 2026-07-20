#!/usr/bin/env bash
# Qwen3-0.6B PPP2 cross-node NCCL/IB relay test.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export SELFUPDATE_V4_STAGE_HOSTS="local agpuh02"
exec compressed/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml \
  configs/experiments/h100_smoke/qwen3_0p6b_v4_ppp2_xnode.yaml
