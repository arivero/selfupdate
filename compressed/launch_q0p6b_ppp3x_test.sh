#!/usr/bin/env bash
# Qwen3-0.6B PPP3 cross-node relay-drain test.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export SELFUPDATE_V4_STAGE_HOSTS="local local agpuh02"
exec compressed/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml \
  configs/experiments/h100_smoke/qwen3_0p6b_v4_ppp3_xnode.yaml
