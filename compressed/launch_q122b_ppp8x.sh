#!/usr/bin/env bash
# Qwen3.5-122B PPP8 cross-node launch from agpuh01.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
unset SELFUPDATE_V4_RELAY_ROOT
export SELFUPDATE_V4_STAGE_HOSTS="local local local local agpuh02 agpuh02 agpuh02 agpuh02"
exec compressed/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_qwen35_122b_v4_full.yaml \
  configs/experiments/h100_smoke/qwen35_122b_v4_ppp8_xnode.yaml
