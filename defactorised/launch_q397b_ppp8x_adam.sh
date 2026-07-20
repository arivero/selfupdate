#!/usr/bin/env bash
# 397B PPP8 cross-node launch, ADAM variant (agpuh01 GPUs 0-3 + agpuh02
# GPUs 0-3), from agpuh01. Mirrors launch_q397b_ppp8x.sh but points at the
# adam experiment config; store+scoped+ROTATE (moments rotate with the block).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
unset SELFUPDATE_V4_RELAY_ROOT
export SELFUPDATE_V4_STAGE_HOSTS="local local local local agpuh02 agpuh02 agpuh02 agpuh02"
exec defactorised/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_qwen35_397b_v4_full.yaml \
  configs/experiments/h100_smoke/qwen35_397b_v4_ppp8_xnode_adam.yaml
