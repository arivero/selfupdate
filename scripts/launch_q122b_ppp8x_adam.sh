#!/usr/bin/env bash
# 122B PPP8 cross-node launch, ADAM variant (agpuh01 GPUs 0-3 + agpuh02
# GPUs 0-3). Run FROM agpuh01. Mirrors launch_q122b_ppp8x.sh but points at
# the adam experiment config; fills the Adam-vs-SGD cell at N=8.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
unset SELFUPDATE_V4_RELAY_ROOT
export SELFUPDATE_V4_STAGE_HOSTS="local local local local agpuh02 agpuh02 agpuh02 agpuh02"
exec scripts/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_qwen35_122b_v4_full.yaml \
  configs/experiments/h100_smoke/qwen35_122b_v4_ppp8_xnode_adam.yaml
