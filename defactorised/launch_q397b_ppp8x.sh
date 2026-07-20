#!/usr/bin/env bash
# 397B PPP8 cross-node launch (agpuh01 GPUs 0-3 + agpuh02 GPUs 0-3), from
# agpuh01. bf16 dequant snapshot on shared Lustre (both nodes read it),
# store+scoped+rotate (100GB/stage owned > 80GB card -> rotation). 3-epoch.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
unset SELFUPDATE_V4_RELAY_ROOT
export SELFUPDATE_V4_STAGE_HOSTS="local local local local agpuh02 agpuh02 agpuh02 agpuh02"
exec defactorised/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_qwen35_397b_v4_full.yaml \
  configs/experiments/h100_smoke/qwen35_397b_v4_ppp8_xnode.yaml
