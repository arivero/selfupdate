#!/usr/bin/env bash
# 122B PPP8 cross-node EVAL-IN (agpuh01 GPU0-3 + agpuh02 GPU0-3).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export SELFUPDATE_V4_STAGE_HOSTS="local local local local agpuh02 agpuh02 agpuh02 agpuh02"
exec defactorised/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_qwen35_122b_v4_full.yaml \
  configs/experiments/h100_smoke/qwen35_122b_v4_ppp8x_evalin.yaml
