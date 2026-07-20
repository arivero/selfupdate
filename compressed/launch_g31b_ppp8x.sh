#!/usr/bin/env bash
# Gemma-31B PPP8 cross-node launch from agpuh01.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export SELFUPDATE_V4_STAGE_HOSTS="local local local local agpuh02 agpuh02 agpuh02 agpuh02"
exec compressed/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_gemma4_31b_v4_full.yaml \
  configs/experiments/h100_smoke/gemma4_31b_v4_ppp8x.yaml
