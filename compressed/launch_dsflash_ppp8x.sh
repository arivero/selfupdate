#!/usr/bin/env bash
# DeepSeek-V4-Flash PPP8 cross-node speed launch from agpuh01.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
unset SELFUPDATE_V4_RELAY_ROOT
export SELFUPDATE_V4_STAGE_HOSTS="local local local local agpuh02 agpuh02 agpuh02 agpuh02"
exec compressed/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_deepseek_v4_flash_v4_full.yaml \
  configs/experiments/h100_smoke/deepseek_v4_flash_ppp8_xnode.yaml
