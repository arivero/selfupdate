#!/usr/bin/env bash
# gemma-4-31B PPP8 cross-node (agpuh01 GPU0-3 + agpuh02 GPU0-3). Fills the
# smaller-model PPP8 gap. Model + g31b cache staged on both nodes' shm.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export SELFUPDATE_V4_STAGE_HOSTS="local local local local agpuh02 agpuh02 agpuh02 agpuh02"
exec scripts/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_gemma4_31b_v4_full.yaml \
  configs/experiments/h100_smoke/gemma4_31b_v4_ppp8x.yaml
