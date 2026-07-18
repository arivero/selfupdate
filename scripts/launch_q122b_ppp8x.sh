#!/usr/bin/env bash
# 122B PPP8 cross-node launch (agpuh01 GPUs 0-3 + agpuh02 GPUs 0-3). Run
# FROM agpuh01. Unsets node-local relay root so the multi-host branch
# picks the shared Lustre exchange; cross-node mail = native IB (auto).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
unset SELFUPDATE_V4_RELAY_ROOT
export SELFUPDATE_V4_STAGE_HOSTS="local local local local agpuh02 agpuh02 agpuh02 agpuh02"
exec scripts/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_qwen35_122b_v4_full.yaml \
  configs/experiments/h100_smoke/qwen35_122b_v4_ppp8_xnode.yaml
