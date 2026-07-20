#!/usr/bin/env bash
# 122B PPP8 cross-node campaign40 launch (agpuh01 GPUs 0-3 + agpuh02 GPUs
# 0-3), 40 epochs, lr 3.0e-6. Run FROM agpuh01. Mirrors
# scripts/launch_q122b_ppp8x.sh (the proven evalin cross-node path,
# NCCL-hang-fixed and validated 2026-07-20 per issues.md) but points at the
# campaign40 e40 config instead of the 3-epoch evalin config. Unsets
# node-local relay root so the multi-host branch picks the shared Lustre
# exchange; cross-node mail = native IB (auto).
#
# Prerequisite: agpuh02 must have its 122B teacher-cache index built
# (scripts/chain_ppp8_when_ready.sh step 2, or confirm
# /dev/shm/$USER/selfupdate-teacher-cache-v4-q122b already has a populated
# index from the prior evalin run) before this launch.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
unset SELFUPDATE_V4_RELAY_ROOT
export SELFUPDATE_V4_STAGE_HOSTS="local local local local agpuh02 agpuh02 agpuh02 agpuh02"
exec scripts/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_qwen35_122b_v4_full.yaml \
  configs/experiments/train40/qwen35_122b_v4_ppp8x_e40.yaml
