#!/usr/bin/env bash
# Qwen3-0.6B PPP2 CROSS-NODE relay TEST (owner 2026-07-19): stage 0 on agpuh01,
# stage 1 on agpuh02. The single boundary (0->1) straddles the node boundary,
# so it MUST ride NCCL/IB — proving the rewritten relay carries epoch +
# store-fill hiddens over InfiniBand with ZERO files on any disk filesystem.
# Assert after: runs/v4_relay is NEVER created.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export SELFUPDATE_V4_STAGE_HOSTS="local agpuh02"
exec defactorised/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml \
  configs/experiments/h100_smoke/qwen3_0p6b_v4_ppp2_xnode.yaml
