#!/usr/bin/env bash
# Qwen3-0.6B PPP3 cross-node relay-drain bench (#24): stage 0,1 on agpuh01,
# stage 2 on agpuh02. 4 epochs; the last stage's eval tail lags the fast
# stages, reproducing the DeepSeek PPP8 finalize crash. PASS = clean exit,
# all stages 4 v4_epoch rows, no NCCL timeout.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export SELFUPDATE_V4_STAGE_HOSTS="local local agpuh02"
exec scripts/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml \
  configs/experiments/h100_smoke/qwen3_0p6b_v4_ppp3_xnode.yaml
