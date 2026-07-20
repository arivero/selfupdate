#!/usr/bin/env bash
# PPP5 cross-node launch (agpuh01 GPU3 + agpuh02 GPUs 0-3). Run FROM
# agpuh01. Unsets the node-local relay root so the launcher's multi-host
# branch picks the shared Lustre exchange.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
unset SELFUPDATE_V4_RELAY_ROOT
export SELFUPDATE_V4_STAGE_HOSTS="local agpuh02 agpuh02 agpuh02 agpuh02"
exec defactorised/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_gemma4_31b_v4_full.yaml \
  configs/experiments/h100_smoke/gemma4_31b_v4_ppp5_xnode.yaml
