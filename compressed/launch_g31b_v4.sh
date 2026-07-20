#!/usr/bin/env bash
# One-shot Gemma-31B PPP4 launch.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
exec compressed/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_gemma4_31b_v4_full.yaml \
  configs/experiments/h100_smoke/gemma4_31b_v4_ppp4.yaml
