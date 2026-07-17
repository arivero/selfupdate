#!/usr/bin/env bash
# One-shot gemma-4-31B-it PPP4 launch (micro_batch 16 fix, 3927eea).
# Wrapper exists because ssh lands in /fs/agustina/arivero/supercomplex,
# not the repo (docs/h100_bringup.md landing-dir trap).
set -u
cd /fs/agustina/arivero/supercomplex/selfup_teacher || exit 1
exec scripts/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_gemma4_31b_v4_full.yaml \
  configs/experiments/h100_smoke/gemma4_31b_v4_ppp4.yaml
