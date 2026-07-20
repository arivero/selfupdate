#!/usr/bin/env bash
# Overnight chain (2026-07-18): once the M1 sequencer finishes (verdict
# file closed), launch the 500-epoch gemma-26B memorization test on the
# freed agpuh01 GPUs. Detached launcher; the agent reviews at its 05:50
# wake.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
LOG=runs/g26b_e500_chain.log
echo "chain start $(date -Is)" >> "$LOG"
# Wait for the M1 verdict to be complete (sequencer's last line).
while ! grep -q "M1 sequencer end" runs/m1_verdict.txt 2>/dev/null; do
  # Bail out to a direct wait if the sequencer died without a verdict:
  # no sequencer process AND no m1 train.py -> proceed anyway.
  if ! ps auxww | grep -q "[r]un_m1_legs.sh" \
     && ! ps auxww | grep "train.py" | grep -v grep | grep -q "m1[a-d]_0p6b"; then
    echo "sequencer gone without verdict; proceeding $(date -Is)" >> "$LOG"
    break
  fi
  sleep 60
done
echo "M1 chain clear $(date -Is)" >> "$LOG"
compressed/launch_v4_stages.sh \
  configs/experiments/h100_smoke/base_gemma4_26b_v4_full.yaml \
  configs/experiments/h100_smoke/gemma4_26b_v4_ppp4_e500.yaml \
  >> "$LOG" 2>&1
echo "g26b e500 launcher exited $(date -Is)" >> "$LOG"
