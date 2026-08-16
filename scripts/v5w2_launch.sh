#!/bin/bash
# v5-week-2 campaign submission (2026-08-16). Run once from the repo root on
# a login node. Submits: the smoke gate, two 6-arm screen chains (one per
# idle H100 node, heads afterok:smoke, links afterany so a crashed arm never
# blocks its lane), and three claude-review appointments (Aug 18/20/23).
# Every knob that distinguishes an arm from its reference is pinned
# explicitly (config-defaults lesson).
set -euo pipefail
cd "$(dirname "$0")/.."
ACCT="--account=supercomplex"

COMMON="--positions aligned --epochs 40 --eval-every 2 --lr 1e-5 \
  --lora-r 32 --lora-alpha 64"
WINNER="--local-loss delta_cosine --layer-gate topk_abs:1"

echo "== smoke gate"
SMOKE=$(sbatch --parsable $ACCT scripts/v5w2_smoke.sbatch)
echo "smoke: $SMOKE"

submit_chain() { # lane_name dep_head arm_spec...
  local lane="$1"; shift
  local dep="afterok:$SMOKE"
  echo "== lane $lane"
  for spec in "$@"; do
    local name="${spec%%|*}" args="${spec#*|}"
    # shellcheck disable=SC2086
    local jid
    jid=$(sbatch --parsable $ACCT -J "$name" -t 06:00:00 \
      --dependency="$dep" scripts/trainv5.sbatch \
      --run-name "$name" $COMMON $args)
    echo "$name: $jid (dep $dep)"
    dep="afterany:$jid"
  done
}

# Lane A — alternate losses
submit_chain A \
  "v5w2_dvmse_tka|--local-loss delta_vmse --layer-gate topk_abs:1" \
  "v5w2_dvmse|--local-loss delta_vmse" \
  "v5w2_mix_tka|--local-loss mix --layer-gate topk_abs:1" \
  "v5w2_mix|--local-loss mix" \
  "v5w2_nmse|--local-loss nmse" \
  "v5w2_cos|--local-loss cosine"

# Lane B — LoRA room (base = campaign winner; rank flags OVERRIDE the
# r32/a64 in COMMON because they come later on the command line)
submit_chain B \
  "v5w2_dcos_tka_r64|$WINNER --lora-r 64 --lora-alpha 128" \
  "v5w2_dcos_tka_r128|$WINNER --lora-r 128 --lora-alpha 256" \
  "v5w2_norms|$WINNER --train-norms" \
  "v5w2_dora|$WINNER --dora" \
  "v5w2_dcos_tka_r8|$WINNER --lora-r 8 --lora-alpha 16" \
  "v5w2_vmse_norms|--local-loss vocab_mse --train-norms"

echo "== review appointments"
for day in 18 20 23; do
  rid=$(sbatch --parsable $ACCT --begin="2026-08-${day}T09:00:00" \
    scripts/claude_review_v5w2.sbatch)
  echo "review Aug $day: $rid"
done
echo "== all submitted"
