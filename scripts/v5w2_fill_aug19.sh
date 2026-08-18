#!/bin/bash
# v5w2 queue fill, Aug 19-21 (owner: "review the results and fill the queue
# for the next three days", 2026-08-18 ~22:40). Run once from the repo root
# on a login node.
#
# Context (self-review #1 + tonight): rank axis closed (r8_long null, wall
# rank-independent), loss menu closed, dvmse/norms closed, ungated tinc
# destroys. Remaining live jobs: dora_mb4 435527 (running), tinc_cos_tka
# 436939 (last tinc variant), r64_h400 435546. This fill executes the
# pre-registered review-2 pivot early: the L54 selection question — is the
# per-step topk_abs re-ranking (chasing its own drift) what caps recall at
# ~0.192, separately from WHERE it writes?
#
# New arms (all delta_cosine, incumbent recipe, knobs pinned):
#   f54     fixed:54          pin the collapse layer, no churn (causal test)
#   f22     fixed:22          pin mid-stack (placement test from the
#                             writing side; dvmse's increment ranking there)
#   frz10   topk_abs:1 frozen after e10 (dynamic exploration, then no churn)
#   tka2    topk_abs:2        widen between 1 (wall) and dense (destroys)
# Longs (steady-prefill past review #2): frz10_long + f54_long, 300 ep —
# the wall's peak epoch is >40 (r32 peaked e160), so churn-free longs are
# the decisive measurement; review #2/#3 may prune (scancel pending = free).
set -euo pipefail
cd "$(dirname "$0")/.."
ACCT="--account=supercomplex"

TINC_SCREEN=436939   # last queued arm on the partition (no dependency)

COMMON="--positions aligned --epochs 40 --eval-every 2 --lr 1e-5 \
  --lora-r 32 --lora-alpha 64"
WINNER="--local-loss delta_cosine"

echo "== gate2 smoke (afterany tinc screen: never blocks on its failure)"
SMOKE=$(sbatch --parsable $ACCT --dependency="afterany:$TINC_SCREEN" \
  scripts/v5w2_smoke_gate2.sbatch)
echo "smoke_gate2: $SMOKE"

submit() { # name time dep args...
  local name="$1" t="$2" dep="$3"; shift 3
  local jid
  # shellcheck disable=SC2086
  jid=$(sbatch --parsable $ACCT -J "$name" -t "$t" --dependency="$dep" \
    scripts/trainv5.sbatch --run-name "$name" $COMMON "$@")
  echo "$name: $jid (dep $dep)"
  LAST=$jid
}

echo "== chain 1: gate-churn mechanism"
submit v5w2_dcos_f54 06:00:00 "afterok:$SMOKE" \
  $WINNER --layer-gate fixed:54
submit v5w2_dcos_tka_frz10 06:00:00 "afterany:$LAST" \
  $WINNER --layer-gate topk_abs:1 --gate-freeze-epoch 10
submit v5w2_dcos_tka_frz10_long 24:00:00 "afterany:$LAST" \
  $WINNER --layer-gate topk_abs:1 --gate-freeze-epoch 10 \
  --epochs 300 --eval-every 10

echo "== chain 2: placement / width"
submit v5w2_dcos_f22 06:00:00 "afterok:$SMOKE" \
  $WINNER --layer-gate fixed:22
submit v5w2_dcos_tka2 06:00:00 "afterany:$LAST" \
  $WINNER --layer-gate topk_abs:2
submit v5w2_dcos_f54_long 24:00:00 "afterany:$LAST" \
  $WINNER --layer-gate fixed:54 --epochs 300 --eval-every 10

echo "== all submitted"
