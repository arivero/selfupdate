#!/usr/bin/env bash
# Launch every pipeline-v4 layer-shard stage of one experiment, one detached
# process per stage. The stage count comes from the CONFIG
# (train.v4_stage_splits: N cuts -> N+1 stages), so this works for any
# number of GPUs — nothing here assumes four.
#
# Usage:
#   scripts/launch_v4_stages.sh <base.yaml> <experiment.yaml>
#
# Each stage k:
#   - runs `train.py --v4-stage k` (train.py pins model.device to
#     v4_stage_devices[k] — physical id, never renumbered; CUDA_VISIBLE_DEVICES
#     is deliberately left alone)
#   - writes runs/<run_name>/stage<k>/ (metrics, checkpoint shard)
#   - logs to runs/<run_name>_stage<k>.log
# Merge shards afterwards with scripts/merge_v4_adapters.py.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${1:?usage: launch_v4_stages.sh <base.yaml> <experiment.yaml>}"
EXP="${2:?usage: launch_v4_stages.sh <base.yaml> <experiment.yaml>}"
PY="${SELFUPDATE_VENV:-/tmp/$USER/selfupdate-venv}/bin/python"

read -r STAGES RUN_NAME <<EOF2
$("$PY" - "$BASE" "$EXP" <<'PYEOF'
import sys
sys.path.insert(0, "src")
from selfupdate.config import load_config
cfg = load_config(sys.argv[1], sys.argv[2])
if cfg.train.pipeline_version != 4:
    raise SystemExit("launch_v4_stages.sh requires pipeline_version=4")
print(len(cfg.train.v4_stage_splits or []) + 1, cfg.run_name)
PYEOF
)
EOF2

export PYTORCH_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export SELFUPDATE_CPU_THREADS="${SELFUPDATE_CPU_THREADS:-8}"

# Prefer a completed RAM stage of the model snapshots (the container-era
# convention): N stage processes cold-loading a 54 GB model from Lustre sit
# in D-state page faults; from /dev/shm they load in seconds. Stage once
# per node with: scripts/stage_hf_cache.sh --shm <org/model>
SHM_HF="/dev/shm/$USER/selfupdate-hf-cache"
if [[ -f "$SHM_HF/.selfupdate-hf-stage-ready" ]]; then
  export HF_HOME="$SHM_HF"
  echo "model snapshots: RAM stage $SHM_HF"
else
  echo "model snapshots: account cache (Lustre) — for big models stage first:"
  echo "  scripts/stage_hf_cache.sh --shm <org/model>"
fi

mkdir -p "$ROOT/runs"
# Launch lease: a second invocation for the same run must refuse while the
# first one's stages are alive (the 2026-07-17 double-launch near-miss:
# two watchers raced; both sets would have written the same run directory
# and OOMed every card). Stale leases from dead pids are reclaimed.
LEASE="$ROOT/runs/.v4-launch-$(echo "$RUN_NAME" | tr '/' '_').pids"
if [[ -f "$LEASE" ]]; then
  while read -r oldpid; do
    if kill -0 "$oldpid" 2>/dev/null; then
      echo "REFUSED: $RUN_NAME already launched (live pid $oldpid, lease $LEASE)" >&2
      exit 3
    fi
  done < "$LEASE"
  rm -f "$LEASE"
fi
echo "launching $STAGES v4 stages of $RUN_NAME"
pids=()
for ((k = 0; k < STAGES; k++)); do
  logfile="$ROOT/runs/${RUN_NAME}_stage${k}.log"
  nohup setsid "$PY" "$ROOT/scripts/train.py" \
    --config "$BASE" --experiment "$EXP" --v4-stage "$k" \
    >> "$logfile" 2>&1 &
  pids+=($!)
  echo "  stage $k -> pid ${pids[-1]}  log $logfile"
  # Small stagger so concurrent model loads do not thrash the snapshot cache.
  sleep 5
done
printf '%s\n' "${pids[@]}" > "$LEASE"
# Reaper: if one stage dies of OOM/gate-abort, terminate the siblings — a
# set with a dead stage cannot publish; do not burn the other cards.
nohup setsid "$ROOT/scripts/v4_stage_reaper.sh" "$LEASE" \
  "$ROOT/runs/${RUN_NAME}_stage" >> "$ROOT/runs/${RUN_NAME}_reaper.log" 2>&1 &
echo "stage pids: ${pids[*]}  (reaper pid $!)"
echo "watch:  tail -f runs/${RUN_NAME}_stage*.log"
echo "merge:  $PY scripts/merge_v4_adapters.py runs/$RUN_NAME"
