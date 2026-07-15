#!/usr/bin/env bash
# Idempotent pipeline-v3 launch: one process materializes the node-local
# epoch-zero cache; concurrent launchers wait/reuse it, then begin training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="configs/base.yaml"
EXPERIMENT=""
WAIT_SECONDS="7200"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --experiment)
      EXPERIMENT="$2"
      shift 2
      ;;
    --node-cache-wait-seconds)
      WAIT_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      echo "usage: scripts/l40s_train_v3.sh [--config PATH] --experiment PATH [--node-cache-wait-seconds N]"
      exit 0
      ;;
    *)
      echo "unsupported v3 launcher argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ -n "$EXPERIMENT" ]] || {
  echo "--experiment is required" >&2
  exit 2
}
RUN_NAME="$(awk '$1 == "run_name:" {print $2; exit}' "$EXPERIMENT")"
[[ -n "$RUN_NAME" ]] || {
  echo "experiment has no run_name: $EXPERIMENT" >&2
  exit 2
}

COMMON=(--config "$CONFIG" --experiment "$EXPERIMENT")

"$ROOT/scripts/l40s_exec.sh" "$ROOT/scripts/build_teacher_cache.py" \
  "${COMMON[@]}" --coordinated-node-cache \
  --node-cache-wait-seconds "$WAIT_SECONDS"
"$ROOT/scripts/l40s_exec.sh" "$ROOT/scripts/train.py" "${COMMON[@]}"
"$ROOT/scripts/l40s_exec.sh" "$ROOT/scripts/report_v2.py" "$RUN_NAME"
