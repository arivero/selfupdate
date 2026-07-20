#!/usr/bin/env bash
# Run a layerwise training-speed probe while recording GPU utilization.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 defactorised/run_speed_check_monitored.sh name \
#     --model Qwen/Qwen3.6-27B --batches 1,2 --no-optimizer
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

name="${1:?first arg is output name under runs/speed_checks}"
shift

out_dir="runs/speed_checks"
mkdir -p "$out_dir"
base="$out_dir/$name"

nvidia-smi topo -m > "$base.topo.txt" 2>&1 || true
nvidia-smi > "$base.before.txt" 2>&1 || true

monitor_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s pucvmt -d 1 -o TD > "$base.dmon.tsv" 2>&1 &
    monitor_pid="$!"
fi

cleanup() {
    if [ -n "$monitor_pid" ] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

.venv/bin/python defactorised/speed_check.py "$@" --out "$base.json" 2>&1 | tee "$base.log"

nvidia-smi > "$base.after.txt" 2>&1 || true
