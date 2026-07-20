#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

if (( $# == 0 )); then
    echo "usage: CUDA_VISIBLE_DEVICES=N $0 RUN_NAME [RUN_NAME ...]" >&2
    exit 2
fi

mkdir -p runs/standard_damage
base_out=runs/standard_damage/teacher_Qwen_Qwen3.5-4B.json
if [[ ! -s "$base_out" ]]; then
    printf 'START %s base\n' "$(date --iso-8601=seconds)"
    printf 'COMMAND full-standard Qwen3.5-4B base n=100/task\n'
    scripts/l40s_exec.sh scripts/standard_destruction_eval.py \
        --config configs/experiments/h100_smoke/base_qwen35_4b_v4_lora.yaml \
        --base --out "$base_out" \
        --tasks arc_easy arc_challenge hellaswag --limit 100 --batch-size 16
    rc=$?
    printf 'EXIT %s base %s\n' "$(date --iso-8601=seconds)" "$rc"
    (( rc == 0 )) || exit "$rc"
fi

for run in "$@"; do
    config="runs/$run/config.yaml"
    checkpoint="runs/$run/checkpoint"
    out="runs/standard_damage/$run.json"
    if [[ ! -s "$config" || ! -d "$checkpoint" ]]; then
        printf 'MISSING %s config_or_checkpoint\n' "$run" >&2
        exit 3
    fi
    printf 'START %s %s\n' "$(date --iso-8601=seconds)" "$run"
    printf 'COMMAND full-standard %s n=100/task\n' "$run"
    scripts/l40s_exec.sh scripts/standard_destruction_eval.py \
        --config "$config" --checkpoint "$checkpoint" --out "$out" \
        --tasks arc_easy arc_challenge hellaswag --limit 100 --batch-size 16
    rc=$?
    printf 'EXIT %s %s %s\n' "$(date --iso-8601=seconds)" "$run" "$rc"
    (( rc == 0 )) || exit "$rc"
done
