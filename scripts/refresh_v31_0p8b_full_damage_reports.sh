#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

LOCK=runs/.v31_report_refresh_k16_lock
LOG=runs/v31_report_refresh_full_damage_k16.log
if ! mkdir "$LOCK" 2>/dev/null; then
    printf 'LOCKED %s\n' "$LOCK" >&2
    exit 75
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

runs=(
    pareto_v31_qwen35_0p8b_flow_student_b256k16_cosine_lr1e5_s17
    pareto_v31_qwen35_0p8b_flow_student_b256k16_huber_lr1e5_s17
    pareto_v31_qwen35_0p8b_flow_student_b256k16_huber_lr3e6_s17
    pareto_v31_qwen35_0p8b_flow_student_b256k16_huber_lr1e6_s17
    pareto_v31_qwen35_0p8b_random_student_b256k16_huber_lr1e5_s17
    pareto_v31_qwen35_0p8b_random_student_b256k16_huber_lr3e6_s17
    pareto_v31_qwen35_0p8b_random_student_b256k16_huber_lr1e6_s17
)

status=0
for run in "${runs[@]}"; do
    tmp=$(mktemp)
    printf 'START %s %s\nCOMMAND scripts/l40s_exec.sh scripts/report_v2.py %s\n' \
        "$(date --iso-8601=seconds)" "$run" "$run" >> "$LOG"
    scripts/l40s_exec.sh scripts/report_v2.py "$run" >"$tmp" 2>&1
    rc=$?
    rg -i 'warning|error|traceback|failed|missing|manifest|report\.pdf' "$tmp" \
        >> "$LOG" || true
    printf 'EXIT %s %s %s\n' "$(date --iso-8601=seconds)" "$run" "$rc" >> "$LOG"
    rm -f "$tmp"
    if (( rc != 0 )); then
        status=$rc
    fi
done

exit "$status"
