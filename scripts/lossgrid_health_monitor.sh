#!/usr/bin/env bash
# Read-only 30-minute health record for the active 1.7B loss grid.
#
# It deliberately does not restart, kill, or otherwise mutate campaign jobs.
# The scheduler owns execution; this monitor makes enough state durable to
# diagnose a stalled lane without watching a terminal.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

OUT="${OUT:-runs/lossgrid_health.log}"
INTERVAL="${INTERVAL:-1800}"
STALE_AFTER="${STALE_AFTER:-2100}"  # 35 min: longer than one expected epoch.
STATE_DIR="${STATE_DIR:-runs/.lossgrid_health_state}"

# Print only error-shaped lines appended since the previous sample.  The agent
# reviews these candidates; this script does not try to decide whether an
# exception is transient, a scheduler race, or a scientific-integrity fault.
error_deltas() {
    local log key cursor size prior delta matches
    mkdir -p "$STATE_DIR"
    for log in runs/pipeline_sched_a_lossgrid_*.log runs/a_lossgrid_joblogs/*.log; do
        [ -f "$log" ] || continue
        key="$(printf '%s' "$log" | tr '/.' '__')"
        cursor="$STATE_DIR/$key.offset"
        size="$(stat -c %s "$log")"
        if [ ! -f "$cursor" ]; then
            printf '%s\n' "$size" > "$cursor"
            continue
        fi
        prior="$(cat "$cursor")"
        case "$prior" in *[!0-9]*|'') prior=0;; esac
        [ "$size" -lt "$prior" ] && prior=0
        if [ "$size" -gt "$prior" ]; then
            delta="$(tail -c "+$((prior + 1))" "$log")"
            matches="$(printf '%s\n' "$delta" | grep -E -i \
                'traceback|exception|error:|cuda.*out of memory|segmentation|fatal|sched: fail' || true)"
            if [ -n "$matches" ]; then
                printf 'new error candidates: %s\n' "$log"
                printf '%s\n' "$matches" | tail -n 40
            fi
        fi
        printf '%s\n' "$size" > "$cursor"
    done
}

snapshot() {
    local now mtime age state line epoch items loss checkpoint evals run
    now="$(date +%s)"
    printf '\n[%s] loss-grid health\n' "$(date -Is)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader 2>&1 || true
    printf 'active campaign workers:\n'
    ps -eo pid=,etime=,stat=,args= | \
        awk '/scripts\/(train|evaluate|standard_destruction_eval)[.]py/ && /a_lossgrid/ {print}' || true
    printf 'recent metrics (checkpoint/eval are final artifacts):\n'
    for run in runs/a_lossgrid_1p7b_combined_*; do
        [ -f "$run/metrics.jsonl" ] || continue
        mtime="$(stat -c %Y "$run/metrics.jsonl")"
        age=$((now - mtime))
        line="$(grep '"kind": "train"' "$run/metrics.jsonl" | tail -n 1 || true)"
        epoch="$(printf '%s' "$line" | sed -n 's/.*"epoch": \([0-9][0-9]*\).*/\1/p')"
        items="$(printf '%s' "$line" | sed -n 's/.*"items_seen": \([0-9][0-9]*\).*/\1/p')"
        loss="$(printf '%s' "$line" | sed -n 's/.*"loss": \([^,}]*\).*/\1/p')"
        checkpoint=no; [ -d "$run/checkpoint" ] && checkpoint=yes
        state="fresh"
        if [ "$checkpoint" = yes ]; then
            state="checkpoint-ready"
        elif [ "$age" -gt "$STALE_AFTER" ]; then
            state="STALE"
        fi
        evals="$(find "$run/eval" -maxdepth 1 -type f 2>/dev/null | wc -l)"
        printf '%s: %s age=%ss epoch=%s items=%s loss=%s checkpoint=%s eval_files=%s\n' \
            "${run#runs/}" "$state" "$age" "${epoch:--}" "${items:--}" \
            "${loss:--}" "$checkpoint" "$evals"
    done
    error_deltas
}

while true; do
    snapshot
    sleep "$INTERVAL"
done
