#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
if (( $# == 0 )); then
    echo "usage: $0 RUN_NAME [RUN_NAME ...]" >&2
    exit 2
fi

LOCK=runs/.report_v2_refresh_lock
LOG=${REPORT_REFRESH_LOG:-runs/report_v2_refresh.log}
if ! mkdir "$LOCK" 2>/dev/null; then
    printf 'LOCKED %s\n' "$LOCK" >&2
    exit 75
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

status=0
for run in "$@"; do
    if [[ ! -s "runs/$run/config.yaml" || ! -s "runs/$run/metrics.jsonl" ]]; then
        printf 'MISSING %s config_or_metrics\n' "$run" >> "$LOG"
        status=3
        continue
    fi
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
