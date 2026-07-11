#!/usr/bin/env bash
# Refresh runs/results.md + curves + report.pdf every 15 min while the
# scheduler lives, and once more after it exits.
cd "$(dirname "$0")/.." || exit 1
refresh() {
    .venv/bin/python scripts/lossgrid_report.py >/dev/null 2>&1
    .venv/bin/python scripts/analyze.py >/dev/null 2>&1
    .venv/bin/python scripts/report.py >/dev/null 2>&1
}
while pgrep -f 'gpu_scheduler[.]sh' >/dev/null; do
    refresh
    sleep 900
done
refresh
echo "[$(date '+%F %T')] refresher: scheduler gone, final refresh done"
