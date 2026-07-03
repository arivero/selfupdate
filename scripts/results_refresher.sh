#!/usr/bin/env bash
# Refresh runs/results.md + curves every 15 min while the scheduler lives.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f 'gpu_scheduler[.]sh' >/dev/null; do
    .venv/bin/python scripts/analyze.py >/dev/null 2>&1
    sleep 900
done
.venv/bin/python scripts/analyze.py >/dev/null 2>&1  # final refresh
echo "[$(date '+%F %T')] refresher: scheduler gone, final refresh done"
