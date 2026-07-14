#!/usr/bin/env bash
# Publish any pending individual reports and grouped pipeline-v2 views every
# 15 minutes while the scheduler lives, then once more after it exits.
cd "$(dirname "$0")/.." || exit 1
refresh() {
    for run in runs/pareto_v2_*; do
        [[ -d "$run/checkpoint" && -f "$run/eval/signal_attribution.json" ]] || continue
        [[ -f "$run/report_manifest.json" ]] && continue
        scripts/l40s_exec.sh scripts/report_v2.py "$run" >/dev/null 2>&1 || true
    done
    scripts/l40s_exec.sh scripts/group_reports_v2.py --group-by all >/dev/null 2>&1 || true
}
while pgrep -f 'gpu_scheduler[.]sh' >/dev/null; do
    refresh
    sleep 900
done
refresh
echo "[$(date '+%F %T')] refresher: scheduler gone, final refresh done"
