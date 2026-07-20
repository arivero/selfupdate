#!/usr/bin/env bash
# Publish any pending individual reports and grouped pipeline-v2 views every
# 15 minutes while the scheduler lives, then once more after it exits.
cd "$(dirname "$0")/.." || exit 1
refresh() {
    for run in runs/pareto_v2_*; do
        [[ -d "$run/checkpoint" && -f "$run/eval/signal_attribution.json" ]] || continue
        # A v2 report is not publishable until its printable individual PDF
        # accompanies the Markdown, figure assets, and manifest.
        [[ -f "$run/report_manifest.json" && -f "$run/report.pdf" ]] \
            && grep -q '"pdf":' "$run/report_manifest.json" && continue
        defactorised/l40s_exec.sh defactorised/report_v2.py "$run" >/dev/null 2>&1 || true
    done
    defactorised/l40s_exec.sh defactorised/group_reports_v2.py --group-by all >/dev/null 2>&1 || true
}
while pgrep -f 'gpu_scheduler[.]sh' >/dev/null; do
    refresh
    sleep 900
done
refresh
echo "[$(date '+%F %T')] refresher: scheduler gone, final refresh done"
