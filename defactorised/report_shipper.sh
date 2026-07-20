#!/usr/bin/env bash
# Ship the auto-generated reports to DEST every 15 min while the scheduler
# lives (companion of results_refresher.sh, which regenerates them), plus a
# final shipment when it exits. Requires ssh key auth (BatchMode never
# prompts). Override DEST for another drop point.
cd "$(dirname "$0")/.." || exit 1
DEST="${DEST:-arivero@lxbifi11.bifi.unizar.es:tmp/}"
ship() {
    scp -o BatchMode=yes -o ConnectTimeout=15 -q \
        runs/results.md runs/report.pdf runs/curves.png "$DEST" \
        && echo "[$(date '+%F %T')] shipper: reports -> $DEST" \
        || echo "[$(date '+%F %T')] shipper: scp FAILED (keys? host down?)"
}
while pgrep -f 'gpu_scheduler[.]sh' >/dev/null; do
    ship
    sleep 900
done
ship
echo "[$(date '+%F %T')] shipper: scheduler gone, final shipment done"
