#!/usr/bin/env bash
# Hourly non-GPU evidence monitor.
#
# This is intentionally lightweight: it does not launch training/eval jobs and
# it never touches checkpoints.  It refreshes derived report artifacts from
# files already present under runs/ and records enough scheduler/GPU state to
# notice when new evidence should be inspected.
set -u

cd "$(dirname "$0")/.." || exit 1

INTERVAL="${INTERVAL:-3600}"
LOG="${LOG:-runs/hourly_evidence_check.log}"
LOCKDIR="${LOCKDIR:-runs/.hourly_evidence_check.lock}"
SIBLING="${SIBLING:-../selfupdate_multi}"

mkdir -p runs
if ! mkdir "$LOCKDIR" 2>/dev/null; then
    echo "[$(date '+%F %T')] hourly evidence check already running; lock=$LOCKDIR" >> "$LOG"
    exit 0
fi
trap 'rmdir "$LOCKDIR"' EXIT

run_step() {
    local label="$1"
    shift
    echo "[$(date '+%F %T')] step: $label"
    "$@" 2>&1 || echo "[$(date '+%F %T')] WARN: $label failed with status $?"
}

check_once() {
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] hourly evidence check"
    echo "================================================================"

    echo
    echo "[gpu]"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>&1 || true

    echo
    echo "[scheduler/processes]"
    ps -u "$USER" -o pid,ppid,etime,cmd \
        | rg 'scripts/(train|evaluate|teacher_ceiling|destruct_eval|standard_destruction_eval)\.py|gpu_scheduler\.sh|results_refresher|hourly_evidence_check' \
        || true

    echo
    echo "[recent local artifacts, last 75 minutes]"
    find runs -maxdepth 4 \( \
        -path '*/eval/recite.json' -o \
        -path '*/eval/destruction.json' -o \
        -path '*/eval/general.json' -o \
        -path '*/eval/text_examples.md' -o \
        -path 'runs/standard_destruction/*.json' -o \
        -path 'runs/standard_destruction/summary.csv' -o \
        -path 'runs/standard_destruction/summary.md' \
        \) -mmin -75 -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort || true

    echo
    echo "[recent sibling artifacts, last 75 minutes]"
    if [[ -d "$SIBLING/runs" ]]; then
        find "$SIBLING/runs" -maxdepth 4 \( \
            -name 'metrics.jsonl' -o \
            -path '*/eval/recite.json' -o \
            -path '*/eval/destruction.json' -o \
            -path '*/checkpoint' \
            \) -mmin -75 -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort || true
    else
        echo "sibling checkout not found: $SIBLING"
    fi

    echo
    echo "[artifact refresh]"
    if [[ -x .venv/bin/python ]]; then
        run_step "build loss-grid scorecard" .venv/bin/python scripts/lossgrid_report.py
        run_step "build run results and curves" .venv/bin/python scripts/analyze.py
        run_step "summarize standard destruction" .venv/bin/python scripts/summarize_standard_destruction.py
        run_step "build experiment report assets" .venv/bin/python scripts/experiment_report_assets.py
        run_step "build report pdf" .venv/bin/python scripts/report.py
    else
        echo "missing .venv/bin/python"
    fi

    echo
    echo "[git]"
    git status --short --branch 2>&1 || true
}

while true; do
    check_once >> "$LOG" 2>&1
    sleep "$INTERVAL"
done
