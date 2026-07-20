#!/usr/bin/env bash
# Readable CPU-only PPn wavefront orchestration demo.
#
# Every stage is a separate Python worker. Workers communicate only through
# validated JSON packets atomically published beneath WORK_DIR; no worker
# imports the package tree. The JSON boundary is a teaching stand-in: production
# same-node PPn uses RAM transport and cross-node PPn uses NCCL/InfiniBand.
# This is not pipeline-v4 PPP, whose stages optimize independently.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="$SCRIPT_DIR/ppn_stage_demo.py"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "/tmp/${USER:-selfupdate}/selfupdate-venv/bin/python" ]]; then
  PYTHON_BIN="/tmp/${USER:-selfupdate}/selfupdate-venv/bin/python"
elif [[ -x /opt/ohpc/pub/apps/anaconda/anaconda-2025/bin/python3 ]]; then
  PYTHON_BIN=/opt/ohpc/pub/apps/anaconda/anaconda-2025/bin/python3
else
  PYTHON_BIN=python3
fi

stages=3
tiles=4
timeout_seconds=20
fail_stage=-1
work_dir=""
keep_work_dir=0

usage() {
  cat <<'EOF'
Usage: ppn_demo.sh [options]

Options:
  --stages N          Number of concurrent pipeline stages (default: 3)
  --tiles N           Number of packets sent through the pipeline (default: 4)
  --timeout-seconds S Worker handoff timeout (default: 20)
  --fail-stage N      Inject failure in stage N; -1 disables it (default: -1)
  --work-dir DIR      Use DIR instead of a fresh /dev/shm directory
  --keep-work-dir     Retain an automatically created work directory
  --python PATH       Python interpreter (default: project venv when present)
  -h, --help          Show this help
EOF
}

while (($#)); do
  case "$1" in
    --stages) stages="$2"; shift 2 ;;
    --tiles) tiles="$2"; shift 2 ;;
    --timeout-seconds) timeout_seconds="$2"; shift 2 ;;
    --fail-stage) fail_stage="$2"; shift 2 ;;
    --work-dir) work_dir="$2"; shift 2 ;;
    --keep-work-dir) keep_work_dir=1; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$stages" =~ ^[1-9][0-9]*$ ]] || { echo "--stages must be positive" >&2; exit 2; }
[[ "$tiles" =~ ^[1-9][0-9]*$ ]] || { echo "--tiles must be positive" >&2; exit 2; }
[[ -f "$WORKER" ]] || { echo "missing worker: $WORKER" >&2; exit 2; }

created_work_dir=0
if [[ -z "$work_dir" ]]; then
  demo_tmp_root="/dev/shm/${USER:-selfupdate}"
  if [[ ! -d /dev/shm || ! -w /dev/shm ]]; then
    demo_tmp_root="${TMPDIR:-/tmp}"
  else
    mkdir -p "$demo_tmp_root"
  fi
  work_dir="$(mktemp -d "$demo_tmp_root/selfupdate-ppn-demo.XXXXXX")"
  created_work_dir=1
else
  mkdir -p "$work_dir"
  work_dir="$(cd "$work_dir" && pwd)"
fi

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  if ((created_work_dir && !keep_work_dir)); then
    rm -rf -- "$work_dir"
  fi
}
trap cleanup EXIT INT TERM

launch_id="ppn-demo-$(date +%s)-$$"
echo "launch: $launch_id"
echo "work directory: $work_dir"
echo "topology: $stages stages, $tiles tiles"

# Start all stages together. Downstream workers wait for their predecessor's
# atomic packet, illustrating pipeline concurrency without framework magic.
for ((stage = 0; stage < stages; stage++)); do
  "$PYTHON_BIN" "$WORKER" \
    --stage "$stage" \
    --stages "$stages" \
    --work-dir "$work_dir" \
    --launch-id "$launch_id" \
    --tiles "$tiles" \
    --timeout-seconds "$timeout_seconds" \
    --fail-stage "$fail_stage" &
  pids+=("$!")
  echo "started stage $stage as pid ${pids[-1]}"
done

failed=0
for ((stage = 0; stage < stages; stage++)); do
  if ! wait "${pids[$stage]}"; then
    echo "stage $stage failed; stopping sibling workers" >&2
    failed=1
    break
  fi
  echo "stage $stage complete"
done

if ((failed)); then
  exit 1
fi
pids=()

echo "final tile envelopes:"
for result in "$work_dir"/results/tile_*.json; do
  [[ -e "$result" ]] || { echo "no final results produced" >&2; exit 1; }
  printf '  %s: ' "$(basename "$result")"
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["value"])' "$result"
done

if ((created_work_dir && !keep_work_dir)); then
  echo "temporary work directory will be removed"
else
  echo "artifacts retained in: $work_dir"
fi
