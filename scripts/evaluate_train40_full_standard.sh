#!/usr/bin/env bash
# Paired 100-item standard-benchmark endpoints for merged campaign40 adapters.
#
# Run on an otherwise idle H100 node.  One visible GPU is sufficient for the
# 26B--35B models.  Set TRAIN40_ENDPOINT_AUTO_MAP=1 and expose all four local
# GPUs for a model that does not fit one card.  Results use report_v2.py's
# canonical runs/standard_damage names.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

if (( $# == 0 )); then
    echo "usage: CUDA_VISIBLE_DEVICES=N $0 RUN_NAME [RUN_NAME ...]" >&2
    exit 2
fi

PY=${PY:-/tmp/$USER/selfupdate-venv/bin/python}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TQDM_DISABLE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
mkdir -p runs/standard_damage

complete_result() {
    "$PY" - "$1" "$2" "$3" "${4:-}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_model, expected_role, expected_checkpoint = sys.argv[2:5]
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
tasks = payload.get("tasks") or {}
required = ("arc_easy", "arc_challenge", "hellaswag")
def valid_task(task):
    try:
        return (int(tasks[task]["n"]) >= 100
                and math.isfinite(float(tasks[task]["accuracy"])))
    except (KeyError, TypeError, ValueError):
        return False

ok = all(valid_task(task) for task in required)
ok = ok and payload.get("model") == expected_model
ok = ok and payload.get("teacher_reference_kind") == expected_role
if expected_role == "epoch_zero":
    ok = ok and payload.get("checkpoint") is None
else:
    ok = ok and payload.get("checkpoint") == expected_checkpoint
raise SystemExit(0 if ok else 1)
PY
}

auto_args=()
if [[ ${TRAIN40_ENDPOINT_AUTO_MAP:-0} == 1 ]]; then
    auto_args+=(--auto-map)
fi

last_model=
for run in "$@"; do
    run_dir="runs/$run"
    config="$run_dir/stage0/config.yaml"
    checkpoint="$run_dir/checkpoint"
    if [[ ! -s "$config" || ! -s "$checkpoint/adapter_config.json" || \
          ! -s "$checkpoint/adapter_model.safetensors" ]]; then
        echo "MISSING $run merged_checkpoint_or_stage0_config" >&2
        exit 3
    fi

    model=$(
        "$PY" - "$config" <<'PY'
import sys
import yaml
payload = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
print((payload.get("model") or {}).get("name") or "")
PY
    )
    if [[ -z "$model" ]]; then
        echo "MISSING $run model_identity" >&2
        exit 4
    fi
    base_out="runs/standard_damage/epoch0_${model//\//_}.json"
    checkpoint_out="runs/standard_damage/$run.json"

    # The first evaluation for a new model reaps expired node-local staging
    # before publishing that model.  The paired checkpoint then reuses it.
    if ! complete_result "$base_out" "$model" epoch_zero; then
        echo "START $(date --iso-8601=seconds) $run epoch-zero n=100/task"
        echo "COMMAND standard_destruction_eval model=$model base tasks=arc_easy,arc_challenge,hellaswag limit=100"
        SELFUPDATE_EVAL_STAGE_TTL_DAYS=0 "$PY" scripts/standard_destruction_eval.py \
            --config "$config" --base --out "$base_out" \
            --tasks arc_easy arc_challenge hellaswag --limit 100 \
            --batch-size 16 --stage-to-local "${auto_args[@]}"
        echo "EXIT $(date --iso-8601=seconds) $run epoch-zero 0"
        last_model=$model
    fi

    if complete_result "$checkpoint_out" "$model" checkpoint "$checkpoint"; then
        echo "SKIP $run complete_checkpoint_endpoint"
        last_model=$model
        continue
    fi
    ttl=7
    if [[ "$last_model" != "$model" ]]; then
        ttl=0
    fi
    echo "START $(date --iso-8601=seconds) $run checkpoint n=100/task"
    echo "COMMAND standard_destruction_eval run=$run checkpoint=$checkpoint tasks=arc_easy,arc_challenge,hellaswag limit=100"
    SELFUPDATE_EVAL_STAGE_TTL_DAYS=$ttl "$PY" scripts/standard_destruction_eval.py \
        --config "$config" --checkpoint "$checkpoint" \
        --out "$checkpoint_out" --tasks arc_easy arc_challenge hellaswag \
        --limit 100 --batch-size 16 --stage-to-local "${auto_args[@]}"
    echo "EXIT $(date --iso-8601=seconds) $run checkpoint 0"
    complete_result "$checkpoint_out" "$model" checkpoint "$checkpoint"
    last_model=$model
done
