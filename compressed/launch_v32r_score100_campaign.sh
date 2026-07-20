#!/usr/bin/env bash
# Launch the eight fixed GPU placements in the reduced-score v3.2 campaign.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="configs/experiments/pareto_v3/base_qwen35_4b.yaml"
LOG_ROOT="runs/pareto_v32r_score100_worker_logs"

launch_one() {
  local host="$1" gpu="$2" config="$3"
  local short="${config##*/}"
  short="${short%.yaml}"
  local log="$LOG_ROOT/$host/gpu${gpu}_${short}.log"
  local pid_file="$LOG_ROOT/$host/gpu${gpu}_${short}.pid"
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$ROOT" "$gpu" "$BASE" "$config" "$log" "$pid_file" <<'SH'
set -euo pipefail
root="$1" gpu="$2" base="$3" config="$4" log="$5" pid_file="$6"
cd "$root"
mkdir -p "$(dirname "$log")"
if [[ -s "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
  echo "refusing duplicate launch: host=$(hostname -s) gpu=$gpu pid=$(<"$pid_file") config=$config" >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="$gpu" \
PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
TRANSFORMERS_VERBOSITY=error \
nohup setsid compressed/l40s_train_v3.sh \
  --config "$base" --experiment "$config" \
  >"$log" 2>&1 </dev/null &
pid=$!
echo "$pid" >"$pid_file"
echo "START_MARKER host=$(hostname -s) gpu=$gpu launcher_pid=$pid config=$config log=$log"
SH
}

launch_one agpul04 0 configs/experiments/pareto_v3/qwen35_4b_v32r_score100_flow_vocabcos256_lora_einf.yaml
launch_one agpul04 1 configs/experiments/pareto_v3/qwen35_4b_v32r_score100_flow_vocabcos256_full_einf.yaml
launch_one agpul04 2 configs/experiments/pareto_v3/qwen35_4b_v32r_score100_intact_huber_lora_einf.yaml
launch_one agpul04 3 configs/experiments/pareto_v3/qwen35_4b_v32r_score100_intact_huber_full_einf.yaml
launch_one agpul05 0 configs/experiments/pareto_v3/qwen35_4b_v32r_score100_flow_cosine_lora_einf.yaml
launch_one agpul05 1 configs/experiments/pareto_v3/qwen35_4b_v32r_score100_flow_cosine_full_einf.yaml
launch_one agpul05 2 configs/experiments/pareto_v3/qwen35_4b_v32r_score100_flow_huber_lora_einf.yaml
launch_one agpul05 3 configs/experiments/pareto_v3/qwen35_4b_v32r_score100_flow_huber_full_einf.yaml
