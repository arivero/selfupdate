#!/bin/bash
#SBATCH --job-name=qwen_ppp4_e50
#SBATCH --partition=rasmia_hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=runs/sbatch_qwen_ppp4_50_%j.out
#SBATCH --error=runs/sbatch_qwen_ppp4_50_%j.err
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$ROOT"
[[ "$#" -eq 3 ]] || {
  echo "usage: sbatch scripts/qwen_ppp4_50.sbatch BASE EXP MODEL" >&2
  exit 2
}
BASE="$1"
EXP="$2"
MODEL="$3"
fail() { echo "FATAL: $*" >&2; exit 1; }

if [[ -f "$ROOT/.campaign_env.sh" ]]; then
  set -a
  source "$ROOT/.campaign_env.sh"
  set +a
fi
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export TRITON_CACHE_DIR="/tmp/$USER/selfupdate-triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER/selfupdate-torchinductor"

echo "==== model PPP4/50 start model=$MODEL exp=$EXP commit=$(git rev-parse HEAD) $(date -Is) ===="
scripts/venv_setup.sh || fail "venv setup"
scripts/venv_check.sh || fail "venv check"
PY="/tmp/$USER/selfupdate-venv/bin/python"
scripts/stage_hf_cache.sh --shm "$MODEL" || fail "model staging"

RUN_NAME="$("$PY" - "$BASE" "$EXP" <<'PY'
import sys
sys.path.insert(0, "src")
from selfupdate.config import load_config
cfg = load_config(sys.argv[1], sys.argv[2])
assert cfg.train.epochs == 50
assert len(cfg.train.v4_stage_splits) == 3
assert cfg.train.micro_batch > 1
print(cfg.run_name)
PY
)"
[[ ! -e "runs/$RUN_NAME" ]] || fail "fresh run already exists: runs/$RUN_NAME"

scripts/launch_v4_stages.sh "$BASE" "$EXP" || fail "PPP4 launch"
LEASE="runs/.v4-launch-${RUN_NAME}.pids.local"
while [[ -f "$LEASE" ]]; do
  alive=0
  while read -r pid _stage; do
    kill -0 "$pid" 2>/dev/null && alive=1
  done < "$LEASE"
  [[ "$alive" -eq 0 ]] && break
  sleep 60
done
for stage in 0 1 2 3; do
  [[ -s "runs/$RUN_NAME/stage$stage/checkpoint/adapter_model.safetensors" ]] \
    || fail "$RUN_NAME stage $stage ended without a LoRA checkpoint"
  grep -q "run complete:" "runs/${RUN_NAME}_stage${stage}.log" \
    || fail "$RUN_NAME stage $stage lacks run-complete marker"
done
echo "==== model PPP4/50 complete run=$RUN_NAME $(date -Is) ===="
