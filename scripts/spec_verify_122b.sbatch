#!/usr/bin/env bash
#SBATCH --job-name=spec_verify_122b
#SBATCH --account=supercomplex
#SBATCH --partition=rasmia_hopper
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --output=runs/spec_verify/batch_122b_%j.log
set -x
cd "$SLURM_SUBMIT_DIR"
scripts/venv_setup.sh && scripts/venv_check.sh || exit 1
PY=/tmp/$USER/selfupdate-venv/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True SELFUPDATE_CPU_THREADS=8
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
# subset+configs via the matrix helper's steps 1-2 (PPP1 build), then staged PPP4:
bash scripts/spec_verify_matrix.sh 122b 0 || true   # builds subset+base+cache (train step may refuse PPP1-resident; ignore)
bash scripts/launch_v4_stages.sh configs/experiments/spec_verify/base_122b_v4_spec.yaml \
                                 configs/experiments/spec_verify/122b_v4_spec_ppp4.yaml
# poll for the goal row on the last stage (launcher detaches its stages)
for i in $(seq 1 150); do
  rg -q '"teacher_output_eval"' runs/spec_122b_v4_ppp4_e1/stage3/metrics.jsonl 2>/dev/null && break
  sleep 30
done
rg '"teacher_output_eval"' runs/spec_122b_v4_ppp4_e1/stage3/metrics.jsonl || echo "NO GOAL ROW"
