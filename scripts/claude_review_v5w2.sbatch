#!/bin/bash
#SBATCH --job-name=claude-review
#SBATCH --partition=agustina_thin
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=runs/sbatch_claude_review_%j.out
#SBATCH --error=runs/sbatch_claude_review_%j.err
#
# v5-week-2 scheduled self-reviews (owner-approved 2026-08-16): resume the
# campaign Claude session on the thin queue with promotion/relaunch
# authority. Installed for Aug 18, 20, 23 at 09:00.
#
#   sbatch --account=supercomplex --begin=2026-08-18T09:00:00 \
#     scripts/claude_review_v5w2.sbatch
#
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$ROOT"
if [[ -f "$ROOT/.campaign_env.sh" ]]; then
  set -a; source "$ROOT/.campaign_env.sh"; set +a
fi
export SSL_CERT_FILE="${SELFUPDATE_SSL_CERT_FILE:-/fs/agustina/arivero/supercomplex/.local/lib/python3.11/site-packages/certifi/cacert.pem}"

SESSION="111e0ae6-a907-48a8-a4a4-dd6c96a0589d"
CLAUDE="/fs/agustina/arivero/supercomplex/.local/bin/claude"

echo "==== claude v5w2 review start $(date -Is) session=$SESSION ===="
"$CLAUDE" -p --resume "$SESSION" --dangerously-skip-permissions \
"Scheduled v5-week-2 self-review (installed 2026-08-16; agustina_thin node,
full permissions, HARD 60-minute Slurm wall; owner likely AFK). The plan you
are executing is WEEKEND_HANDOFF.md section 'CAMPAIGN v5-WEEK-2 PLAN'.
Determine the phase from today's date and squeue/sacct state, then act:
1. Review every v5w2_* job since the last checkpoint: sacct states, stage
   logs, metrics.jsonl. Diagnose and patch ANY failure first (venv, OOM,
   new-flag bugs) and relaunch lost arms immediately — idle H100 lanes are
   the enemy.
2. Write verdicts into WEEKEND_HANDOFF.md: peak recall vs each run's own e0
   baseline, evals-above-baseline, argmax erosion slope, arc damage,
   backprop_count depth distributions for gated arms, and for --train-norms
   /--dora arms whether the new parameters helped or destabilized.
3. Phase actions (Aug 18 = review 1): promote the best TWO arms per lane to
   300-epoch longs (--epochs 300 --eval-every 10, same pinned knobs,
   run-name suffix _long), chained afterany per lane. Guardrail: an arm
   must beat the incumbent screen window (sustained 0.1878 @e36-40,
   dcos+topk_abs) or show a qualitatively healthier argmax arc.
   (Aug 20 = review 2): launch best-loss x best-capacity combined long, a
   seed-43 replication of the champion, and a 500-epoch horizon run; if
   nothing beats the incumbent, spend the slots on the L54 question
   (implement an epoch-frozen topk_abs threshold gate in trainv5.py).
   (Aug 23 = review 3): final verdicts, regenerate the joint campaign PDF
   (scratchpad script pattern from the Aug-16 session; publish as
   runs/v5_campaign_report/report.pdf force-add), cancel leftover jobs,
   release the nodes.
4. Commit and push everything (push from a login node if the thin node
   lacks the https helper); sync memory files.
The wall kills you at 60 minutes: verdicts and relaunches first, polish
never. Never leave a lane idle if any planned arm is unlaunched. End with a
clean tree; then release this node with: scancel \$SLURM_JOB_ID."
rc=$?
echo "==== claude v5w2 review end rc=$rc $(date -Is) ===="
exit $rc
