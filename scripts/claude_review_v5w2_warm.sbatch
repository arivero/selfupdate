#!/bin/bash
#SBATCH --job-name=claude-review-warm
#SBATCH --partition=agustina_thin
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=runs/sbatch_claude_review_warm_%j.out
#SBATCH --error=runs/sbatch_claude_review_warm_%j.err
#
# POST-CLOSURE backstop review (installed by review #2, 2026-08-20): the
# campaign is CLOSED; the only pending science is the teacher-ceiling probe
# (438230) and the warm-start fade test (438302), which wait for a free
# H100 behind other users. Session monitors die with each session, so this
# Slurm job is the durable check on their outcome.
#
#   sbatch --account=supercomplex --begin=2026-08-21T09:07:00 \
#     scripts/claude_review_v5w2_warm.sbatch
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

echo "==== claude v5w2 warm-verdict review start $(date -Is) ===="
"$CLAUDE" -p --resume "$SESSION" --dangerously-skip-permissions \
"Post-closure backstop review (thin node, 60-min wall, owner may be AFK).
Campaign v5w2 is CLOSED (WEEKEND_HANDOFF 'CAMPAIGN v5w2 CLOSED'); do NOT
launch new training arms. Tasks: (1) sacct/squeue for 438230 (teacher
ceiling probe) and 438302 (v5w2_f54_warm120). (2) If the probe ran, read
runs/v5_teacher_ceiling/ceiling.json: teacher_uncensored ~0.95+ validates
the eval decode path; much lower = eval-path artifact — document either
way in WEEKEND_HANDOFF 'Teacher-ceiling calibration'. (3) If the warm run
ran: FIRST check its e0 eval reproduces the source run's e120 row (~quij
0.1924, r64_h400 metrics.jsonl) — mismatch means --init-adapter is broken
(document, kill if still running). Then the fade verdict vs the churny
control (r64_h400 e120-320 faded to 0.176): warm fixed:54 HOLDS = churn
caused the fade; FADES = intrinsic to the stored state. Write the verdict
+ content-level decomposition (scaffold-stripped LCS from recall_texts,
script pattern in the Aug-19 transcript) into WEEKEND_HANDOFF. (4) If
either job crashed, diagnose, fix, resubmit (warm needs --lora-r 64
--lora-alpha 128 to pass the init-adapter gate). (5) If both still pend
behind other users, one-line note, nothing else. Commit+push everything,
timestamp from date, end with clean tree, then: scancel \$SLURM_JOB_ID."
rc=$?
echo "==== review exit=$rc $(date -Is) ===="
exit $rc
