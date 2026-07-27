#!/bin/bash
#SBATCH --job-name=claude-review
#SBATCH --partition=agustina_thin
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=runs/sbatch_claude_review_%j.out
#SBATCH --error=runs/sbatch_claude_review_%j.err
#
# Scheduled self-review (owner, 2026-07-27): resume the v5-campaign Claude
# session on the thin queue so it can review the paced experiments and
# reroute them with full authority. Installed for Aug 1, 6, 12 at 09:00.
#
#   sbatch --account=supercomplex --begin=2026-08-01T09:00:00 \
#     scripts/claude_review.sbatch
#
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$ROOT"
if [[ -f "$ROOT/.campaign_env.sh" ]]; then
  set -a; source "$ROOT/.campaign_env.sh"; set +a
fi
export SSL_CERT_FILE="${SELFUPDATE_SSL_CERT_FILE:-/fs/agustina/arivero/supercomplex/.local/lib/python3.11/site-packages/certifi/cacert.pem}"

SESSION="2f2bd81e-6f27-4944-89b3-c5ddf5d287ed"
CLAUDE="/fs/agustina/arivero/supercomplex/.local/bin/claude"

echo "==== claude review start $(date -Is) session=$SESSION ===="
"$CLAUDE" -p --resume "$SESSION" --dangerously-skip-permissions \
"Scheduled self-review checkpoint (installed 2026-07-27; you are resumed on
an agustina_thin node with full permissions and a HARD 60-minute Slurm wall). The owner is likely
AFK. Do, in order:
1. squeue/sacct review of every v5 paced arm and v4 A4B job that ran since
   your last checkpoint; read their metrics.jsonl and stage logs.
2. Write verdicts into WEEKEND_HANDOFF.md per its ANALYSIS PROTOCOL —
   especially: peak recall vs e0 baseline, epochs-above-baseline, erosion
   slope, backprop_count depth distributions for gated arms (topk_abs vs
   relative — the tail-ban evidence), surprise_ema convergence events, and
   the drift/span/residual profile decomposition.
3. REROUTE with full authority: cancel arms made redundant by results,
   resubmit remaining begin-time slots with better-informed args (same
   pacing: one arm per ~2 days, 24h caps, auto-abort protected), preserving
   the owner's two long-arm concepts (topk_abs / surprise_ema).
4. Publish any new runs/*/report/report.pdf the pdf-only way; commit and
   push everything; sync memory files.
5. If something is broken (crashed arms, stuck queue, code bug), diagnose
   and patch trainv5.py per its tripwire discipline before relaunching.
The wall kills you at 60 minutes: commit early and often, verdicts first,
reroutes second, polish never; end with a clean tree and an updated handoff. Do not start interactive waits; use begin-time scheduling."
rc=$?
echo "==== claude review end rc=$rc $(date -Is) ===="
exit $rc
