# v5rs RAG-compaction interim report

Generated 2026-07-12 from completed `v5rs_0p6b_*` checkpoints only. This is
an interim style comparison, not a winner declaration: the chapter arms have
not completed and the `lens_kl` pad-random mate is still running.

## What is being compared

The teacher sees a RAG passage delivered through the repaired `rag_system`
conversation. The student sees either:

- **remove** — the passage is absent; or
- **pad-random** — the same span is replaced by deterministic distinct
  ordinary-vocabulary tokens, retaining length/positions but carrying no
  passage information.

Both members of a pair use Qwen3-0.6B, six epochs, bucketed batch 8, the
same window-RAG data, a uniform sliding window, and teacher-sourced readout.
Recall is word-level LCS recall (spaces, line breaks and punctuation do not
count). `standard mean` averages ARC-Easy, ARC-Challenge and HellaSwag at
100 examples each; higher is less damage.

## Completed window pairs

| loss / slide | compaction | Machado | Quijote 1 | Quijote 4 | recall mean | standard mean |
|---|---|---:|---:|---:|---:|---:|
| Huber / 2 | remove | .102 | .129 | .113 | .115 | .433 |
| Huber / 2 | pad-random | .103 | .095 | .101 | .100 | .453 |
| Jacobian lens KL / 1 | remove | .096 | .147 | .121 | .121 | .430 |
| Jacobian lens KL / 1 | pad-random | .118 | .147 | .103 | .122 | .447 |

## Pairwise reading

| paired comparison | Δ recall mean (pad − remove) | Δ standard mean | reading |
|---|---:|---:|---|
| Huber slide-2 | −.015 | +.020 | Padding costs mainly Quijote-1 recall, while slightly improving benchmark retention. |
| Jacobian lens KL slide-1 | +.001 | +.017 | Essentially tied on mean recall; padding reallocates recall from Quijote-4 to Machado and improves retention. |

The two losses therefore do **not** support one universal compaction winner.
For Huber, removal is better for the present recall objective. For Jacobian
lens KL, pad-random is a retention-favourable near-tie, but the loss of
Quijote-4 means it is not clearly superior.

## Incomplete coverage

| arm | state | implication |
|---|---|---|
| Lens KL slide-1, window/remove | complete: M .095, Q1 .125, Q4 .114, mean .111, standard .440 | Needs its pad-random mate before a style comparison. |
| Lens KL slide-1, window/pad-random | training at capture | Do not compare yet. |
| All chapter/remove and chapter/pad-random arms | not yet complete | No statement about chapter-context compaction is warranted. |
| 1.7B / 4B v5rs arms | not yet complete | Current evidence is strictly a 0.6B result. |

## Practical interim decision

Keep both compaction styles in the scan. Prefer **remove** for Huber when
recall is primary; retain **pad-random** for Jacobian lens KL as the
low-damage counterpart. Revisit only after the Lens-KL mate and matched
chapter pairs complete.
