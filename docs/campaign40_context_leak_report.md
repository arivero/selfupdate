# Campaign40 v4 context-censorship audit

Date: 2026-07-20. Model: `google/gemma-4-31B-it`. Dataset: the complete
2,071-item campaign corpus. Diagnostic commit: `28ec339`; repaired-context
implementation: `c52034d`.

## Question and conditions

The token-span audit first checked all 2,071 records with the Gemma and Qwen
tokenizers.  Segmented tokenization matched one-shot tokenization, every
privileged run was nonempty, and the removed interval contained exactly the
RAG wrapper and passage.  The question, assistant opening, answer, and system
closing tokens were not removed.  The location of the raw RAG tokens is not
the defect.

The GPU probe then compared two block-local conditions with the same detached,
uncensored teacher `h[L]` target and the same initial LoRA block:

1. **Legacy teacher-local:** uncensored teacher `h[L-1]` supplies the query and
   ordinary frozen K/V positions; direct privileged K/V columns are masked.
2. **Fully censored:** an adapters-off trajectory starting at embeddings uses
   `flow_keep` at every preceding block and supplies both the query and context.

No optimizer or parameter write occurs.  Adapter gradient norms are measured
with `autograd.grad`; every condition remains block-local.

## Result

Thirty-two deterministic probe groups produced 464 sampled observations (412
unique examples; 51 examples occurred in more than one seeded group).  Values
below are unweighted means across the 32 probe groups.  Parentheses are
approximate 95% intervals across group means, not independent-item confidence
intervals.  `loss ratio` is fully-censored Huber loss divided by the legacy
teacher-local loss against the identical target.

| block | input relative L2 | input cosine | loss ratio |
|---:|---:|---:|---:|
| 1 | 0.0000 (±0.0000) | 1.0000 (±0.0000) | 1.00 (±0.00) |
| 2 | 0.0530 (±0.0024) | 0.9985 (±0.0001) | 1.83 (±0.04) |
| 6 | 0.0480 (±0.0015) | 0.9989 (±0.0001) | 1.64 (±0.07) |
| 12 | 0.0939 (±0.0037) | 0.9969 (±0.0002) | 7.09 (±0.20) |
| 24 | 0.1451 (±0.0054) | 0.9928 (±0.0004) | 4.99 (±0.09) |
| 36 | 0.2841 (±0.0068) | 0.9645 (±0.0015) | 10.02 (±0.61) |
| 48 | 0.6468 (±0.0227) | 0.8652 (±0.0051) | 6.27 (±0.22) |
| 60 | 0.9348 (±0.0201) | 0.5638 (±0.0167) | 14.47 (±1.25) |

The two inputs are identical at block 1 because both begin at embeddings.
Afterward, ordinary teacher states progressively carry information acquired
from the privileged RAG keys in preceding uncensored teacher blocks.  Masking
the raw privileged columns at the current block cannot remove that information
from the query or from ordinary-position K/V.  By block 60, the fully censored
input is almost one teacher-input norm away and the legacy objective
understates the deployment-proximate same-target loss by roughly fourteenfold.

## Scientific interpretation

This explains the campaign's central anomaly: legacy local losses can fall
while composed censored recall stays flat or declines.  The optimizer is
trained on increasingly easier, RAG-informed teacher context that the deployed
censored student does not receive.  A new loss family (`lens_kl`, cosine,
vocabulary metrics, or residual-update cosine) does not itself repair this
context mismatch and must be labelled legacy/confounded if run on the old
context.

The immediate legal repair is the explicit opt-in
`v4_context_source: flow_censored_teacher`: detached adapters-off censored
teacher `h_c[L-1]` supplies both query and K/V, while the uncensored teacher
`h_u[L]` remains the target.  It preserves one-block gradients and the frozen
vocabulary law.  It is initially supported only by the online, fully resident,
dense-attention route; cache/store, staged, rotating, compressed, and recurrent
paths fail loudly.  Its full-size one-epoch Gemma-31B run is
`campaign40_g31b_context_repair_huber_e1`.

Success is not a lower repaired local loss alone.  The repaired arm must close
the censored deployment-trajectory gap and improve full-corpus recall relative
to the legacy context and a censored-target negative control, with locality,
gradient-share, CE/KL evaluation coverage, and parameter-delta evidence.
