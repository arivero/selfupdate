# Connected Windows: Precise Semantics

*(Owner specification, 2026-07-04. This document exists because "forward
then backward" can be misread as classical backpropagation that merely
uses a window to manage GPU memory — gradient checkpointing. That is NOT
what this project does. Read this before touching `tail_step`,
`conn_window`, or any schedule.)*

## The defining property

A window [L0..L1] is a **gradient-isolation unit**, not a memory trick:

- The window is rooted at a **detached** input: the backward pass exists
  only inside blocks L0..L1 and **stops at the input of block L0**.
  Nothing propagates below L0, ever. There is no outer graph.
- The frozen vocabulary (embedding, final norm, lm_head) receives no
  gradient from any window, in any scheme.

Classical backprop with activation checkpointing recomputes activations
in windows but the GRADIENT still spans the full depth. Here the
gradient itself is windowed. A model trained with k-windows has never
experienced a credit path longer than k blocks.

## The 2×2 design space (per window)

Two independent choices define every variant:

**Loss placement** — with teacher outputs t_L and student outputs h_L:
1. *Endpoint*: loss(h_L1, t_L1) only. Blocks L0..L1 are all updated,
   credited for producing the right OUTPUT of the window.
2. *Weighted in-window*: sum of w_L · loss(h_L, t_L) for L in L0..L1
   (`tail_hidden_weight` scales this family in the tail; the hybrid tail
   additionally adds the answer-CE, which exists only where logits do).

**Window input** — what h_{L0-1} is:
1. *Student-stream*: the student's own (detached) h_{L0-1} — inputs
   drift as lower blocks train (the `summed` convention).
2. *Teacher-stream*: the teacher's h_{L0-1} (privileged rows censored,
   teacher position ids — the `teacher_censored` convention). Inputs
   are STATIONARY, so windows are independent and embarrassingly
   parallel across depth.

## What is implemented (2026-07-04)

| variant | knobs | status |
|---|---|---|
| student-stream, endpoint, sliding stride-1 | `conn_window: k`, `conn_stride: 1` | implemented (`lw_r_slide*` arms) — every layer's target is the endpoint of one k-window; ALL covered blocks updated; uniform k-deep credit |
| student-stream, in-window sum, disjoint | `conn_window: k`, `conn_stride: 0` | implemented — cheap approximation, credit depth varies inside window |
| student-stream, in-window sum + answer-CE, top window | `tail_ce_blocks: k` | implemented — as the LAST window position this is legitimate; as the ONLY connected window it is a labeled ablation (see CLAUDE.md publication-critical constraints) |
| pure truncated distillation (CE only, no hidden loss) | `tail_hidden_weight: 0` | implemented as ablation `lw_r_tailpure` — the caricature, never the method |
| teacher-stream k-windows | — | NOT implemented; the natural C3 item: extends `teacher_censored` (k=1, stationary, depth-parallel) to k>1 — stationary inputs + k-deep credit + full depth-parallelism |

## Why the top window carries the CE

Not because the tail is special: because logits only exist after block
n. Under sliding windows the CE is simply the extra term available at
the one window position whose endpoint has a decode path. The naming
contract (CLAUDE.md) bounds its weight and requires reporting its
gradient share (`scripts/signal_attribution.py`).
