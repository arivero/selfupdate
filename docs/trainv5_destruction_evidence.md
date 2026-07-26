# trainv5 destruction evidence — the two failed runs, raw numbers embedded

*(2026-07-27. Written so this evidence survives the periodic `runs/` sweeps:
the source `metrics.jsonl` live in gitignored run dirs
`runs/trainv5_g31b_selfdistill_lr1e4_destroyed/` and
`runs/trainv5_g31b_selfdistill_huber_anchor_destroyed/`; every number a
future agent needs is reproduced here verbatim.)*

Both runs: google/gemma-4-31B-it, LoRA r32/a64 on all 410 text-decoder
Linears, layerwise law (detached block inputs), huber local loss at answer
rows, 2071 items/epoch, AdamW, clip 1.0. Trainer: `scripts/trainv5.py`
pre-`f35f74a` — **the L59 column is contaminated** in both runs (the
last-layer target was the post-final-norm state, independent-review finding
f1); L0–L58 values are valid.

## Run 1 — lr 1e-4, NO anchor (job 423270)

| eval | argmax | KL_eval | CE_eval | arc | recall mach/quij |
|---|---|---|---|---|---|
| e0 | 0.4290 | 11.14 | 11.16 | 0.330 | 0.173 / 0.180 |
| e1 | 0.0000 | 122.65 | 122.66 | 0.190 | 0 / 0 |
| e2 | 0.0000 | 163.52 | 163.52 | 0.200 | 0 / 0 |
| e3–e5 | 0.0000 | 160→153 | — | 0.21–0.26 | 0 / 0 |

Epoch telemetry (mean local loss; L58 valid, L59 contaminated):
ep1 0.0444 (L58 0.213, L59* 0.279) → ep6 0.0280 (L58 0.136, L59* 0.269).
Destroyed inside epoch 1; loss fell while the model died.

## Run 2 — lr 1e-5 + self-anchor w=1.0/64 rows (job 423293)

| eval | argmax | KL_eval | CE_eval | arc | recall mach/quij |
|---|---|---|---|---|---|
| e0 | 0.4290 | 11.1427 | 11.1569 | 0.330 | 0.173 / 0.172 |
| **e2** | **0.1332** | **9.8960** | **9.9018** | 0.200 | 0.059 / 0.063 |
| e4 | 0.0000 | 66.50 | 66.50 | 0.160 | 0 / 0 |
| e6 | 0.0000 | 121.71 | 121.71 | 0.200 | 0 / 0 |
| e8 | 0.0000 | 141.59 | 141.59 | 0.190 | 0 / 0 |

Epoch telemetry:

| ep | mean loss | L58 | L59* (contam) | mean grad norm | adapter L2 |
|---|---|---|---|---|---|
| 1 | 0.0764 | 0.328 | 0.579 | 0.00263 | 66.278 |
| 2 | 0.0628 | 0.278 | 0.576 | 0.00227 | 66.505 |
| 4 | 0.0535 | 0.233 | 0.568 | 0.00090 | 66.788 |
| 8 | 0.0443 | 0.192 | 0.562 | 0.00068 | 67.221 |

## The load-bearing observation: the e2 BLURRING PHASE

At run 2's e2, **CE/KL moved TOWARD the teacher (11.16 → 9.90)** while
argmax collapsed (0.429 → 0.133) and recall halved. The student's answer-row
distributions were, on average, getting *closer* to the teacher's — they
spread toward the teacher's support while losing their sharp modes — and
only afterwards (e4) did the distribution snap into degenerate collapse
(CE 66 → 142).

**Why this matters (owner thesis, 2026-07-27): the gradient direction is
right; destruction is an OVERSHOOT phenomenon, not a wrong-direction
phenomenon.** This is the best evidence so far that a non-destructive
version of the same update rule accumulates alignment — i.e., that the
viable band between "destroys the model" and "v4's frozen safety"
(argmax flat at 0.556 for 50 epochs, nothing learned) is non-empty. The
research program is to find that band: metric (vocab_mse), anchor strength,
LR, surprise gating, positions.

Supporting facts from the same tables:
- Destruction is not weight explosion: adapter L2 moved 66.28 → 67.22
  (init-dominated; the change is ~1.4%), grad norms ~0.001, clip never
  binding. Tiny coherent drift compounds through the 60-layer composition.
- The mean local loss is a LIAR: it fell monotonically in both runs while
  the model died. Read the profile by depth; never the mean.
- The anchor + 10x lower LR slowed destruction (e1 CE 122.7 in run 1 vs
  e2 CE 9.9 in run 2) — the defenses bend the curve; they did not, alone,
  reach the band.

## Caveats for re-analysis

- L59 rows in both runs are contaminated by the f1 target bug (fixed in
  f35f74a); do not cite them. L0–L58 are valid.
- Neither run used a layer gate ('all' mode), so gate-selection was NOT
  affected by the f1 contamination in these two runs (it would have been —
  see the Opus verification note in WEEKEND_HANDOFF.md).
- Both runs predate `--positions aligned`; the ~215-position RoPE shift
  between teacher targets and student states is present in these numbers.
- Eval CE/KL here predate the softcapping fix (f3): absolute values are
  computed without Gemma4's final_logit_softcapping=30, so they overstate
  large divergences; trends and comparisons within/between these two runs
  remain valid (same convention), but do not compare these absolute CE
  values against post-f35f74a runs. Measured anchor for the discontinuity
  (Opus review): the SAME untrained model's e0 KL_eval is 11.143 under the
  old convention and 5.709 under the new one (job 423314).
- First clean-run comparison point: run `trainv5_g31b_vmse` (job 423314,
  vocab_mse, all fixes).
