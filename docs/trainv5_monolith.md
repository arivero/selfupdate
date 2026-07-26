# trainv5.py — a guided tour of the monolith

*(2026-07-26/27 night. One file, 1040 lines, ~12.5k tokens, three destroyed
models, four tripwires, and one poem it is trying to teach a 31B network to
remember.)*

`scripts/trainv5.py` is pipeline v5, experiment 1: **layerwise
self-distillation with censored context**. It is deliberately a monolith —
no `selfupdate` imports, no shared cache identities, new run names — so a v5
run cannot collide with or silently reuse any v4 artifact. Everything it
needs beyond stdlib is torch, transformers, and peft; everything it consumes
from the repo is read-only (the examples jsonl and the vLLM answers).

## 1. The idea in one paragraph

Take one model. Show it a poem and a question, let it answer — that answer,
and the hidden states it had while answering, are the **teacher** (adapters
off, frozen). Now hide the poem and ask again — that is the **student**
(same weights + LoRA), living in the deployment condition where no retrieval
exists. At the answer tokens, layer by layer, nudge each student block so
that *its* output looks like the teacher's state at the same depth. If this
works, the poem's contribution to the computation migrates into the
adapters: discussing Machado literally makes the model a Machado expert. In
the long-run vision (vN), "hide the poem" generalizes to "mask whatever
distant tokens the model attends to strongly" — continuous, per-user
personalization of adapters.

## 2. The law (and who enforces it)

```
teacher (no-grad, adapters OFF, WITH passage):
    h_t[0] -> block_0 -> h_t[1] -> ... -> h_t[60]      record h_t[L][answer]

student (ONE forward, adapters ON, passage REMOVED):
    h_s[0] -> [detach] -> block_0+LoRA -> h_s[1] -> [detach] -> block_1+LoRA ...
                 |                             |
                 v                             v
        loss_0(y_0[ans], h_t[1][ans])   loss_1(y_1[ans], h_t[2][ans])   ...
```

Every block's **input is detached** by a forward pre-hook, so block L
transforms the student's real, current trajectory state h_s[L-1] — but its
loss term's graph roots *only* in block L's LoRA. One backward call delivers
every block its purely local gradient. **Cross-block gradient flow is
structurally unrepresentable**, not merely forbidden: there is no edge in
the autograd graph to carry it. This is the project's *nombre de pila* —
every training is layerwise, always.

Four mechanisms enforce the laws at runtime (tests were abolished
repo-wide in July; the code defends itself instead):

| tripwire | when | what it catches |
|---|---|---|
| approval-gate asserts | startup | any trainable param outside decoder-block LoRA; a trainable embed/head/norm |
| layerwise isolation cert | first batch | backward ONE middle block's term alone; assert gradient landed in exactly that block's 14 LoRA tensors, zero leaks |
| detached-input guard | every batch, no sync | a block input carrying `requires_grad` |
| frozen-vocab fingerprint | every epoch | embed_tokens / lm_head / final norm moving by even one bit |

Output-level KL/CE against the teacher exist **only** inside `evaluate()`
under `no_grad` — training on output logits is forbidden (that would be a
commodity fine-tune, of which the internet has hundreds; the layerwise law
is the project).

## 3. War stories — how each scar got here

The monolith was written in one day and immediately taught us three lessons,
each now carved into the code:

**Run 1 (`..._lr1e4_destroyed`).** LoRA suffix targets (`q_proj`, …) also
matched Gemma-4's *vision tower*, whose projections PEFT cannot wrap —
instant crash. Fixed by enumerating exact module names inside the resolved
text stack (410 Linears; the ten full-attention layers have no `v_proj` at
all — KV-sharing). Relaunched at AdamW lr 1e-4: **one epoch lobotomized the
model** (argmax 0.43→0.0, CE 11→123, ARC to chance) while the mean local
loss *fell*. Only ~38 answer rows per item were constrained; 245M adapter
params happily wrecked every other position to satisfy them. Lesson → the
**self-anchor**: at 64 fixed sampled *prompt* rows of the same censored
sequence, each block's output must stay near the adapters-OFF base states.
Learn the passage where there is signal; change nothing elsewhere. (Also the
continuous-learning requirement: personalization must never lobotomize the
base.)

**Run 2 (`..._huber_anchor_destroyed`).** lr 1e-5 + anchor: destroyed
*slower*, with a fascinating e2 phase — CE fell toward the teacher
(11.2→9.9) while argmax collapsed (0.43→0.13): the distribution blurred
toward the teacher's support, losing its modes, then snapped (CE 66→142).
Telemetry acquitted weight explosion: adapter L2 moved 66.28→67.22
(init-dominated), grad-norm ~0.001, clip never binding. Tiny coherent drift,
compounded through 60 layers. The depth profile was the tell: L45-59 carried
0.09–0.58 of relative error while 50 shallow layers averaged the mean down
to a soothing 0.03. **Huber weighs all 5376 channels equally; the frozen
head does not.** Lesson → the loss *menu* matters (`vocab_mse` measures
distance in the unembedding's own metric W^T W), the destruction warning,
the auto-abort, and the analyst rule: *read the surprise profile by depth,
never the mean*.

**Run 3 (in flight as of this writing).** `vocab_mse`, single-variable vs
run 2. The referendum on the metric-mismatch hypothesis.

## 4. Tour by section

**Data reconstruction (`build_items`).** No dataset builder, no cache
identity: the vLLM responses carry exact `prompt_token_ids` and answer
`token_ids`; the examples jsonl carries the `privileged` passage text.
Censorship is *text surgery*: decode the prompt, cut the passage string,
re-encode. Two hard gates: the decoded prompt must re-encode to the exact
stored ids (round-trip guard — 2071/2071 pass), and the passage must be
found (else abort). An exhaustive audit confirmed the cut removes exactly
the system-turn block where the system informs the LLM of the literal text —
persona and question always survive; the only passage fragments remaining
are the cue lines the *question itself* legitimately quotes.

**Capacity check.** Owner rule: each parameter stores ~3 bits of compressed
text. gzip the Machado+Cervantes corpora, compare to trainable-params × 3.
r32 across 410 matrices = 244.9M params ≈ 650× the corpus — capacity is
never the binding constraint (run 1 proved the binding constraint is
stability, not capacity).

**`resolve_stack` / LoRA targeting.** Finds the text decoder inside
multimodal composites (`model.language_model` for Gemma-4) and enumerates
exact Linear names under it. Suffix matching is banned by scar tissue.

**`collate` and `--positions aligned`.** Right-padded teacher-forced
batches, with the answer's *predictive* rows tracked per item (the state at
position t predicts token t+1). The `aligned` mode numbers the censored
student's RoPE positions *as if the passage were still present* — because a
token nobody may attend to differs from a deleted token only in position
numbering, this makes remove-view positionally equivalent to v4's
`flow_mask` without any 4D-mask surgery, and removes the ~215-position shift
between teacher targets and student states (a candidate confound in the
deep-layer residual).

**`LayerwiseTaps`.** The heart. A context manager that, for the duration of
one student forward, detaches every block input (and keeps it — it doubles
as the `delta_cosine` anchor) and collects every block output. Installed
only in the training step; eval and generation run the unhooked model.

**`LocalLoss`.** The menu — the project's active research axis: `huber`
(teacher-RMS units), `nmse`, `cosine`, `delta_cosine` (increment vs
increment from the block's own input), `vocab_mse` (Gram-metric; the
historical recall recipe's loss). Formulas copied verbatim from the module's
`losses.py`. The Gram matrix W^T W is built once, chunked, and cached per
GPU.

**Layer gates.** The forward computes all 60 local losses anyway (owner
insight: per-block backward is independent, so skipping a backward is free
money). Modes: `all`; `topk:N` (each step, only the N largest losses
backprop — `topk:1` is both a long-run arm and the maximal drift limiter);
`minfrac:F` (skip layers below F× the step's max); `surprise_ema:F` (owner
concept: surprise as *prediction error* — each layer keeps an EMA of its own
loss and trains only when the current loss exceeds F× its expectation).
Selection values are gathered with ≤1 sync per GPU (hot-loop law), and the
loss normalizes by n_layers so every gate mode trains selected layers at the
same effective LR. `backprop_count` per layer is logged — it doubles as the
load-balance measurement for a future PPP4 port, where a global top-k would
need a per-step all-gather.

**Caches.** The teacher is frozen, so its answer-row hiddens (~47 GiB) and
the base-model anchor states (~80 GiB) are computed once and parked in host
RAM; epochs 2+ skip both no-grad forwards. Epochs measured at 117–131 s
after warm-up — which is what makes tonight's iterate-every-80-minutes
science possible.

**Telemetry (`metrics.jsonl`).** Per epoch: mean local loss, the per-layer
`surprise_profile` (the same quantity the loss optimizes — where it starts
high and falls fastest is empirically *which layer remembers poetry*),
`backprop_count`, teacher-cache hit fraction, mean pre-clip grad norm,
adapter L2, anchor weight. Per eval: per-corpus generative recall (word-LCS
vs the teacher's answer, fixed subset every eval), arc_easy (damage guard),
teacher-forced `student_argmax` (v4's frozen-at-0.556 metric — direct
comparison), and eval-only KL/CE with `optimizer_weight: 0.0` stamped in the
row.

**Auto-abort.** Two consecutive evals with argmax below half of epoch-0 →
save checkpoint, log `aborted_destruction`, exit cleanly. Scheduled arms can
no longer burn a 24 h slot on a corpse, and the cluster gets the node back.

## 5. Knobs (the ones that matter)

| flag | default | why |
|---|---|---|
| `--local-loss` | huber | the research axis; `vocab_mse` = head-metric distance |
| `--lr` | 1e-5 | 1e-4 destroyed a model in one epoch |
| `--anchor-weight/-rows` | 1.0 / 64 | the stay-yourself term at prompt rows |
| `--layer-gate` | all | `topk:1`, `minfrac:F`, `surprise_ema:F` |
| `--positions` | natural | `aligned` = flow_mask-equivalent numbering |
| `--teacher-cache` | cpu | epochs 2+ skip the teacher forward |
| `--epochs / --eval-every` | 40 / 2 | epochs are ~2 min warm |
| `--dry-data` | — | CPU-only data-gate (the sbatch runs it before touching GPUs) |

Launch: `sbatch --account=supercomplex scripts/trainv5.sbatch [args...]` —
the sbatch stages the model to /dev/shm, runs the data gate, then the
trainer. Runs land in `runs/<run-name>/` with per-eval checkpoints.

## 6. Where this is going

The v4↔v5 symmetry is the finding so far: v4 fed each block the teacher's
own context and the local objective was *trivially satisfiable without
learning* (argmax flat at 0.556 for 50 epochs, every loss, every rank); v5
feeds each block its own censored trajectory and the naive objective is
*satisfiable while destroying the model*. The viable science is between
those poles, and the search space is exactly this file's knobs: the loss
metric, the anchor, the gates, the positions, the step size. The scheduled
arms (one per two days, ≤24 h each, auto-abort protected) walk that space
until mid-August.
