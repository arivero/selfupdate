# Multi-Agent Code Review — 2026-07-10

Reviewer: Claude (Fable 5) driving four parallel review subagents, one per
surface, each instructed to verify every finding against the code before
reporting. Scope: the whole Python surface (~215k tokens / 110 files; see
TOKEN_REPORT.md), all 118 configs, 13 queue TSVs, and the then-uncommitted
2026-07-10 eval diff. Companion docs: FableReviewBy55xh.md (2026-07-05
review), docs/fable_review_status_2026-07-07.md (its status tracker).

Same-day fixes landed in commits `0cdd526` (eval), `800e546` (report +
ALIA artifacts), `723e7af` (validator, certified 13/13 in
runs/certify_gate_20260710_review.log), `9aa3a62` (docs/issues). The
still-open items are tracked in issues.md "Open review findings"; this
file is the full finding record.

## Executive verdict

The publication-critical core is sound and multiply locked. All four
agents independently confirmed: frozen vocabulary (four locks + save-time
tripwire, verified across lora_fused / full_resident / full_offload),
teacher-sourced targets (no `cross_entropy` in train/; `readout_source`
hard-raises on anything but teacher_kl), depth uniformity (no
depth-indexed weight vector exists in the code at all), window gradient
isolation (including `_sliding_windows_dedup` graph-lifetime bookkeeping),
B=1 padded-batch bit-exactness, offload-Adam paging, and hot-loop sync
discipline. Zero live constraint violations in any config or queue.

The defects clustered in exactly two places: (a) the validation
perimeter — knobs that silently did nothing (the repo's own knob-flow-law
bug class); (b) the new eval diff. Nothing was wrong in the loss math or
the training walk itself.

## 1. Constraint-compliance sweep (agent 1)

`scripts/audit_configs.py`: passed, exit 0. All 118 configs individually
extracted and checked. Verdict groups:

- `base.yaml`: k=1 local default, no readout knobs — LEGAL.
- 28 `clean_*` arms + slide/lens/nmse arms: sliding conn 2/4/8, stride 1,
  readout==conn, teacher_kl pinned, hidden weight 1.0 — LEGAL (method).
- `lw_r_slide8kl`, `ragchannel`, `scramble`, 4× `pp2diag_*`:
  `window_hidden_weight: 0.0` ablations — LEGAL; verified in code that
  zeroing affects ONLY the top readout window, body sliding windows keep
  full uniform credit (layerwise.py window walk), matching the crown
  lineage snapshot (`tail_hidden_weight: 0.0` under old names).
- 23 `mlt_*slide2/3*` arms (ALIA/Gemma4/gpt-oss/Qwen3.5/Mistral incl.
  MoE `_tf`/`_ra` modes): LEGAL; router_aligned arms pin
  moe_router_weight (validator-enforced).
- 6 `mlt_*k1local_lora`, 13 strict/local arms, 5 teacher_censored arms,
  8 `data_*`, 9 `eval_*`, 12 `configs/teacher_references/*`: LEGAL.

No occurrence of `tail_ce_*`/`lens_ce_*`/`task_label` in any active
config; no readout source other than teacher_kl; grandfathered tailpure
config deleted; `scripts/train.py` hard-exits on non-layerwise method.
Frozen-vocab locks: LoRA targets exclude embed/lm_head (lora.py),
windows root at detached inputs (test_conn_window asserts nil grads on
embed/head), `check_vocab_frozen` fingerprints embed/norm/head at every
save (runtime.py).

Hygiene items found: the purged word "gold" survives in
retention_eval.py / surprise_probe.py / cross_report.py / make_figs.py
(eval-side role, illegal vocabulary); 4 dead PENDING rows in
scripts/queue.tsv referencing deleted configs (3 xs_* + crown17, the
known owner-decision item); queue TSVs sit outside the audit perimeter,
and a train entry pointing at an old run-dir config snapshot would have
legacy keys silently dropped by load_config (silent-fork trap).

Guard gaps (worst first): (1) tail-only ban was gated on
`run_class == "method"` — an ablation-class config could dispatch a
tail-only arm [FIXED `723e7af`]; (2) queues outside audit perimeter
[open]; (3) `readout_weight` without window silently ignored [FIXED];
(4) depth-uniformity enforced structurally but OLD_KEYS blacklist is
name-based [open]; (5) lexicon purge unenforced by any guard [open].

## 2. Core trainer (agent 2)

Reviewed line-by-line: layerwise.py, runtime.py, blocks.py, losses.py;
lighter: moe.py, lora.py, tuned_lens.py. `torch.autocast("cuda:0")`
empirically tested (accepted in torch 2.11 — the OnlineTeacherSource
pattern is NOT a bug).

Major (both FIXED in `723e7af`):
- H1 `anchor_kl_weight` silently ignored on every schedule except
  summed (`_make_anchor` wired into `_train_summed` only): a
  mixed+anchor config trained with no anchor and no error.
- H2 `conn_stride` domain unvalidated: any value other than 1 fell into
  the disjoint branch — `conn_stride: 2` silently trained different
  credit assignment. docs/windows.md defines only strides 0 and 1.

Medium:
- H3 `readout_weight > 0` with `readout_window_blocks == 0` trained no
  readout (readout0 = n+1 unreachable) yet classified as a readout arm;
  also readout window on `sequential` silently ignored [FIXED].
- H4 tail-only hard stop enforced only for method class [FIXED,
  including the `hidden_loss='zero'` disguise].
- `router_aligned` + `window_dedup`: `_sliding_windows_dedup` never
  drains `pending_router_loss()`, so the combo dies deep in item 1 with
  a misleading "graph leak" tripwire instead of at validation [open,
  issues.md #6].

Minor/latent: `tasks_eval` never calls `model.eval()` (matters when a
dropout arm appears); stale `_shared_kv_states` in `_censored_item` on
the LoRA path (harmless for every current architecture, wrong-length
teacher KV for a future shared-KV family); `teacher_censored` + PP would
crash cross-device at item 1 (fail-loud, combo unused); docs/runtime.md
overstates sliding-path activation-residency (peak detached-state
residency is full depth; graphs do follow W); `teacher_rows.clamp_min(0)`
would mis-map the first readout row if a data mode ever had empty mid
(masking.py currently guarantees nonempty); `readout_window_blocks > n`
KeyErrors (nonsense config, unvalidated).

Explicitly clean: frozen-vocab locks across all optimizer plans;
training-target law; depth uniformity; detach discipline including
dedup-walk graph lifetimes (last_use frees each block graph exactly
after its last covering window); full_offload event/stream ordering with
record_stream both directions and end-of-step synchronize; B=1
bit-exactness (no pad rows, pure-copy gathers, prefix-mask avoids
nonzero() syncs, backward scalar equals single-example loss, no dropout,
RNG order unchanged); hot loop free of .item()/.cpu(); MoE tripwires
(no-router-fired, undrained-pending, row-count mismatch).

## 3. Eval stack + 2026-07-10 diff (agent 3)

CRITICAL [FIXED `0cdd526`, artifacts regenerated `800e546`]:
`_score_pairs` in standard_destruction_eval.py assumed right padding;
ALIA-40b's tokenizer ships `padding_side="left"`. Reproduced
empirically: every non-longest option per batch scored on pad/prompt
tokens; the `end > mask.sum()` guard cannot fire (end <= real length).
Both ALIA standard_damage artifacts were corrupted and content-dependent
(not even the checkpoint-vs-epoch-0 delta was paired-safe). After the
fix: teacher arc_easy/arc_challenge/hellaswag 0.52/0.34/0.48 →
0.82/0.56/0.64; checkpoint 0.29/0.26/0.30 → 0.41/0.34/0.35. True LoRA
capability damage at 40B is arc_easy −0.41, not the fake −0.23. All
other fleet tokenizers verified right-padding.

MAJOR:
- Quijote rung conflation [FIXED]: (a) `tasks_report.py` keyed bases by
  (model, "quijote") — every quijote row's epoch-0 column showed the ch8
  value (whichever base-tasks dir sorted last); (b) ch1..ch16 batteries
  aggregated as one comparable column; (c) `evaluate.py` hardcoded
  raw_ch1.txt, so re-evals of ch8/ch16 checkpoints would score their
  best-trained prefix ("checkpoints score on THEIR corpus" re-broken).
  Fix: rung-level corpora (quijote_chN) everywhere, historical v2
  artifacts rekeyed by measured poem_path, scope-dynamic report tables.
- `tasks.py` EOS lookup returns unk id 0 for SentencePiece (Mistral):
  generation never stops; deflates `exact`, 2-3× compute [open, #1].
- `--layer-residuals` adopts model+poem from the checkpoint but
  examples/mask geometry from base.yaml [open, #2].
- `_stage_source` mkdir-lock has no stale-owner detection; a killed
  copy job wedges lanes silently [open, #3].

Minor: stale report header numbers [fixed by regeneration]; damage
prose mislabeled the suite [FIXED]; standard_bases prefers the 3-task
teacher file over a richer 5-bench destruction base (transitional);
dead CLI flags on evaluate.py silently ignored; `training_scope` field
mislabeled under --recall-corpora override; schema v2 dropped "general"
but analyze.py:31 and build_corpus_index.py read it unguarded; only
ARC-Easy is repo-pinned (hellaswag/arc_challenge/wikitext float on HF
revisions; a pinned wikitext file exists but is ignored); /tmp staging
never cleaned (~170 GB/node at full fleet); report denominators are
self-referential (a run missing tasks.json vanishes silently); test
coverage misses base keying and collect() end-to-end.

Verified clean: queue_coverage TSV regenerated and diffed against its
generator (identical modulo already-completed repairs); queue is 100%
eval commands (no constraint exposure); no eval-vs-train contamination;
destruction.py/probes.py span math and seeded pairing; recite.py stop
handling and OOM backoff; layer_residuals math (teacher routing, span
conventions, loss_view norm-at-n); wikitext chunking denominators.

## 4. Data / masking / teacher / config (agent 4)

No critical findings. Majors:
- readout_weight/window hole (same as H3 — independently found) [FIXED].
- `cache.hidden_dtype` is hash-only: the writer hardcodes fp16
  regardless (the knob changes the cache dir hash but never the
  payload), and fp32→fp16 truncation has no finite-check — 8B+ deep-layer
  outlier channels above 65504 would cache as inf and poison every run
  on that cache. Latent (default matches hardcode) [open, #4].
- `find_poem_spans` lacks word-boundary anchors: censoring can start
  mid-word ("aqu|el que mira…"), and one inserted comma lets a
  near-verbatim verse escape whole-verse censoring — weakens the
  thinking_selective premise [open, #5].

Minors: continuation/maieutic window ranges are off by one — the last
verse is unreachable in any continuation/maieutic answer (715-verse
poem, w=12 s=4: verses 714-715 appear only in sect-/full- tasks);
exposure-deficit class; fix belongs in a v-next dataset (v1 is
byte-identity-guarded). Config merge is one level deep (an experiment
overriding one key of a nested dict silently resets its siblings to
dataclass defaults — latent). `adapt_records` probes only records[0]
(mixed-template jsonl trains later records on the wrong format
silently). `_matches` KeyErrors on legacy records before the curated
"rebuild examples.jsonl" error can fire. Trace harvest stops only on
</think> (malformed traces freeze turn-boundary tokens inside the
privileged block, unwarned). Dataset/cache span check omits t0 /
position_gap (free to check; narrows a tokenizer-drift window).
rag_tool/thinking builds drop the corpus style's system prompt
(identity for shipped verse datasets).

Clean: masking alignment core (segment-wise tokenization contract,
token-identity asserts, gap accounting — pinned by test_alignment +
test_thinking_selective); dataset collate/mask discipline (readout_index
arithmetic, prefix-valid invariant, per-example denominators exclude pad
rows, seeded bucket sampler); teacher cache layer convention
(writer hidden_states[L], L=1..n ↔ reader loss_view norm-only-at-n;
hash covers model+mode+compaction+examples-bytes+schema, enforced at
train time); chatfmt template derivation and loud foreign-template
rejection; config sentinels (tail_ce_* keys fully removed → legacy
configs raise "unknown key", stronger than a sentinel; readout_source
UNSET enforced for every windowed run); conftest.py sys.path pinning.

## Test-suite economics (owner question)

126 tests, ~51 s warm / ~149 s cold; slowest single test 4.7 s. No
single hog — the cost is ~10 GPU modules each loading Qwen3-0.6B
separately. Proposals in issues.md: session-scoped shared stack fixture,
`slow` marker for a fast certify lane, replace torch.jit usage in the
online-teacher path (14 deprecation warnings; removal risk on the next
torch bump).
