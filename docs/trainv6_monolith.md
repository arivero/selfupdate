# trainv6.py — experimental causal-training monolith

scripts/trainv6.py is a deliberately isolated successor to the experimental
v5 trainer. It is not part of src/selfupdate, imports no repository Python
package, and does not change the supported pipeline-v4.6 law.

## Presets and laws

Every invocation must pin one model/artifact preset and one law:

    PY=/tmp/$USER/selfupdate-venv/bin/python
    $PY scripts/trainv6.py --preset gemma4_31b \
      --method causal_residual --run-name v6_g31b_cr_s17 --dry-data
    $PY scripts/trainv6.py --preset qwen36_27b \
      --method partial_teacher --run-name v6_q27b_pt_s17 --dry-data

- causal_residual replays each frozen block twice on the same detached
  current input, with passage access visible and blocked. The trainable
  blocked block is matched to the visible output.
- partial_teacher exposes the passage only at the preset retrieval layers
  in the frozen teacher trajectory, then matches a fully blocked student.

Gemma blockage masks passage keys for post-passage queries. Qwen applies the
same mask to its 16 softmax layers and zeros passage content at the token
mixer of its 48 GatedDeltaNet layers while retaining the sequence clock and
recurrent decay. A mandatory launch certification replaces the passage and
requires blocked answer states to remain invariant.

Both laws update every decoder layer with the same normalized hidden Huber
weight and the same bounded Jensen–Shannon weight through a frozen copy of
the model's final norm and LM head. The default weights are 1.0 and 0.05.
The prompt anchor defaults to 1.0. Embeddings, final norm, and vocabulary
head are fingerprinted and never optimized.

## Metric contract

runs/<run>/metrics.jsonl uses schema v6_metrics_v1.

- CE-eval-loss, KL-eval-loss, teacher/student token acceptance, and
  exact-sequence rates cover every stored teacher answer token in all 2,071
  training items after every completed epoch. Every row says evaluation only,
  no backward, and optimizer weight zero.
- Generation uses ordinary passage-removed natural positions as the primary
  condition. A fixed stratified panel runs at ordinary evaluation boundaries;
  epoch zero, final, and promotion candidates run the whole set.
- The exact-path uncensored teacher ceiling is produced at epoch zero.
  Sufficient answer-length and 96-token deployment-budget scores are emitted
  from the same generations. Aligned-position generation is diagnostic only.
- Canonical corpus targets, not model-specific teacher wording, define the
  primary content score. Per-item artifacts include full and content LCS,
  recitation, first-content-token correctness, longest prefix, first
  divergence, lengths, finish reason, corpus/chapter, and task kind.
- ARC-Easy, ARC-Challenge, and HellaSwag use the fixed vendored subsets.
- Epoch rows include per-layer hidden/JS/anchor loss, causal-effect magnitude,
  adapter delta, and separate objective gradient norms/share/cosine.

Paired item bootstraps compare each evaluation to the same run's epoch zero.
A run cannot stop for scientific flatness before 12,000 observed items.
Promotion requires a whole-set content gain of at least +0.03 with the
interval above zero, a new recitation or +0.03 prefix gain, preserved
teacher-forced acceptance and standard capability, and a second training seed.

## Gates and launch

Pure metrics and model-specific data gates:

    $PY scripts/trainv6.py --self-test-metrics
    $PY scripts/trainv6.py --preset gemma4_31b \
      --method causal_residual --run-name unused --dry-data
    $PY scripts/trainv6.py --preset qwen36_27b \
      --method causal_residual --run-name unused --dry-data

--preflight limits the run to two items and one optimizer step while still
executing intervention, locality, frozen-vocabulary, evaluation-coverage, and
artifact gates. It is a runtime smoke, never scientific evidence.

Real launches use scripts/trainv6.sbatch. The wrapper stages the selected
model and passes the identical argument array through --dry-data and the
trainer, so a Qwen launch cannot accidentally validate Gemma defaults.
Before either command it also compiles the monolith and runs
--self-test-metrics. A queued launch may export SELFUPDATE_V6_CODE_SHA256;
the wrapper then refuses to run if the monolith changed while the job waited.
