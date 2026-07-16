# selfupdate - layerwise forward distillation of context

The same model plays teacher and student. The teacher prompt contains
privileged context - a RAG passage or a visible `<think>` trace - while the
student receives the same prompt with that context hidden. Training asks each
student block to reproduce the teacher's hidden state at aligned token
positions.

The research target in this branch is strict layerwise forward distillation:
block-local hidden-state learning with no behavioral readout or final-logit
training. A local `lens_kl` objective may use the frozen vocabulary head as a
metric, but it never updates that head or creates credit across blocks. The
former readout runtime is being deleted and remains recoverable from Git
history only.

## The Pierre Menard Program

- Stage 1: memorize *La tierra de Alvargonzalez* (Antonio Machado, 1912) with
  Qwen3-0.6B through larger Qwen checkpoints.
- Stage 2: scale the same masking and layerwise training machinery to
  Don Quijote on 120B-class dense or MoE models.

## Layout

```
configs/            base + layerwise experiment YAMLs
data/poem/          raw.txt + generated examples.jsonl variants
caches/             teacher hidden-state caches (gitignored)
runs/               experiment outputs/checkpoints (gitignored)
scripts/            dataset/cache/train/eval/analysis/scheduler tools
src/selfupdate/     masking, data, teacher cache, layerwise train, eval, utils
tests/              removed 2026-07-11; historical tests remain in Git
```

## Method Notes

- Every example is segmented as `shared_prefix | privileged | shared_mid |
  answer`. The teacher sees all four segments; the student skips
  `privileged`. The aligned span is `shared_mid + answer`.
- Qwen3 uses RoPE with full attention. A constant position offset is
  output-invariant, so teacher/student divergence at aligned positions comes
  from attention into the privileged block: the signal being distilled.
- The student-side privileged block can be removed, replaced by a stub token,
  or position-rebased. Current evidence favors removal.
- Teacher caches store per-layer hidden states only. Online-teacher LoRA runs
  skip the disk cache: adapters off is the frozen teacher, adapters on is the
  student.
- The core loss is hidden matching (champion metric: `vocab_mse`, MSE in the
  frozen vocabulary's coordinates; `nmse` matches it under uniform windows).
  `lens_kl` is likewise a depth-uniform local metric through the frozen norm
  and head: it trains only the intended block and never updates the head or
  crosses blocks. There is no behavioral readout or final-logit training.
  Reference-text cross-entropy is never a training target on this branch
  (eval against the reference text is correct and required); the embedding
  and logits matrix are never trained (Frozen-Vocabulary Principle).

`CE-eval-loss` and `KL-eval-loss` are evaluation-only output distances
collected over every answer token in the whole training-set traversal once
per epoch. They are not validation-subset losses and are NEVER used for
backward or optimization; reports show their coverage and zero optimizer
weight explicitly.

See [docs/hidden_loss.md](docs/hidden_loss.md) for locality proofs and
[docs/scaling.md](docs/scaling.md) for the large-model plan. The typed
answer/token aggregation and future state/attention/routing strategy surface
is specified in [docs/training_pipeline_v2.md](docs/training_pipeline_v2.md).
Completed pipeline-v2 trainings get their atomic individual report with
`scripts/report_v2.py`; see [docs/report_v2.md](docs/report_v2.md).
The online B=1/K=1 successor—with immediate state-free local writes,
student- versus uncensored-teacher trajectories, information-flow masking,
and per-answer causal-history lifetime—is specified in
[docs/training_pipeline_v3.md](docs/training_pipeline_v3.md).

On driver-560 L40S nodes, launch v1/v2 training through
`scripts/l40s_exec.sh`; it already launches Python, so use
`scripts/l40s_exec.sh scripts/train.py ...` without an extra `python`
argument. Launch v3 through `scripts/l40s_train_v3.sh`: it coordinates one
same-runtime epoch-zero teacher pass into `/dev/shm` per node/cache identity,
and concurrent or later arms reuse the atomically published cache. The cu128
container is reserved for nodes with a compatible newer driver. The thin
cu126 dependency layer and its no-second-torch invariant are documented in
`AGENTS.md`. Counterintuitively, these older L40S nodes require the newer
glibc 2.35 userspace: this is not a GPU or driver requirement, but a
requirement of the precompiled `causal_conv1d` Python wheel. The wrapper
starts Python through the glibc 2.35 dynamic loader with an explicit library
path. Merely running `module load glibc/2.35` keeps the old loader, mixes
incompatible libc internals (`GLIBC_PRIVATE` failures), and is not a valid
substitute. The wrapper restores the host library path before child compilers
run so Triton's host `gcc` does not inherit that mixed userspace.
The same wrapper resolves model weights from the RAM-backed offline HF cache.
The standard-damage subsets are consequently vendored under `data/eval/` at
their pinned revisions; `scripts/vendor_standard_eval.py` is the explicit,
one-time online rebuild path rather than a hidden training-time download.
For a concrete annotated `ps auxww` example of a live L40S scheduler and its
glibc-loader/Python workers, including the duplicate-worker warning, see the
L40S Cluster Environment section of `AGENTS.md`.
Model weights and teacher hidden-state caches are staged independently. V1/v2
can copy a durable cache with `scripts/stage_teacher_cache_shm.sh`; v3 instead
regenerates it locally through the coordinated epoch-zero launcher above. An
idle CPU and idle GPU during layerwise training commonly means Lustre-backed
teacher shard faults, not a slow GPU kernel.

## Evaluation generation pipeline v2

Epoch-zero censorship controls and post-training checkpoint recall use
`scripts/teacher_ceiling.py`, which submits the deterministic evaluation
prompts and their individual token budgets directly to vLLM. Pass
`--checkpoint runs/<run>/checkpoint` to evaluate a trained model through the
same prompt construction and scoring path used for its base reference. The
retired Transformers `model.generate` implementation is intentionally not
kept in-tree; it remains recoverable from Git history.

Here, **epoch zero** means the untrained network evaluated in exactly the same
conditions as the later student checkpoints. The separate evaluation in which
the base network receives the original uncensored RAG is recorded historically
as a teacher ceiling, teacher reference, or intact-RAG control. It is not the
epoch-zero checkpoint baseline unless that same uncensored input is also the
declared checkpoint-evaluation condition.

For repeated base-model or checkpoint loads, stage snapshots into Unix tmpfs
with `scripts/stage_hf_cache.sh --shm <org/model>`. Container launches prefer
the completed RAM stage automatically. Direct vLLM launches set `HF_HOME` to
`/dev/shm/$USER/selfupdate-hf-cache`; ordinary safetensors file access then
benefits from kernel-shared resident pages without a custom model object.

## Current Policy

Pareto v2 is closing. Pipeline v3 is the active implementation target: every
write is strict block-local and immediate at one answer/token coordinate,
with no cross-token gradient aggregation or behavioral readout. Dataset v5
and its uncensored teacher targets remain the scientific source. Atomic reports
continue to be produced per run; synthesis groups them by pipeline, model,
loss, censorship, trajectory source, and causal-history policy.

## Historical readout-era findings

Historical readout-era experiments found that storage and readout dissociate.
Hidden matching writes distributed,
redundant storage; behavior comes from bounded sliding connected windows
with uniform k-deep credit — recall arrives by k=4, a clean destruction
battery by k=8 (the connectivity law). The readout is where the
pathologies live: it is template-locked (cured by maieutic dialogue
data) and intrudes on neighbor-genre text ("catastrophic remembering").
Matching the teacher's with-context trajectory near the output is what
installs the intrusion groove; a mimicry-free top window removes it,
and multi-genre anchor-KL plus content dilution keep it down. Pure
distribution matching (`teacher_kl`) converges to the teacher's own
~97% token fidelity; verbatim recall lives in the last ~3% the teacher
definitionally lacks (the last-3% law), so the pre-law high-recall arms
are recorded as labeled hybrid baselines, not the method.
In that historical regime, distribution-shaped hidden losses (`lens_kl`,
`vocab_fisher`) amplified the groove; `vocab_mse`/`nmse` were safe. These
readout-bearing results are not current frontier evidence. Crown checkpoint
(slide8pure,
two seeds): 0.6B recites the whole 715-verse romance self-chained with
its first error at verse 708 — CER 0.007 / 99.3% line-exact / 2.5%
intrusion (n=200). Laws and the evidence chain: `EXPERIMENTS.md`;
machine-readable claims: `runs/conclusions.yaml`.

## Bootstrap

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e . && .venv/bin/pip install pytest
.venv/bin/python scripts/fetch_poem.py
.venv/bin/python scripts/build_dataset.py
.venv/bin/python scripts/build_teacher_cache.py
.venv/bin/python scripts/audit_configs.py
```

On the L40S cluster, use the interpreter and CUDA-wheel guidance in
`AGENTS.md`; do not rely on `/usr/bin/python3`.

## Container runtime

For Lustre-heavy GPU jobs, use the repository's Singularity runtime rather
than copying a virtual environment:

```bash
scripts/container_exec.sh python scripts/build_teacher_cache.py
```

The launcher binds this checkout as `/work`, uses the pinned PyTorch SIF and
Python-dependency overlay under `containers/`, preserves physical
`CUDA_VISIBLE_DEVICES`, and keeps Singularity and Torch caches under `/tmp`.
See `AGENTS.md` for the image contents, cache staging, development overlay,
and node validation commands; the concise runtime contract is in
[`docs/container_runtime.md`](docs/container_runtime.md).  For durable Hugging Face snapshots, see
[cache staging](docs/cache_staging.md).
# Local model-cache staging

For GPU campaigns, keep durable Hugging Face snapshots in
`$HOME/.cache/huggingface` and stage only the needed models to node-local
`/tmp` with `scripts/stage_hf_cache.sh`. The container launcher automatically
uses a completed stage. See [cache staging](docs/cache_staging.md).
