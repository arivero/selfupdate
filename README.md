# selfupdate - layerwise forward distillation of context

The same model plays teacher and student. The teacher prompt contains
privileged context - a RAG passage or a visible `<think>` trace - while the
student receives the same prompt with that context hidden. Training asks each
student block to reproduce the teacher's hidden state at aligned token
positions.

The research target in this branch is layerwise forward distillation:
block-local hidden-state learning, plus a bounded sliding readout window
with depth-uniform credit for free-run behavior. Whole-network logit
distillation is not an active method in this tree.

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
tests/              alignment / cache / locality / layerwise hybrid tests
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
  The behavioral term is a bounded sliding connected window (`conn_window` +
  `conn_stride: 1`) — uniform k-deep credit for every block — whose top
  window may carry a teacher-sourced readout (`readout_source: teacher_kl`).
  Reference-text cross-entropy is never a training target on this branch
  (eval against the reference text is correct and required); the embedding
  and logits matrix are never trained (Frozen-Vocabulary Principle). Window
  semantics: [docs/windows.md](docs/windows.md).

See [docs/hidden_loss.md](docs/hidden_loss.md) for locality proofs and
[docs/scaling.md](docs/scaling.md) for the large-model plan.

## Evaluation generation pipeline v2

Epoch-zero censorship controls and post-training checkpoint recall use
`scripts/teacher_ceiling.py`, which submits the deterministic evaluation
prompts and their individual token budgets directly to vLLM. Pass
`--checkpoint runs/<run>/checkpoint` to evaluate a trained model through the
same prompt construction and scoring path used for its base reference. The
retired Transformers `model.generate` implementation is intentionally not
kept in-tree; it remains recoverable from Git history.

For repeated base-model or checkpoint loads, stage snapshots into Unix tmpfs
with `scripts/stage_hf_cache.sh --shm <org/model>`. Container launches prefer
the completed RAM stage automatically. Direct vLLM launches set `HF_HOME` to
`/dev/shm/$USER/selfupdate-hf-cache`; ordinary safetensors file access then
benefits from kernel-shared resident pages without a custom model object.

## Current Finding

Storage and readout dissociate. Hidden matching writes distributed,
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
Distribution-shaped hidden losses (`lens_kl`, `vocab_fisher`) amplify
the groove; `vocab_mse`/`nmse` are safe. Crown checkpoint (slide8pure,
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
.venv/bin/python -m pytest tests/ -q
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
