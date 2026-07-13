"""Experiment configuration: plain dataclasses loaded from YAML.

``load_config`` reads a base YAML plus an optional experiment YAML that
overrides it (deep-merged: nested overrides keep their siblings).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    device: str = "cuda"
    # >0: pipeline parallel — blocks 1..split on cuda:0, the rest on cuda:1.
    # pipeline_splits generalizes this to N visible GPUs: e.g. [12, 24, 36]
    # maps a 48-layer model in four 12-layer chunks. Implemented via an HF
    # device_map, so accelerate's alignment hooks move activations even through
    # our direct block calls.
    pipeline_split: int = 0
    pipeline_splits: list = field(default_factory=list)
    # Optional HF placement for large scale probes. "auto" shards across all
    # visible devices; leave empty when using manual pipeline_split(s).
    device_map: str = ""


@dataclass
class DataConfig:
    poem_path: str = "data/poem/raw.txt"
    examples_path: str = "data/poem/examples.jsonl"
    window: int = 12
    stride: int = 4
    include_full: bool = True
    full_lines: int = 24
    context_pad: int = 4
    include_sections: bool = True
    section_max_lines: int = 24
    long_windows: list = field(default_factory=list)  # e.g. [24, 48]
    paraphrase: bool = False
    part_chunk_lines: int = 0
    catechism: bool = False
    maieutic: bool = False  # dialogue-framed elicitation specs (maieutic v4)
    corpus_style: str = "verse"  # verse | prose_quijote (question phrasing + system prompt)
    # -- v5 question-only datasets (owner, 2026-07-12) ----------------------
    # The jsonl carries questions + master-RAG passages, NO answers: the
    # teacher generates the answer at the teacher stage and the student
    # trains on its forward hidden states (src/selfupdate/data/questions.py).
    question_set: str = "legacy"  # legacy | v5
    # multi-corpus emission: [{poem_path, corpus_style, prefix}, ...];
    # empty = single corpus from poem_path/corpus_style above
    corpora: list = field(default_factory=list)
    rag_scope: str = "window"  # chapter | window (master-RAG granularity)
    rag_window_lines: int = 4  # window scope: target span ± this many lines
    v5_next_windows: list = field(default_factory=lambda: [1, 3, 6])
    v5_prev_stride: int = 3
    v5_cloze_block: int = 4
    v5_cloze_deletions: list = field(default_factory=lambda: [1, 2, 4, 8])
    v5_seed: int = 20260712


@dataclass
class MaskConfig:
    mode: str = "rag"  # rag | rag_tool | rag_system | thinking | thinking_selective
    max_think_tokens: int = 512
    # how the student's side compacts the privileged block:
    #   remove     — block deleted outright (zero size)
    #   stub       — replaced by a short uninformative placeholder token
    #   stub_gap   — stub + position-id gap so RoPE geometry matches the teacher
    #   remove_gap — no stub, but the aligned span's position ids are rebased
    #                by the gap. Added 2026-07-03 to complete the 2x2
    #                (stub tokens x RoPE geometry): wave F concluded
    #                "teacher-geometry imitation is harmful" from the
    #                stub_gap-vs-stub delta, where the stub tokens are a
    #                confound; remove-vs-remove_gap isolates pure geometry.
    #                Constant-offset invariance (test_position_invariance)
    #                says the rebase is output-neutral for the base model, so
    #                any difference is purely about which geometry the
    #                distilled memory is written at.
    #   pad_random — length-matched random fill of the privileged block
    #                (every token distinct, ordinary vocabulary only, seeded
    #                per example): position gap is zero by construction.
    #                Owner 2026-07-12: fixed pad tokens and repeated fillers
    #                are attendable attractors — random non-repeating fill is
    #                the sanctioned length-preserving censor.
    compaction: str = "remove"


@dataclass
class CacheConfig:
    root: str = "caches"
    shard_size: int = 128
    hidden_dtype: str = "float16"
    # Extra generation allowance for question-only RAG teacher targets.  The
    # 96-token margin is certified separately by the RAG gate; it prevents
    # conversational framing from truncating the answer span.
    generation_extra_tokens: int = 96
    # Open-answer teacher generation.  B=1 preserves the historical cache
    # builder; larger values use left-padded greedy batches with OOM backoff.
    generation_batch: int = 1
    # Exact allowance groups at 1. Larger values round allowances up to this
    # width; zero places each outer batch in one group.
    generation_budget_bucket: int = 1
    # Optional Transformers decode compilation.  reduce-overhead uses PyTorch
    # CUDA graphs where the model/cache permit it; off is the eager baseline.
    generation_compile: bool = False
    generation_cache_implementation: str = ""
    # Graph-shape controls. Dynamic is the safe default; dense-model probes
    # can pin cache and physical batch shapes to amortize one capture.
    generation_compile_dynamic: bool = True
    generation_cache_max_tokens: int = 0
    generation_fixed_batch: bool = False
    # Zero preserves dataset order.  A positive seed deterministically shuffles
    # within allowance buckets and shuffles aligned batches; answers are
    # restored to their original example ids.
    generation_shuffle_seed: int = 0
    # Optional hard ceiling for prompt + generated answer.  Zero delegates to
    # the model configuration.  Campaigns benchmarking an 8k deployment pin
    # 8192 explicitly so an overlong record fails before model.generate.
    max_sequence_tokens: int = 0
    # Deterministic evenly-spaced subset for performance probes.  Zero means
    # the complete dataset and is the only setting used for campaign caches.
    limit: int = 0


@dataclass
class LoraConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0


@dataclass
class TrainConfig:
    method: str = "layerwise"
    # method | teacher_reference | ablation | control | legacy_archive | confounded | open
    run_class: str = "method"
    # summed | sequential | teacher_censored | mixed
    schedule: str = "summed"
    # mixed schedule: probability an item routes through the teacher-stream
    # (censored) branch, linear from start (epoch 0) to end (last epoch)
    mix_teacher_start: float = 1.0
    mix_teacher_end: float = 0.0
    lr: float = 1e-5
    epochs: int = 10
    micro_batch: int = 1
    grad_accum: int = 8
    # item: historical path, loops over examples one by one even when
    # micro_batch > 1. padded: one block forward/backward over a right-padded
    # batch. bucketed: same padded path, but randomized length buckets reduce
    # pad waste without globally sorting the corpus.
    batching: str = "item"  # item | padded | bucketed
    length_bucket_width: int = 128
    # MoE handling:
    # dense_or_black_box = ordinary block-output layerwise distillation. For
    # MoE, the router/expert mechanism is inside the block and remains valid
    # method evidence; expert agreement remains an extra measured claim.
    # teacher_forced = replay teacher-selected experts during training.
    # router_aligned = train/regularize the student router toward teacher
    # routing and report top-k overlap. The latter two are MoE-specific method
    # innovations for sparse-expert families.
    moe_mode: str = "dense_or_black_box"  # dense_or_black_box | teacher_forced | router_aligned
    # router_aligned only: UNIFORM per-MoE-layer weight of the
    # KL(teacher routing || student routing) regularizer (depth-uniform by
    # construction — the naming contract applies to routers too). No silent
    # default: router_aligned arms must pin this > 0 explicitly.
    moe_router_weight: float = 0.0
    seed: int = 17
    max_steps: int = 0  # 0 = no cap
    # nmse | l2mse | cosine | huber (absolute-state geometric) | vocab_mse
    # | lens_kl | lens_js | tuned_lens_kl | vocab_fisher
    #   (absolute-state frozen-vocabulary; lens_js is the bounded symmetric
    #   Jensen-Shannon control)
    # | jacobian_nmse (pure frozen JᵀJ transport metric)
    # | jacobian_vocab_mse | jacobian_lens_kl (frozen downstream transport,
    # then the corresponding vocabulary metric; all need jacobian_lens_path)
    # | delta_nmse | delta_cosine | delta_vocab_cos (successive raw block
    # increments; L=1/h_n use the paired state fallback because the cache has
    # no h0 and h_n is post-final-norm).  See losses.py / docs/hidden_loss.md.
    hidden_loss: str = "nmse"
    tuned_lens_path: str = ""
    jacobian_lens_path: str = ""
    # Frozen offline per-layer precision matrices for mahalanobis hidden loss.
    mahalanobis_path: str = ""
    # Raw multi-layer displacement offsets; legal only inside a faithful
    # connected window and averaged uniformly across eligible offsets.
    multi_delta_scales: list[int] = field(default_factory=lambda: [1, 2, 4])
    # Top readout term attached ONLY to sanctioned sliding windows:
    # conn_window > 0, conn_stride == 1, and readout_window_blocks == conn_window.
    # The connected graph is still a gradient-isolation unit rooted at a
    # detached window input; nothing below the window receives gradient.
    readout_window_blocks: int = 0
    readout_weight: float = 0.0
    # Hidden-loss weight inside a connected window. Method arms keep this 1.0;
    # zero or reduced values are ablations only.
    window_hidden_weight: float = 1.0
    # readout-term source (owner correction 2026-07-05): 'teacher_kl' =
    # KL(teacher || student) on the TEACHER'S context-conditioned logits
    # (derived from targets[n] through the frozen head — zero extra
    # compute, 100% teacher-sourced. No reference-text source is allowed.
    # No base config default is allowed; readout runs must pin this explicitly.
    readout_source: str = 'UNSET'
    # sliding k-connected windows over the BODY (owner proposal 2026-07-04):
    # every layer gets k-deep credit assignment, peak activation graph
    # stays k blocks. 0/1 = classic block-local. The top readout is just
    # the last sliding window position — the only one where logits exist.
    conn_window: int = 0
    # 0 = DISJOINT windows (detach every k blocks; walk compute unchanged;
    # credit depth depends on position inside the window). 1 = FAITHFUL
    # sliding windows: every body layer's target is matched as the ENDPOINT
    # of a k-deep window that updates ALL covered blocks — uniform k-deep
    # credit everywhere, at ~k x body compute.
    conn_stride: int = 0
    # forward-deduplicated faithful sliding windows: identical window/credit
    # semantics (every block still receives W backward passes), but each block
    # is grad-forwarded ONCE from its detached trajectory root and windows
    # chain backward through the stored per-block graphs, instead of
    # re-forwarding the whole window per endpoint (~1.3-1.5x on window arms,
    # same peak graph memory). Gradients agree up to autocast replay rounding
    # (exact in fp32; see tests/test_window_dedup.py), so this is a PINNED
    # knob, default off: flipping it mid-campaign would fork queued arms.
    # Memory price of this and every 2026-07-06 speed fix: docs/memory.md
    # "Speed/Memory Ledger" (this one is zero; batching is the VRAM dial).
    window_dedup: bool = False
    # anchor-KL: KL(teacher/base || student) on anchor fragments through the
    # top readout window. Needs an online teacher for base logits.
    anchor_kl_weight: float = 0.0
    # Depth-uniform, block-local preservation of frozen-base hidden states on
    # generic anchor fragments (separate from output anchor KL).
    anchor_hidden_weight: float = 0.0
    anchor_path: str = "data/anchors_es.txt"
    # sequential schedule
    plateau_patience: int = 3
    stage_max_steps: int = 500
    # LoRA-only: compute teacher targets per step by disabling the adapters
    # (student = base + adapters, so the frozen teacher is already resident).
    # Replaces the disk cache entirely — the choice at 120B scale.
    online_teacher: bool = False
    # full-FT counterpart: keep a resident frozen bf16 copy of the base model
    # as online teacher (~1.2 GB at 0.6B). Needed by schedules that consume
    # full-sequence teacher states (teacher_censored, mixed) without LoRA.
    # Explicit rather than automatic so a run's VRAM footprint is never a
    # surprise (see AGENTS.md VRAM lessons).
    frozen_teacher_copy: bool = False
    # summed full-FT: page each block's Adam moments to CPU between its
    # steps. Blocks step one at a time, so resident optimizer state drops
    # from all-blocks (8 B/param, the largest full-FT term) to one block's.
    # Costs PCIe traffic per optimizer step — pair with grad_accum. This is
    # what lets true full-FT summed fit at 4B on one 46 GB card.
    offload_adam: bool = False
    # AUDIT knob: permute which layer's teacher state each layer is trained
    # toward (fixed seeded permutation). Destroys trajectory structure while
    # preserving marginal statistics, data, CE and budget. If recall
    # survives, states were not carrying layer-structured signal; expected:
    # collapse toward the label-only (kd-SFT) level. Ablation-only.
    scramble_targets: bool = False
    # warm-start: load student weights from runs/<init_from>/checkpoint
    # (teacher stays the base model — cache identity is untouched)
    init_from: str = ""
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class EvalConfig:
    recite_lines: int = 20
    every_epochs: int = 2
    # Optional explicit corpus set for in-training recall telemetry.  Empty
    # retains the historical one-corpus cfg.data.poem_path behavior; combined
    # campaigns must pin both corpus names from eval.tasks.RECALL_CORPUS_PATHS.
    recall_corpora: list = field(default_factory=list)
    # Fixed fast subset of ARC-Easy / ARC-Challenge / HellaSwag, evaluated in
    # process.  0 disables it for legacy/certification configs; campaign arms
    # pin 1 to record epoch 0 and every completed epoch.
    standard_damage_every_epochs: int = 0
    standard_damage_limit: int = 16
    standard_damage_batch_size: int = 8
    # Batched greedy decode for the in-training three-task recall telemetry
    # (tasks_eval generation_batch).  1 = historical per-item loop; measured
    # 2026-07-11: B=1 per-epoch eval was 42-56% of loss-grid arm wall time.
    generation_batch: int = 1


@dataclass
class ExperimentConfig:
    run_name: str = "dev"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _from_dict(cls, d: dict):
    known = {f.name for f in dataclasses.fields(cls)}
    extra = sorted(set(d) - known)
    if extra:
        raise ValueError(f"unknown {cls.__name__} key(s): {', '.join(extra)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if isinstance(v, dict) and f.default_factory is not dataclasses.MISSING:
            sub_cls = type(f.default_factory())
            if dataclasses.is_dataclass(sub_cls):
                v = _from_dict(sub_cls, v)
        kwargs[f.name] = v
    return cls(**kwargs)


def _load_yaml_mapping(path: str | Path) -> dict:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"failed to parse YAML config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"YAML config {path} must be a mapping at top level")
    return data


def _merge_deep(base: dict, over: dict) -> dict:
    """Recursive dict merge: an override that sets one key of a nested dict
    keeps its siblings. The old one-level `cfg[k].update(v)` silently RESET
    siblings of any dict nested two levels down (e.g. an experiment pinning
    train.lora.enabled dropped the base's lora.r/alpha back to dataclass
    defaults) — the silent-config-fork bug class. Landed 2026-07-11 after an
    A/B audit of all 179 real (base, experiment) pairs showed zero semantic
    diffs, so no existing config relied on the reset behavior."""
    merged = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _merge_deep(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(base: str | Path, experiment: str | Path | None = None) -> ExperimentConfig:
    cfg = _load_yaml_mapping(base)
    if experiment:
        cfg = _merge_deep(cfg, _load_yaml_mapping(experiment))
    return _from_dict(ExperimentConfig, cfg)
