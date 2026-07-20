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
    # Optional physical ids used by a PPn launcher.  An empty list preserves
    # the historical visible-device order (0 .. P-1); ids are never
    # renumbered by the runtime.
    pipeline_devices: list = field(default_factory=list)
    # A launcher may pin the expected number of stages when it is not
    # discoverable from the current process (for example before torch.distributed
    # is initialized).  Zero means infer it from pipeline_splits/devices.
    pipeline_world_size: int = 0
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
    # teacher generates the answer at the teacher stage; v4 trains each block
    # on teacher h[L-1] -> teacher h[L].
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
    # how validation censors the privileged block (training uses teacher
    # coordinates with the corresponding v4 attention mask):
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
    #   flow_mask  — length- and token-preserving information-flow censor:
    #                privileged rows stay in the sequence but are zeroed at
    #                every block boundary and excluded from attention/state
    #                writes. This is the architecture-generic censorship
    #                control used by v4 validation.
    #   intact     — diagnostic control: student sees the original privileged
    #                block, so student_ids == teacher_ids exactly.
    compaction: str = "flow_mask"


@dataclass
class CacheConfig:
    root: str = "caches"
    # Runtime target placement. ``durable`` uses root (or the historical
    # SELFUPDATE_TEACHER_CACHE_ROOT staging override). ``node_epoch0`` uses a
    # numerically local cache generated once per host and atomically published
    # in node-local shared memory; all later arms on that host memory-map it.
    runtime_policy: str = "durable"  # durable | node_epoch0
    node_root: str = "/dev/shm/$USER/selfupdate-teacher-cache-v4"
    # Optional student-view-independent cache selector.  Teacher hidden states
    # and generated answer ids do not depend on how the privileged RAG block
    # is censored for the student.  Setting this to (for example) ``remove``
    # lets a pad_random arm consume the already-certified remove cache while
    # recomputing all student offsets from its active masking view.
    source_compaction: str = ""
    shard_size: int = 128
    hidden_dtype: str = "float16"
    # Pipeline-v3 teacher-seeded execution can consume each block's complete
    # frozen-teacher input directly.  This is a distinct, larger cache
    # identity: i{L}=h[L-1] over the full teacher sequence.  Ordinary caches
    # remain aligned-target-only.
    store_full_teacher_inputs: bool = True
    full_input_shard_size: int = 1
    # Bound Python-owned teacher tensors. Safetensors already mmap the durable
    # cache, so evicted rows remain available through the kernel page cache
    # without pinning the complete multi-GB corpus in every trainer process.
    item_cache_items: int = 64
    # Extra generation allowance for question-only RAG teacher targets.  The
    # 96-token margin is certified separately by the RAG gate; it prevents
    # conversational framing from truncating the answer span.
    generation_extra_tokens: int = 96
    # Optional fixed answer ceiling. Zero keeps the proportional per-record
    # budget above; a positive value replaces it for every record. This is a
    # scientific protocol knob, not merely a throughput setting: Qwen3.5-0.8B
    # needs the historical 4096-token ceiling to terminate naturally.
    generation_max_tokens: int = 0
    # Open-answer teacher generation.  B=1 preserves the historical cache
    # builder; larger values use left-padded greedy batches with OOM backoff.
    generation_batch: int = 1
    # Teacher-forced hidden-state forwards after answer generation/import.
    # Kept independent because large teachers may decode at B=64 but only fit
    # a much smaller all-hidden-states batch. OOM backoff persists the safe B.
    teacher_batch: int = 1
    # Exact allowance groups at 1. Larger values round allowances up to this
    # width; zero places each outer batch in one group.
    generation_budget_bucket: int = 1
    # Optional Transformers decode compilation.  reduce-overhead uses PyTorch
    # CUDA graphs where the model/cache permit it; off is the eager baseline.
    generation_compile: bool = False
    generation_cache_implementation: str = ""
    # Graph-shape controls. Dynamic is the safe default; dense-model probes
    # can pin cache and physical batch shapes to amortize one CUDA-graph
    # capture (the CUDA term, unrelated to v4 teacher forwards).
    generation_compile_dynamic: bool = True
    generation_cache_max_tokens: int = 0
    generation_fixed_batch: bool = False
    # Optional exact-token response JSONL produced by the graph/continuous
    # batching benchmark. Empty means generate inside this process.
    generation_responses_path: str = ""
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
    # This checkout exposes only pipeline-v4 teacher-hidden training.
    pipeline_version: int = 4
    pipeline_revision: str = "4.0"
    expert_routing_source: str = "black_box"       # future: teacher_routing_cache
    method: str = "layerwise"
    # method | teacher_reference | ablation | control | legacy_archive | confounded | open
    run_class: str = "method"
    lr: float = 1e-5
    epochs: int = 10
    micro_batch: int = 1
    # item: historical path, loops over examples one by one even when
    # micro_batch > 1. padded: one block forward/backward over a right-padded
    # batch. bucketed: same padded path, but randomized length buckets reduce
    # pad waste without globally sorting the corpus.
    batching: str = "bucketed"  # cohort construction, not a training trajectory
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
    # nmse | l2mse | cosine | huber (absolute-state geometric)
    # | delta_cosine (block-local update direction around detached teacher
    #   input; explicit absolute-cosine fallback at the post-norm final block)
    # | vocab_mse | vocab_cycle_mse (frozen output-logit/input-embedding
    #   round trip; distinct from vocab_mse even when weights are tied)
    # | lens_kl | lens_js | tuned_lens_kl | vocab_fisher
    #   (absolute-state frozen-vocabulary; lens_js is the bounded symmetric
    #   Jensen-Shannon control)
    # | jacobian_nmse (pure frozen JᵀJ transport metric)
    # | jacobian_vocab_mse | jacobian_lens_kl (frozen downstream transport,
    # then the corresponding vocabulary metric; all need jacobian_lens_path)
    # See losses.py / docs/hidden_loss.md.
    hidden_loss: str = "nmse"
    # Deterministic frozen-vocabulary score sketch used only by
    # vocab_cosine_sampled. Rows are sampled from the frozen unembedding and
    # centred by its vocabulary-wide mean; no vocabulary parameter is trained.
    vocab_cosine_samples: int = 0
    vocab_cosine_seed: int = 17
    tuned_lens_path: str = ""
    jacobian_lens_path: str = ""
    # Frozen offline per-layer precision matrices for mahalanobis hidden loss.
    mahalanobis_path: str = ""
    # warm-start: load student weights from runs/<init_from>/checkpoint
    # (teacher stays the base model — cache identity is untouched)
    init_from: str = ""
    # ------------------------------------------------------------------
    # Pipeline-v4: blockwise teacher-forced training with frozen teacher KV.
    # Every training loss is block-local against cached teacher states; the
    # attention context is the teacher's OWN frozen K/V (adapters-off
    # projections of cached i{L}=h[L-1]); the student's trajectory is never a
    # loss input.  Because both the block input and the attention context are
    # teacher-fixed, layers are fully independent — the multi-GPU strategy is
    # layer-sharding across processes, not pipelining.
    # ------------------------------------------------------------------
    # teacher_frozen: K/V from adapters-off projections, computed once (they
    # never change). student_refresh: recompute the K/V through the current
    # adapters every v4_kv_refresh_epochs epochs (still no gradient through
    # K/V; inputs stay teacher states).
    v4_kv_source: str = "teacher_frozen"  # teacher_frozen | student_refresh
    # Where teacher hidden states come from (owner contract: "where IF ANY
    # the teacher hidden are cached ... or just keep calculating it").
    #   cache  — read i{L}/h{L} from the store_full_teacher_inputs cache.
    #   online — ONE adapters-off forward per cohort per epoch records the
    #            owned layers' inputs and targets on the GPU; the cache
    #            shrinks to an index of spans + generated answer ids
    #            (build_teacher_cache.py --index-only). Requires item_major
    #            (layer_major would redo the teacher forward once per layer).
    #   store  — fill ONCE before the epoch loop via the stage relay (store-fill)
    #            (v4_store.py): stage 0 embeds and walks its owned layers
    #            adapters-off, ships the boundary hidden downstream; every
    #            stage fills its per-(layer,cohort) store and later epochs
    #            run ZERO teacher forwards (the measured 3.2x lever at 27B;
    #            the 122B/397B scaling lane pairs it with v4_stage_scoped).
    v4_teacher_source: str = "cache"  # cache | online | store
    # Which detached adapters-off state supplies block L's query input and
    # frozen attention context. teacher_uncensored is the historical v4 law.
    # flow_censored_teacher is the legal-A diagnostic repair: an independent
    # fully flow-censored adapters-off walk supplies h_c[L-1] for BOTH query
    # and K/V, while the ordinary uncensored teacher walk still supplies the
    # target h_u[L].  Initially supported only by the online, fully resident,
    # single-process route; validate/runtime fail loudly everywhere else.
    v4_context_source: str = "teacher_uncensored"  # teacher_uncensored | flow_censored_teacher
    # Backpressure for the store-fill relay: at most this many boundary
    # files of one producer stage alive in the exchange (consumer deletion
    # is the ack).
    v4_capture_inflight: int = 2
    # Staged store-fill under rotation walks chunk-wise layer-outer: this
    # many cohorts are ingested, walked through every owned block (one
    # page-in per block per chunk), and shipped downstream together.
    # Larger = fewer page-ins but a coarser relay pipeline and more resident
    # chunk hiddens (~0.5 GB/cohort at 397B). Ignored without rotation.
    v4_fill_chunk_cohorts: int = 8
    # Boundary-tensor carrier between stage processes — the "mail".
    # Owner decision 2026-07-18: cross-node mail is NATIVE InfiniBand
    # (NCCL over IB verbs; measured dual HDR-200 at line rate). Lustre is
    # NOT a carrier for boundary mail — it serves only checkpoints, logs
    # and the small control-plane envelopes. auto (default): stages on
    # ONE node keep the /dev/shm file exchange (RAM-speed, zero deps);
    # the moment the stage set spans hosts (launcher exports
    # SELFUPDATE_V4_CROSS_NODE=1) the mail goes NCCL. "files"/"nccl"
    # force a carrier for a paired comparison or debug only.
    v4_relay_transport: str = "auto"  # auto | files | nccl
    # 1800 not 600: the epoch relay is async and stages finish at very
    # different times (a fast stage with few owned layers + no eval tail can be
    # 3 epochs ahead of the last stage's eval tail). The finalize barrier (#24)
    # must tolerate that lag; DeepSeek PPP8 hit the old 600 s exactly.
    v4_nccl_timeout_s: int = 1800
    v4_kv_refresh_epochs: int = 0
    # immediate_sgd performs one state-free write per block and cohort.
    # adam gives each owned block its own AdamW (more memory; pairs
    # naturally with layer_major, which keeps one block's moments hot).
    v4_optimizer: str = "immediate_sgd"  # immediate_sgd | adam
    # layer_major: for each owned layer, traverse every cohort (teacher
    # tensors and optimizer state stay hot per layer). item_major: for each
    # cohort, walk the owned layers (streaming-friendly).
    v4_loop_order: str = "layer_major"  # layer_major | item_major
    # Which teacher-coordinate positions carry the training loss. answer =
    # teacher-realized answer tokens. aligned = the whole cached
    # aligned span (shared_mid + answer; everything non-censored the cache
    # carries targets for). thinking_answer is reserved: it needs per-record
    # thinking-span metadata the dataset does not expose yet, so dispatch
    # raises rather than silently training on a guess.
    v4_loss_positions: str = "answer"  # answer | aligned | thinking_answer
    # Student-trajectory validation relay: 0 disables; any positive value
    # enables it AT EPOCH BOUNDARIES in the current implementation. The
    # sub-epoch cohort cadence the name promises is deliberately deferred:
    # under the default layer_major order, layers finish their cohorts at
    # different times, so "all owned layers current to cohort c" only exists
    # at epoch boundaries anyway. An item_major sub-epoch relay would need a
    # sequence protocol on top of _RelayFiles and is future work.
    # In v4_battery_mode=distributed, positive means run the exact
    # synchronized b trajectory at native battery epoch boundaries; the
    # approximate asynchronous relay is not started in that mode.
    v4_relay_every_cohorts: int = 4
    # Where per-layer teacher tensors live during training. gpu_corpus keeps
    # the active layer's whole-corpus inputs/targets/KV resident (layer_major
    # on small models: ~4 GB/layer at 0.6B). cpu_stream stages per cohort
    # from pinned host memory. auto sizes the corpus and picks.
    v4_teacher_residency: str = "auto"  # auto | gpu_corpus | cpu_stream
    # Batch-chunking for the teacher forward (v4_teacher_source
    # =online). It runs the FULL teacher-forced sequence (length T)
    # through every block to reach the owned range; a full-attention gemma4
    # block over a long cohort materializes O(B*heads*T*T) SDPA scores, which
    # OOMs at micro_batch=32 / T~4096 on 80 GB. Each item's forward is
    # independent (attention is within-sequence, causal), so splitting the
    # cohort's B items into chunks of this size and concatenating is
    # numerically exact. 0 = no chunking (whole cohort at once, historical
    # behaviour). Only affects the no-grad teacher forward, never the training step.
    v4_capture_micro_batch: int = 0
    # Layer-ownership partition for multi-process v4. Deliberately SEPARATE
    # from model.pipeline_split(s): those are teacher-cache loading controls,
    # while every v4 process loads the full model on ONE card and trains only
    # its owned contiguous block range. Cuts use pipeline_splits semantics
    # (blocks 1..n; cut c means the next stage starts at block c+1). Empty =
    # one stage owning every block.
    v4_stage_splits: list = field(default_factory=list)
    # One physical CUDA device id per stage (never renumbered).
    v4_stage_devices: list = field(default_factory=list)
    # Which stage THIS process is (set via scripts/train.py --v4-stage).
    # -1 = single-process mode: one process owns every block.
    v4_stage: int = -1
    # Utilization gate (owner, 2026-07-17): any run whose TRAINING-phase GPU
    # utilization stays below this percentage is a FAIL and must abort; the
    # goal is 90. Measured by NVML sampling at cohort boundaries during the
    # v4 walk only — the mandated per-epoch generation evals are inherently
    # low-util and are excluded from the gate on purpose. 0 disables (smoke
    # mechanics runs). The measured mean is logged in every v4_epoch row as
    # train_phase_gpu_util.
    v4_min_train_gpu_util: float = 0.0
    # Scaling lane (models over one card per stage): materialize only the
    # owned blocks + vocab stack from the safetensors index; foreign blocks
    # stay on the meta device (shard_load.py). Requires a staged launch and
    # LoRA; the per-epoch battery degrades to a loudly-marked skip row until
    # v4_battery_mode=subprocess lands (plan B6).
    v4_stage_scoped: bool = False
    # Where owned FROZEN block weights live between visits (plan B4).
    # resident: on the card (fits through ~122B at 4 stages). rotate: CPU
    # mmap masters, paged per layer_major visit with the block's Adam
    # moments (the 397B lane). auto: measure owned bytes vs free VRAM at
    # load. Non-resident requires v4_stage_scoped + layer_major.
    v4_weight_residency: str = "resident"  # resident | rotate | auto
    # How the owner-mandated per-epoch battery runs in staged mode. distributed:
    # all ranks synchronously evaluate with their live owned blocks over a
    # dedicated NCCL communicator (supported resident Qwen/Gemma families).
    # graft:
    # stage 0 grafts every stage's adapters onto its full resident model
    # (impossible under v4_stage_scoped). subprocess: all stages release
    # VRAM at the boundary; stage 0 spawns scripts/v4_battery.py which
    # loads the model device_map=auto over every card, grafts, probes,
    # exits (plan B6). Requires scripts/train.py's SELFUPDATE_V4_CONFIG.
    v4_battery_mode: str = "graft"  # distributed | graft | subprocess
    # Adam hyperparameters for v4_optimizer=adam (one AdamW per owned block).
    # Defaults reproduce torch AdamW. v4_grad_clip=0 disables clipping; >0
    # clips each block's gradient to that max L2 norm before the step, and the
    # LOGGED grad norm is then the honest PRE-clip value.
    v4_adam_betas: tuple = (0.9, 0.999)
    v4_adam_eps: float = 1e-8
    v4_adam_weight_decay: float = 0.0
    v4_grad_clip: float = 0.0
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
    # Optional live-PP a' control: autoregress from each training record's
    # complete uncensored/RAG prompt and compare with its vLLM answer.  Zero
    # preserves historical battery cost/rows; positive values select the
    # deterministic first N records. This control is distinct from epoch-zero
    # censored recall and is currently native-distributed-only.
    vllm_uncensored_generation_limit: int = 0
    vllm_uncensored_max_extra_tokens: int = 48


@dataclass
class ExperimentConfig:
    run_name: str = "dev"
    # Historical project label retained in run identity; the only executable
    # training protocol in this checkout is pipeline v4.
    layerwise_project_version: str = "3.4"
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
    paired-diff audit of all 179 real (base, experiment) pairs showed zero semantic
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
    removed = sorted(
        set((cfg.get("train") or {}))
        & {"readout_window_blocks", "readout_weight", "readout_source",
           "anchor_kl_weight"}
    )
    if removed:
        raise ValueError(
            "removed output-readout training key(s): " + ", ".join(removed)
            + "; this branch is strictly hidden-state layerwise. The old "
              "runtime and configs are recoverable from git history."
        )
    return _from_dict(ExperimentConfig, cfg)
