"""Experiment configuration: plain dataclasses loaded from YAML.

``load_config`` reads a base YAML plus an optional experiment YAML that
overrides it (shallow-merged per section).
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
    catechism: bool = False  # drill Q&A (follow/precede/cloze/section anchors)


@dataclass
class MaskConfig:
    mode: str = "rag"  # rag | thinking
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
    compaction: str = "remove"


@dataclass
class CacheConfig:
    root: str = "caches"
    shard_size: int = 128
    hidden_dtype: str = "float16"


@dataclass
class LoraConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0


@dataclass
class TrainConfig:
    method: str = "layerwise"
    schedule: str = "summed"  # layerwise only: summed | sequential
    lr: float = 1e-5
    epochs: int = 10
    micro_batch: int = 1
    grad_accum: int = 8
    seed: int = 17
    max_steps: int = 0  # 0 = no cap
    hidden_loss: str = "nmse"  # nmse | l2mse
    # auxiliary CE on gold answer tokens (0 = pure distillation). Pins the
    # student's argmax to the gold recitation and counters free-run drift
    # caused by teacher formatting quirks at the trained positions.
    answer_ce_weight: float = 0.0
    # layerwise hybrid: gold-CE on the LAST block only, computed through the
    # frozen final norm + lm_head. The graph is rooted at block n's detached
    # input, so the backward stays confined to block n — output supervision
    # without giving up block-locality.
    last_block_ce_weight: float = 0.0
    # tail-CE hybrid (summed schedule): the last `tail_ce_blocks` blocks train
    # JOINTLY — gradient flows within that window so the answer-CE at the top
    # can do multi-block credit assignment — while everything below stays
    # block-local. Motivated by the logit-lens finding (2026-07-03): strict
    # hidden matching stores recall below the top window, while the deficit is
    # confined to final-block readout. 0 = off (pure block-local, the default).
    tail_ce_blocks: int = 0
    tail_ce_weight: float = 0.0
    # per-block lens-CE (summed schedule): every block >= lens_ce_from gets a
    # behavioral auxiliary through the frozen logit lens — Belilovsky-style
    # local heads. Strictly block-local (unlike tail_ce): the personalization
    # / parallelism story is fully preserved. 0 = off.
    lens_ce_weight: float = 0.0
    lens_ce_from: int = 1
    grad_checkpointing: bool = True
    # sequential schedule
    plateau_patience: int = 3
    stage_max_steps: int = 500
    # LoRA-only: compute teacher targets per step by disabling the adapters
    # (student = base + adapters, so the frozen teacher is already resident).
    # Replaces the disk cache entirely — the choice at 120B scale.
    online_teacher: bool = False
    # warm-start: load student weights from runs/<init_from>/checkpoint
    # (teacher stays the base model — cache identity is untouched)
    init_from: str = ""
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class EvalConfig:
    recite_lines: int = 20
    every_epochs: int = 2


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


def load_config(base: str | Path, experiment: str | Path | None = None) -> ExperimentConfig:
    cfg = yaml.safe_load(Path(base).read_text()) or {}
    if experiment:
        over = yaml.safe_load(Path(experiment).read_text()) or {}
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return _from_dict(ExperimentConfig, cfg)
