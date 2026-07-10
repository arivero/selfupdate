"""Training runtime: device placement, optimizer policy, memory accounting.

Separates HOW training executes (model/teacher placement, optimizer state
location, paging, synchronization, VRAM accounting) from WHAT is trained
(the schedule loops in ``layerwise.py``). Schedule code receives a built
``TrainingRuntime`` and never touches ``from_pretrained``, device maps, or
optimizer construction — the "PP2 failure" class of bug (an execution knob
silently forking an experiment) stays confined to this module.

Optimizer policy is explicit rather than implied by booleans:

- ``lora_fused``     adapters only; one AdamW, foreach stepping (the extra
                     tensor-list intermediates are negligible at LoRA size).
- ``full_resident``  full-FT, moments on GPU; one AdamW, non-foreach (peak
                     memory wins over step latency at model scale).
- ``full_offload``   full-FT, moments on CPU; per-block AdamW so paging
                     stays block-granular (``train.offload_adam``).

All three preserve the historical PER-BLOCK gradient clipping norm: clipping
is part of the experiment, not of the execution policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..teacher.cache import TeacherCache, resolve_cache_dir
from .blocks import BlockStack


def load_causal_lm(src, **kw):
    """Load a decoder LM regardless of head registration. Multimodal
    releases like Mistral-Medium-3.5 (mistral3) register ONLY as
    image-text-to-text and are absent from the causal-LM auto-map, so
    ``AutoModelForCausalLM`` raises; the ITT wrapper exposes the same
    ``.model.language_model`` decoder stack BlockStack navigates. gemma4 and
    qwen3_5_moe ARE in the causal map and take the first path unchanged."""
    try:
        return AutoModelForCausalLM.from_pretrained(src, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(src, **kw)


def pp_device_map(cfg) -> dict:
    """Pipeline map: embedding on cuda:0; decoder blocks partitioned by
    ``pipeline_split`` (2 GPUs) or ``pipeline_splits`` (N GPUs). The final
    norm/head live on the last card for untied models, so the top readout
    window stays colocated with logits."""
    from transformers import AutoConfig

    mc = AutoConfig.from_pretrained(cfg.model.name)
    text_cfg = getattr(mc, "text_config", mc)
    n = text_cfg.num_hidden_layers
    splits = list(cfg.model.pipeline_splits or [])
    if splits:
        if torch.cuda.device_count() < len(splits) + 1:
            raise ValueError(
                f"pipeline_splits {splits} needs {len(splits) + 1} visible GPUs"
            )
        if splits != sorted(splits) or splits[0] <= 0 or splits[-1] >= n:
            raise ValueError(f"pipeline_splits {splits} outside 1..{n - 1}")
    else:
        if torch.cuda.device_count() < 2:
            raise ValueError("pipeline_split needs 2 visible GPUs (queue n_gpus=2)")
        split = cfg.model.pipeline_split
        if not 0 < split < n:
            raise ValueError(f"pipeline_split {split} outside 1..{n - 1}")
        splits = [split]
    # tied embeddings (Qwen3 <=1.7B): embed IS lm_head — one tensor cannot
    # live on two cards, so the whole vocabulary stack stays on cuda:0 and
    # readout-window loss calls hop back (an [A,H] transfer per call). Untied
    # models put norm+head on cuda:1 with the readout window.
    tied = getattr(mc, "tie_word_embeddings",
                   getattr(text_cfg, "tie_word_embeddings", False))
    last_dev = len(splits)
    vocab_dev = 0 if tied else last_dev
    prefix = "model.language_model" if getattr(mc, "model_type", "") == "gemma4" else "model"
    dm = {f"{prefix}.embed_tokens": 0, f"{prefix}.rotary_emb": 0,
          f"{prefix}.norm": vocab_dev, "lm_head": vocab_dev}
    if prefix != "model":
        dm["model.vision_tower"] = 0
        dm["model.embed_vision"] = 0
    for i in range(n):
        dev = 0
        while dev < len(splits) and i >= splits[dev]:
            dev += 1
        dm[f"{prefix}.layers.{i}"] = dev
    return dm


def uses_pipeline_map(cfg) -> bool:
    return cfg.model.pipeline_split > 0 or bool(cfg.model.pipeline_splits)


def vocab_signature(stack) -> tuple:
    """Cheap exact fingerprint of the frozen vocabulary tensors (embedding,
    final norm, head). Computed at trainer start and re-checked before
    save: NO learning of any kind may modify these — they are the fixed
    basis of every lens and every cached teacher target."""
    sig = []
    seen: set[int] = set()
    for m in (stack.embed_tokens, stack.final_norm, stack.lm_head):
        for p in m.parameters():
            # tied-embedding models (Qwen3 <=1.7B): embed IS lm_head — one
            # pass over the shared tensor, not two (only compared within-run)
            if id(p) in seen:
                continue
            seen.add(id(p))
            # chunked fp64 sums: a full p.double() copy of a 200k-vocab
            # embedding is ~4 GB — enough to OOM a 20B-resident card
            s = a = 0.0
            for chunk in p.detach().reshape(-1).split(1 << 22):
                c = chunk.double()
                s += c.sum().item()
                a += c.abs().sum().item()
            sig.append((s, a))
    return tuple(sig)


def _move_opt_state(opt, device) -> None:
    """Page an optimizer's per-param state tensors between devices (Adam
    moments dominate full-FT memory at 8 B/param). Moving "back" targets
    each PARAM's own device — under pipeline parallel the blocks live on
    different cards and a global device string would silently migrate
    moments to the wrong one."""
    to_cpu = torch.device(device).type == "cpu"
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p)
            if not st:
                continue
            tgt = torch.device("cpu") if to_cpu else p.device
            for k, v in st.items():
                if torch.is_tensor(v) and v.device != tgt:
                    st[k] = v.to(tgt)


@dataclass
class OptimizerPlan:
    """Explicit optimizer policy: state placement + stepping strategy.

    ``step()`` preserves the historical per-block clip norm in every policy —
    combining AdamW instances changes speed, global clipping would change
    the experiment.
    """

    kind: str  # 'lora_fused' | 'full_resident' | 'full_offload'
    foreach: bool
    block_params: dict[int, list[torch.nn.Parameter]]
    optimizers: list[torch.optim.Optimizer]

    @classmethod
    def build(cls, stack, cfg, blocks: range | None = None) -> "OptimizerPlan":
        """Resolve the policy table for this config. ``blocks`` restricts to a
        subset (the sequential schedule optimizes one block per stage)."""
        offload = cfg.train.offload_adam
        if cfg.train.lora.enabled and not offload:
            kind, foreach = "lora_fused", True
        elif offload:
            kind, foreach = "full_offload", False
        else:
            # Foreach's extra tensor-list intermediates are negligible for
            # LoRA and expensive for full-FT: large-model memory wins.
            kind, foreach = "full_resident", False
        blocks = blocks if blocks is not None else range(1, stack.n_layers + 1)
        block_params = {
            L: [p for p in stack.block_params(L) if p.requires_grad]
            for L in blocks
        }
        if kind == "full_offload":
            optimizers = [torch.optim.AdamW(params, lr=cfg.train.lr, foreach=False)
                          for params in block_params.values()]
        else:
            all_params = [p for params in block_params.values() for p in params]
            optimizers = [torch.optim.AdamW(all_params, lr=cfg.train.lr,
                                            foreach=foreach)]
        return cls(kind=kind, foreach=foreach, block_params=block_params,
                   optimizers=optimizers)

    def step(self) -> None:
        for params in self.block_params.values():
            torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=self.foreach)
        offload = self.kind == "full_offload"
        for opt in self.optimizers:
            if offload:
                _move_opt_state(opt, "cuda")
            opt.step()
            opt.zero_grad(set_to_none=True)
            if offload:
                _move_opt_state(opt, "cpu")


class TrainingRuntime:
    """Owns the executable side of a run: student model + placement, LoRA,
    teacher source, disk cache, frozen-vocabulary tripwire, VRAM accounting.

    Construction order matches the historical trainer exactly (load → LoRA →
    train() → freeze → signature → teacher → cache); RNG-consuming steps see
    the same global-seed state as before the extraction."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.model.device
        if uses_pipeline_map(cfg) and cfg.model.device_map:
            raise ValueError(
                "model.pipeline_split(s) and model.device_map are mutually exclusive")
        if cfg.model.device_map not in ("", "auto"):
            raise ValueError("model.device_map must be empty or 'auto'")
        self.pp_map = pp_device_map(cfg) if uses_pipeline_map(cfg) else None
        self.auto_map = cfg.model.device_map == "auto"
        # bf16 base for LoRA (frozen weights) AND for the sequential
        # schedules: only actively-training blocks need fp32 master weights
        # (cast per stage / per window); summed full-FT trains all blocks
        # every step and keeps fp32 masters throughout.
        full_ft_all_blocks = (not cfg.train.lora.enabled
                              and cfg.train.schedule != "sequential")
        self.base_dtype = torch.float32 if full_ft_all_blocks else torch.bfloat16
        # warm-start: student weights from a prior run's checkpoint; the
        # teacher (cache identity / frozen copy / adapters-off) stays
        # cfg.model.name
        self.student_src = (str(Path("runs") / cfg.train.init_from / "checkpoint")
                            if cfg.train.init_from else cfg.model.name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
        self.model = None
        self.peft_model = None
        self.stack = None
        self.teacher = None
        self.cache = None
        self._vocab_sig0 = None

    # -- construction ------------------------------------------------------

    def _load_placed(self, src, dtype, **kw):
        if self.pp_map is not None:
            return load_causal_lm(src, dtype=dtype, device_map=self.pp_map, **kw)
        if self.auto_map:
            return load_causal_lm(src, dtype=dtype, device_map="auto", **kw)
        model = load_causal_lm(src, dtype=dtype, **kw)
        model.to(self.device)
        return model

    def load(self, moe_load_kw: dict | None = None) -> "TrainingRuntime":
        moe_load_kw = moe_load_kw or {}
        self.model = self._load_placed(self.student_src, self.base_dtype,
                                       **moe_load_kw)
        if self.cfg.train.lora.enabled:
            from .lora import attach_lora

            self.peft_model = attach_lora(self.model, self.cfg.train.lora)
            self.model = self.peft_model.get_base_model()
        self.model.train()
        self.stack = BlockStack(self.model)
        self.stack.freeze_non_blocks()
        self._vocab_sig0 = vocab_signature(self.stack)
        return self

    def load_teacher(self, moe_load_kw: dict | None = None):
        """Online teacher (adapters-off LoRA base, or a resident frozen bf16
        copy for full-FT) — None when targets come from the disk cache."""
        from .layerwise import OnlineTeacherSource

        cfg = self.cfg
        if cfg.train.online_teacher and self.peft_model is None:
            raise ValueError("train.online_teacher requires train.lora.enabled")
        if cfg.train.online_teacher:
            self.teacher = OnlineTeacherSource(self.stack,
                                               peft_model=self.peft_model)
        elif cfg.train.frozen_teacher_copy:
            t_model = self._load_placed(cfg.model.name, torch.bfloat16,
                                        **(moe_load_kw or {}))
            t_model.eval().requires_grad_(False)
            self.teacher = OnlineTeacherSource(self.stack,
                                               frozen_stack=BlockStack(t_model))
        return self.teacher

    def load_cache(self):
        cache_root, chash = resolve_cache_dir(self.cfg)
        self.cache = TeacherCache(cache_root, expect_hash=chash)
        return self.cache

    @property
    def online(self) -> bool:
        return self.teacher is not None

    def optimizer_plan(self, blocks: range | None = None) -> OptimizerPlan:
        return OptimizerPlan.build(self.stack, self.cfg, blocks=blocks)

    # -- invariants & accounting --------------------------------------------

    def check_vocab_frozen(self) -> None:
        if vocab_signature(self.stack) != self._vocab_sig0:
            raise RuntimeError(
                "frozen-vocabulary violation: embedding/final-norm/head changed "
                "during training — refusing to save (docs/hidden_loss.md)"
            )

    @staticmethod
    def memory_summary() -> dict:
        n_dev = torch.cuda.device_count()
        return {
            # summed across visible cards (pipeline-parallel jobs use several)
            "vram_gb": round(sum(torch.cuda.max_memory_allocated(d)
                                 for d in range(n_dev)) / 2**30, 2),
            # reserved = what the allocator actually holds from the device —
            # the honest footprint for "does it fit on this card" claims
            "vram_reserved_gb": round(sum(torch.cuda.max_memory_reserved(d)
                                          for d in range(n_dev)) / 2**30, 2),
            "vram_per_device_gb": [round(torch.cuda.max_memory_reserved(d) / 2**30, 2)
                                   for d in range(n_dev)],
        }

    def save_checkpoint(self, run_dir: Path) -> None:
        self.check_vocab_frozen()
        if self.peft_model is not None:
            self.peft_model.save_pretrained(run_dir / "checkpoint")
        else:
            self.model.to(torch.bfloat16)
            self.model.save_pretrained(run_dir / "checkpoint")
        self.tokenizer.save_pretrained(run_dir / "checkpoint")
