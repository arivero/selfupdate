"""Pipeline-v4 model loading, cache access, invariants, and publication.

Each v4 process owns one physical device.  It either loads a complete model
on that device or materializes only its stage-owned blocks for the rotation
lane.  Layer-local optimizer construction remains in :mod:`online_v4`.
"""

from __future__ import annotations

import shutil
import tempfile
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


def vocab_signature(stack) -> tuple:
    """Cheap exact fingerprint of every frozen token/vocabulary tensor.

    Besides embedding, final norm and head, this includes architecture-owned
    token-identity inputs (Gemma PLE) and the mHC collapse head.  They are all
    part of the immutable mapping from token ids to vocabulary logits.
    """
    sig = []
    seen: set[int] = set()
    modules = [stack.embed_tokens, stack.final_norm, stack.lm_head,
               getattr(stack, "hc_head", None)]
    modules.extend(getattr(stack, "frozen_input_modules", []))
    for m in modules:
        if m is None:
            continue
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


class TrainingRuntime:
    """Own one v4 process's model, cache, invariants, and publication."""

    def __init__(self, cfg):
        self.cfg = cfg
        if getattr(cfg.model, "device_map", ""):
            raise ValueError(
                "pipeline-v4 runtime is one process per physical device; "
                "model.device_map must be empty")
        self.device = torch.device(cfg.model.device)
        if self.device.type == "cuda":
            # Pin the default device: deviceless CUDA Events/Streams/tensors
            # must not create a context on a different stage's card.
            torch.cuda.set_device(self.device)
            index = (self.device.index if self.device.index is not None
                     else torch.cuda.current_device())
            self.owned_devices = (index,)
        else:
            self.owned_devices = ()
        self.base_dtype = (torch.bfloat16 if cfg.train.lora.enabled
                           else torch.float32)
        self.student_src = (str(Path("runs") / cfg.train.init_from / "checkpoint")
                            if cfg.train.init_from else cfg.model.name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
        self.model = None
        self.peft_model = None
        self.stack = None
        self.cache = None
        self.cache_manifest = None
        self._vocab_sig0 = None

    # -- construction ------------------------------------------------------

    def _load_placed(self, src, dtype, **kw):
        model = load_causal_lm(src, dtype=dtype, **kw)
        model.to(self.device)
        return model

    def _load_stage_scoped(self):
        """Scaling lane (plan B3): materialize only this stage's owned
        blocks + the vocab stack; foreign blocks stay meta (zero bytes,
        loud on touch). LoRA attaches to owned layers only, then every
        real tensor moves to the stage device — `.to(device)` on the whole
        model would raise on meta params, so movement is per-tensor."""
        from transformers import AutoConfig

        from .online_v4 import _owned_range
        from .shard_load import stage_scoped_load

        if self.cfg.train.init_from:
            raise NotImplementedError(
                "stage-scoped warm start needs per-stage checkpoint "
                "assembly; init_from is not wired for v4_stage_scoped yet")
        acfg = AutoConfig.from_pretrained(self.cfg.model.name)
        text = getattr(acfg, "text_config", None) or acfg
        owned = _owned_range(self.cfg, int(text.num_hidden_layers))
        owned0 = range(owned.start - 1, owned.stop - 1)  # 1-based -> 0-based
        model = stage_scoped_load(self.cfg.model.name, owned0,
                                  dtype=self.base_dtype)
        if self.cfg.train.lora.enabled:
            from .lora import attach_lora

            self.peft_model = attach_lora(model, self.cfg.train.lora,
                                          owned_layers=owned0)
            model = self.peft_model.get_base_model()
        # Residency: 'resident' moves everything real to the card; 'rotate'
        # leaves owned FROZEN block weights/buffers on host (mmap masters —
        # BlockRotator pages them per layer_major visit); 'auto' measures.
        # Trainable (LoRA) params and the vocab stack always live on-card.
        import re as _re

        layer_re = _re.compile(r"\blayers\.(\d+)\.")
        residency = getattr(self.cfg.train, "v4_weight_residency", "resident")
        if residency == "auto":
            from .rotation import decide_rotate

            owned_bytes = sum(
                p.numel() * p.element_size()
                for name, p in model.named_parameters()
                if not p.is_meta and not p.requires_grad
                and layer_re.search(name))
            residency = ("rotate"
                         if decide_rotate(owned_bytes, self.device)
                         else "resident")
        rotate = residency == "rotate"
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.is_meta:
                    continue
                if rotate and not p.requires_grad and layer_re.search(name):
                    continue  # CPU master; BlockRotator owns placement
                p.data = p.data.to(self.device)
            for name, b in model.named_buffers():
                if b.is_meta:
                    continue
                if rotate and layer_re.search(name):
                    continue
                b.data = b.data.to(self.device)
        return model

    def load(self, moe_load_kw: dict | None = None) -> "TrainingRuntime":
        moe_load_kw = moe_load_kw or {}
        if getattr(self.cfg.train, "v4_stage_scoped", False):
            self.model = self._load_stage_scoped()
        else:
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
        self.assert_own_gpu_only("post_load")
        return self

    def assert_own_gpu_only(self, phase: str) -> None:
        """Owner defect-hunt 2026-07-18 ("catch it definitely"): raise if
        THIS process holds memory on any CUDA device other than its
        assigned one. Uses nvidia-smi because a bare ~518 MB CUDA context
        is invisible to torch's allocator counters."""
        import os
        import subprocess

        if self.device.type != "cuda":
            return
        try:
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout
            gpus = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return  # no NVML — the tripwire is best-effort
        index_of = {}
        for line in gpus.strip().splitlines():
            idx, uuid = (part.strip() for part in line.split(","))
            index_of[uuid] = int(idx)
        me = os.getpid()
        foreign = sorted({
            index_of[uuid] for uuid, pid in
            (tuple(part.strip() for part in line.split(","))
             for line in apps.strip().splitlines() if line.strip())
            if pid.isdigit() and int(pid) == me
            and index_of.get(uuid) not in (None, self.owned_devices[0])})
        if foreign:
            raise RuntimeError(
                f"stray CUDA context: pid {me} holds memory on "
                f"cuda:{foreign} while pinned to {self.device} "
                f"(phase={phase}) — a deviceless CUDA call ran off-card")

    def load_cache(self):
        cache_root, chash = resolve_cache_dir(self.cfg)
        if self.cfg.cache.runtime_policy == "node_epoch0":
            from ..teacher.node_epoch0 import ready_manifest, runtime_identity

            ready = ready_manifest(
                cache_root, chash, compatibility=runtime_identity())
            if ready is None:
                raise RuntimeError(
                    "node-local epoch-zero teacher cache is not ready at "
                    f"{cache_root}; run scripts/build_teacher_cache.py with "
                    "--coordinated-node-cache under this node's GPU runtime")
            self.cache_manifest = ready
        self.cache = TeacherCache(cache_root, expect_hash=chash)
        return self.cache

    # -- invariants & accounting --------------------------------------------

    def check_vocab_frozen(self) -> None:
        if vocab_signature(self.stack) != self._vocab_sig0:
            raise RuntimeError(
                "frozen-vocabulary violation: embedding/final-norm/head changed "
                "during training — refusing to save (docs/hidden_loss.md)"
            )

    def memory_summary(self) -> dict:
        devices = self.owned_devices
        return {
            # Each v4 process accounts only for its assigned physical GPU.
            "vram_gb": round(sum(torch.cuda.max_memory_allocated(d)
                                 for d in devices) / 2**30, 2),
            # reserved = what the allocator actually holds from the device —
            # the honest footprint for "does it fit on this card" claims
            "vram_reserved_gb": round(sum(torch.cuda.max_memory_reserved(d)
                                          for d in devices) / 2**30, 2),
            "vram_per_device_gb": [
                round(torch.cuda.max_memory_reserved(d) / 2**30, 2)
                for d in devices
            ],
            "vram_physical_devices": list(devices),
        }

    def save_checkpoint(self, run_dir: Path) -> None:
        """Publish a complete checkpoint atomically to queue consumers.

        The scheduler treats ``run_dir/checkpoint`` as its completion signal.
        Saving directly there lets a dependent evaluator see the directory
        after the first file but before the tokenizer/model is complete.  Write
        a sibling staging directory, then rename it on the same filesystem so
        the signal means a loadable checkpoint, not merely an in-progress save.
        """
        self.check_vocab_frozen()
        target = run_dir / "checkpoint"
        fold_in = None
        if target.exists():
            # The v4 trainer writes Adam moments to checkpoint/ before this
            # publish runs (they pair with the adapter for warm-start). A
            # checkpoint dir holding ONLY that sibling artifact is our own
            # in-progress save, not a prior publication — fold it in rather
            # than refuse (2026-07-18: crashed every completed rotary run).
            contents = list(target.iterdir())
            if contents and all(p.name == "adam_moments.pt" for p in contents):
                fold_in = target / "adam_moments.pt"
            else:
                raise FileExistsError(
                    "refusing to replace existing checkpoint publication: "
                    f"{target}")
        staging = Path(tempfile.mkdtemp(
            prefix=".checkpoint.incomplete-", dir=run_dir))
        try:
            if self.peft_model is not None:
                self.peft_model.save_pretrained(staging)
            else:
                self.model.to(torch.bfloat16)
                self.model.save_pretrained(staging)
            self.tokenizer.save_pretrained(staging)
            if fold_in is not None:
                shutil.move(str(fold_in), str(staging / "adam_moments.pt"))
                target.rmdir()  # now empty; rename can claim the name
            staging.rename(target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
