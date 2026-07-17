"""Stage-scoped model loading for pipeline-v4 (plan B3, 2026-07-17).

A v4 stage process trains only its owned contiguous block range, yet
``TrainingRuntime.load`` historically materialized the FULL model on one
card — fatal at Qwen3.5-122B (~250 GB bf16) and absurd at 397B (~807 GB).
This module materializes ONLY:

- the owned decoder blocks (``model.layers.{i}`` for i in the owned
  0-based range),
- the vocabulary stack everywhere (embed_tokens, final norm, lm_head —
  every stage embeds for capture/relay and the last stage evaluates
  CE/KL through the frozen head; cost is a few GB),
- tied-weight aliases of the above.

Everything else stays on the meta device: zero bytes, and any accidental
compute touch fails loudly (meta tensors reject kernels), which is exactly
the failure mode we want for "a foreign block was executed on this stage".

Mechanics: build the model skeleton under ``init_empty_weights`` (buffers
real — rotary ``inv_freq`` and friends are computed, not stored), map
checkpoint tensor names to module names via the class's own
``_checkpoint_conversion_mapping`` (multimodal repos rename e.g.
``model.language_model.*``), open only the safetensors shards holding kept
tensors, and materialize with ``load_state_dict(..., assign=True)``.
``safetensors.torch.load_file`` returns mmap-backed tensors, so the kept
CPU masters share the snapshot's page cache across stage processes instead
of copying it per process — the property that keeps four 397B stages
inside one node's RAM.

Vision towers: the CausalLM classes of the multimodal families
(``Qwen3_5MoeForCausalLM`` etc.) are text-only and list vision/mtp weights
in ``_keys_to_ignore_on_load_unexpected`` — those checkpoint tensors are
simply never read here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

__all__ = ["stage_scoped_load", "assert_materialized", "owned_layer_names"]

_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")


def _snapshot_dir(model_name: str) -> Path:
    """Resolve the local snapshot directory (offline; no Hub traffic)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_name, local_files_only=True))


def _conversion(model) -> list[tuple[re.Pattern, str]]:
    mapping = getattr(model, "_checkpoint_conversion_mapping", None) or {}
    return [(re.compile(pat), repl) for pat, repl in mapping.items()]


def _to_module_key(ckpt_key: str, conv) -> str:
    for pat, repl in conv:
        ckpt_key = pat.sub(repl, ckpt_key)
    return ckpt_key


def owned_layer_names(owned0: range) -> set[str]:
    """0-based owned indices -> the set of decoder-layer name prefixes."""
    return {f"layers.{i}." for i in owned0}


def _keep(module_key: str, owned0: range) -> bool:
    m = _LAYER_RE.search(module_key)
    if m is not None:
        return int(m.group(1)) in owned0
    # Non-layer text weights: embeddings, final norm, lm_head, rotary — the
    # vocabulary stack plus scalars. Vision/mtp keys never reach here (they
    # are dropped at the unexpected-key filter below).
    return True


def stage_scoped_load(model_name: str, owned0: range, *, dtype,
                      trust_remote_code: bool = False):
    """Build the causal-LM skeleton and materialize only what this stage
    needs. Returns the model on CPU: owned blocks + vocab stack are real
    (mmap-backed), unowned blocks are meta. The caller decides device
    placement (resident ``.to(device)`` per owned block, or rotation).

    ``owned0`` is the ZERO-based decoder-layer range (the trainer's 1-based
    ``owned`` minus one).
    """
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=trust_remote_code)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=dtype,
            trust_remote_code=trust_remote_code)
    model.eval()

    snap = _snapshot_dir(model_name)
    index_path = snap / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        single = sorted(snap.glob("*.safetensors"))
        if not single:
            raise FileNotFoundError(f"no safetensors in snapshot {snap}")
        from safetensors import safe_open

        weight_map = {}  # filled per-shard below
        for f in single:
            with safe_open(f, framework="pt") as handle:
                for k in handle.keys():
                    weight_map[k] = f.name

    conv = _conversion(model)
    expected = set(model.state_dict().keys())
    ignore = [re.compile(p) for p in
              (getattr(model, "_keys_to_ignore_on_load_unexpected", None)
               or [])]

    by_shard: dict[str, list[tuple[str, str]]] = {}
    for ckpt_key, shard in weight_map.items():
        module_key = _to_module_key(ckpt_key, conv)
        if module_key not in expected:
            if any(p.search(ckpt_key) or p.search(module_key)
                   for p in ignore):
                continue  # vision tower / mtp — text-only class skips them
            raise KeyError(
                f"checkpoint tensor {ckpt_key!r} (-> {module_key!r}) not in "
                f"{type(model).__name__} state dict and not ignorable — the "
                "conversion mapping is incomplete for this family")
        if _keep(module_key, owned0):
            by_shard.setdefault(shard, []).append((ckpt_key, module_key))

    partial: dict[str, torch.Tensor] = {}
    for shard, pairs in sorted(by_shard.items()):
        tensors = load_file(snap / shard, device="cpu")  # mmap-backed
        for ckpt_key, module_key in pairs:
            t = tensors[ckpt_key]
            if t.dtype != dtype and t.is_floating_point():
                # Quantized checkpoints (fp8/fp4 experts) are NOT handled
                # here — they carry non-float or scale tensors and need the
                # dequant lane (plan B8). Plain half/bf16/f32 mismatches
                # convert (copy: conversion cannot stay mmap-backed).
                t = t.to(dtype)
            partial[module_key] = t

    missing = {k for k in expected
               if _keep(k, owned0) and k not in partial}
    # Tied weights (lm_head <- embed_tokens) are absent from checkpoints by
    # design; load_state_dict(assign=True) + tie_weights restores them.
    tied = {k for k in missing if "lm_head" in k}
    missing -= tied
    if missing:
        raise KeyError(
            f"stage-scoped load left {len(missing)} kept tensors "
            f"unmaterialized, e.g. {sorted(missing)[:4]}")

    model.load_state_dict(partial, strict=False, assign=True)
    if tied:
        model.tie_weights()
    _apply_fp32_strict(model)
    assert_materialized(model, owned0)
    return model


def _apply_fp32_strict(model) -> None:
    """Honor ``_keep_in_fp32_modules_strict``: DeepSeek-V4 pins its mHC
    hyper-connections, sinks, position biases and small norms in fp32
    (Sinkhorn/softmax stability).  ``from_pretrained`` upcasts these during
    a normal load; the raw shard path must do it itself or the stage would
    train against subtly different fp32-sensitive numerics."""
    patterns = (getattr(model, "_keep_in_fp32_modules_strict", None)
                or getattr(model, "_keep_in_fp32_modules", None) or [])
    if not patterns:
        return
    import itertools

    for name, t in itertools.chain(model.named_parameters(),
                                   model.named_buffers()):
        if t.is_meta or not t.is_floating_point() \
                or t.dtype == torch.float32:
            continue
        if any(part in name.split(".") for part in patterns):
            t.data = t.data.to(torch.float32)


def assert_materialized(model, owned0: range) -> None:
    """Owned blocks + vocab stack real; unowned blocks meta. Loud names."""
    bad_meta, bad_real = [], []
    for name, p in model.named_parameters():
        m = _LAYER_RE.search(name)
        if m is None:
            if p.is_meta:
                bad_meta.append(name)
            continue
        idx = int(m.group(1))
        if idx in owned0 and p.is_meta:
            bad_meta.append(name)
        elif idx not in owned0 and not p.is_meta:
            bad_real.append(name)
    if bad_meta:
        raise RuntimeError(
            f"stage-scoped load: {len(bad_meta)} kept params are still meta, "
            f"e.g. {bad_meta[:4]}")
    if bad_real:
        raise RuntimeError(
            f"stage-scoped load: {len(bad_real)} foreign-block params were "
            f"materialized, e.g. {bad_real[:4]} — the keep predicate leaked")
