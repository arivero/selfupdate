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

import os
import json
import re
from pathlib import Path

import torch

__all__ = ["stage_scoped_load", "assert_materialized", "owned_layer_names"]

_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")
# Per-expert routed-MoE weights in an UNFUSED checkpoint (e.g. Qwen3.5-397B
# fp8, which stores `...experts.{e}.gate_proj.weight` per expert). The HF
# Qwen3_5Moe class expects them FUSED into stacked params
# (`experts.gate_up_proj` [E, 2*I, H] and `experts.down_proj` [E, H, I]) and
# does that fusion in a from_pretrained load hook that the raw assign-load
# here bypasses. `_EXPERT_RE` recognizes the unfused components so we can
# assemble the fused targets ourselves. (2026-07-18)
_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.experts)\.(?P<idx>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$")


def _snapshot_dir(model_name: str) -> Path:
    """Resolve the local snapshot directory (offline; no Hub traffic).

    allow_patterns limits the hub's snapshot-completeness check to the
    files this loader actually reads: `hf download` snapshots routinely
    lack README/.gitattributes/eval yamls, and the strict whole-repo check
    would refuse an otherwise complete weights cache (hit 2026-07-18 on
    gemma-4-31B-it; from_pretrained never notices because it fetches
    per-file)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import IncompleteSnapshotError

    if Path(model_name).is_dir():
        return Path(model_name)
    try:
        return Path(snapshot_download(
            model_name, local_files_only=True,
            allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt",
                            "*.model"]))
    except IncompleteSnapshotError:
        # A staged snapshot can lack an OPTIONAL repo file that still
        # matches allow_patterns (Qwen ships `configuration.json` for
        # trust_remote_code; hf download skips it). The weights are all
        # present — resolve the local snapshot dir directly rather than
        # refusing (2026-07-18: killed 35B PPP4). Pick the newest snapshot
        # dir that actually holds the safetensors index.
        from huggingface_hub.constants import HF_HUB_CACHE

        repo = "models--" + model_name.replace("/", "--")
        snaps = sorted(
            (Path(os.environ.get("HF_HOME", "")) / "hub" / repo
             / "snapshots").glob("*") if os.environ.get("HF_HOME")
            else (Path(HF_HUB_CACHE) / repo / "snapshots").glob("*"),
            key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snaps:
            if (snap / "model.safetensors.index.json").exists() or list(
                    snap.glob("*.safetensors")):
                return snap
        raise


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
    from safetensors.torch import load as _st_load
    from transformers import AutoConfig, AutoModelForCausalLM

    def load_shard_seq(path: Path) -> dict:
        # Read the whole shard SEQUENTIALLY (one bulk read) then deserialize
        # from RAM. `safetensors` get_tensor/mmap reads each tensor at its own
        # offset; on an UNFUSED-expert checkpoint that is 1536 tiny scattered
        # reads per layer, which measured ~78 MB/s on Lustre vs 368 MB/s for a
        # sequential shard read (2026-07-18, 397B). Scattered reads also make
        # per-stage load times diverge, which times out the cross-node NCCL
        # rendezvous. The bulk read costs deserializing a few unowned tensors
        # (cheap memcpy) but keeps every stage's load fast and uniform.
        return _st_load(path.read_bytes())

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
    sd_meta = model.state_dict()
    expected = set(sd_meta.keys())
    ignore = [re.compile(p) for p in
              (getattr(model, "_keys_to_ignore_on_load_unexpected", None)
               or [])]

    def _resolve(module_key: str) -> str:
        # Multimodal repos prefix text weights with `language_model.`;
        # some text-only classes (Qwen3_5MoeForCausalLM) expect the
        # stripped name but ship no conversion entry for it
        # (hit 2026-07-18 on Qwen3.5-122B `mlp.experts.gate_up_proj`).
        if module_key not in expected:
            stripped = module_key.replace("model.language_model.", "model.")
            if stripped != module_key and stripped in expected:
                return stripped
        return module_key

    # A fused-expert component is described by (fused_target, expert_idx,
    # slot) where slot in {"gate","up","down"}; assembled after loading.
    by_shard: dict[str, list[tuple[str, str]]] = {}
    fuse_by_shard: dict[str, list[tuple[str, str, int, str]]] = {}
    fuse_targets: set[str] = set()
    for ckpt_key, shard in weight_map.items():
        module_key = _resolve(_to_module_key(ckpt_key, conv))
        if module_key in expected:
            if _keep(module_key, owned0):
                by_shard.setdefault(shard, []).append((ckpt_key, module_key))
            continue
        # Not a direct match — is it an unfused routed-expert component whose
        # FUSED target the model expects?
        m = _EXPERT_RE.match(module_key)
        if m is not None:
            prefix = m.group("prefix")
            proj = m.group("proj")
            fused_target = _resolve(
                f"{prefix}.{'down_proj' if proj == 'down_proj' else 'gate_up_proj'}")
            if fused_target in expected and _keep(fused_target, owned0):
                slot = {"gate_proj": "gate", "up_proj": "up",
                        "down_proj": "down"}[proj]
                fuse_by_shard.setdefault(shard, []).append(
                    (ckpt_key, fused_target, int(m.group("idx")), slot))
                fuse_targets.add(fused_target)
                continue
            if fused_target in expected:
                continue  # a foreign (unowned) layer's expert — skip
        if any(p.search(ckpt_key) or p.search(module_key) for p in ignore):
            continue  # vision tower / mtp — text-only class skips them
        raise KeyError(
            f"checkpoint tensor {ckpt_key!r} (-> {module_key!r}) not in "
            f"{type(model).__name__} state dict and not ignorable — the "
            "conversion mapping is incomplete for this family")

    partial: dict[str, torch.Tensor] = {}
    # Pre-allocate the fused expert tensors from the model's own shapes; fill
    # per-expert slices as components stream in (experts may span shards).
    #   gate_up_proj [E, 2I, H]: gate -> rows 0:I, up -> rows I:2I
    #   down_proj    [E, H,  I]: down -> [e]
    fuse_filled: dict[str, int] = {}
    for tgt in fuse_targets:
        partial[tgt] = torch.empty(sd_meta[tgt].shape, dtype=dtype)
        fuse_filled[tgt] = 0

    for shard in sorted(set(by_shard) | set(fuse_by_shard)):
        tensors = load_shard_seq(snap / shard)  # sequential bulk read
        for ckpt_key, module_key in by_shard.get(shard, []):
            t = tensors[ckpt_key]
            if t.dtype != dtype and t.is_floating_point():
                # Plain half/bf16/f32 mismatches convert (copy: conversion
                # cannot stay mmap-backed). Quantized (fp8/fp4) tensors never
                # reach here — the dequant lane (B8) writes a bf16 snapshot.
                t = t.to(dtype)
            partial[module_key] = t
        for ckpt_key, tgt, idx, slot in fuse_by_shard.get(shard, []):
            t = tensors[ckpt_key]
            if t.dtype != dtype and t.is_floating_point():
                t = t.to(dtype)
            fused = partial[tgt]
            if slot == "down":
                fused[idx].copy_(t)
            else:
                half = fused.shape[1] // 2  # = moe_intermediate_size (I)
                if slot == "gate":
                    fused[idx, 0:half].copy_(t)
                else:  # up
                    fused[idx, half:2 * half].copy_(t)
            fuse_filled[tgt] += 1

    # Each fused gate_up target needs 2E components (gate+up per expert);
    # each down target needs E. A short-count means a shard was missing.
    for tgt in fuse_targets:
        E = int(sd_meta[tgt].shape[0])
        need = E if tgt.endswith("down_proj") else 2 * E
        if fuse_filled[tgt] != need:
            raise KeyError(
                f"fused expert {tgt!r} got {fuse_filled[tgt]}/{need} "
                "components — an expert shard is missing")

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
