"""Per-layer LoRA axis: adapters on every block projection, base frozen.

LoRA B matrices initialize to zero, so attaching adapters leaves the forward
pass identical until training moves them — the same-initial-weights alignment
with the teacher cache is preserved exactly.
"""

from __future__ import annotations

import math
import re

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# MLA families name their attention projections by the latent decomposition;
# without these the adapter would only reach o_proj and the MLP — a silently
# weaker attention adapter (plan B8).  V3-era naming:
MLA_V3_TARGET_MODULES = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa",
                         "kv_b_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]
# DeepSeek-V4-Flash renames the MLA stack: kv_proj (single shared-KV head),
# o_a_proj (grouped block-diagonal — EXCLUDED: a dense LoRA delta would break
# the grouped structure) + o_b_proj.  Its compressor/indexer submodules also
# contain kv_proj/gate_proj leaves that must NOT get adapters: they are
# key-side (compressed entries are served frozen from the teacher), so their
# adapters would train against a signal the frozen context never lets flow.
MLA_V4_TARGET_MODULES = ["q_a_proj", "q_b_proj", "kv_proj", "o_b_proj",
                         "gate_proj", "up_proj", "down_proj"]
_KV_SIDE_SUBMODULES = (".compressor.", ".indexer.")
TARGET_LEAVES = tuple(set(TARGET_MODULES) | set(MLA_V3_TARGET_MODULES)
                      | set(MLA_V4_TARGET_MODULES))

_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")


def _owned_targets(model, owned_layers) -> list[str]:
    """Exact adapter targets restricted to the owned decoder layers.

    Stage-scoped loads (shard_load.py) leave foreign blocks on the meta
    device; PEFT would happily create adapters on meta Linears and the
    stage checkpoint would then carry dead zero shards for blocks it never
    trains. Matching by exact name keeps adapters exactly where gradients
    can flow. Vision towers are excluded by name and by requiring plain
    nn.Linear (gemma4 vision wraps projections in Gemma4ClippableLinear)."""
    import torch

    targets = []
    for name, module in model.named_modules():
        if name.split(".")[-1] not in TARGET_LEAVES:
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if "visual" in name or "vision" in name:
            continue
        if any(part in name for part in _KV_SIDE_SUBMODULES):
            continue
        m = _LAYER_RE.search(name)
        if m is None or int(m.group(1)) not in owned_layers:
            continue
        targets.append(name)
    if not targets:
        raise ValueError(
            f"owned-layer LoRA target discovery found no projections for "
            f"layers {sorted(owned_layers)[:4]}...")
    return targets


def _target_modules(model):
    """Gemma 4's vision tower wraps projections in Gemma4ClippableLinear,
    while the text stack uses ordinary Linear projections. Return exact
    text-stack module names so PEFT does not inject adapters into vision."""
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if model_type == "deepseek_v3":
        return MLA_V3_TARGET_MODULES
    if model_type == "deepseek_v4":
        import torch

        # Exact names: the compressor/indexer submodules reuse the
        # kv_proj/gate_proj leaf names, and PEFT's suffix matching would
        # silently adapt them (see _KV_SIDE_SUBMODULES note above).
        targets = []
        for name, module in model.named_modules():
            if name.split(".")[-1] not in MLA_V4_TARGET_MODULES:
                continue
            if not isinstance(module, torch.nn.Linear):
                continue
            if any(part in name for part in _KV_SIDE_SUBMODULES):
                continue
            if ".layers." not in name:
                continue
            targets.append(name)
        if not targets:
            raise ValueError(
                "DeepSeek-V4 LoRA target discovery found no projections")
        return targets
    if model_type != "gemma4":
        return TARGET_MODULES
    import torch

    prefix = "model.language_model.layers."
    targets = []
    for name, module in model.named_modules():
        if not name.startswith(prefix):
            continue
        parts = name.split(".")
        if len(parts) < 1 or parts[-1] not in TARGET_LEAVES:
            continue
        if isinstance(module, torch.nn.Linear):
            targets.append(name)
    if not targets:
        raise ValueError("Gemma 4 LoRA target discovery found no text projection .linear modules")
    return targets


def _canonical_target_specs(model) -> list[tuple[str, int, int]]:
    """Exact PEFT traversal order and Linear shapes for a full-model attach.

    ``_target_modules`` may return exact paths (Gemma/DeepSeek-V4) or leaf
    suffixes (the ordinary decoder families).  PEFT visits ``named_modules``
    in model order and applies the same exact-or-suffix match.  Reconstructing
    that ordered list lets a stage-scoped attach preserve the full attach's
    RNG stream without creating adapters on foreign meta blocks.
    """
    import torch

    selected = list(_target_modules(model))
    exact = set(selected)
    specs = []
    for name, module in model.named_modules():
        if name not in exact and not any(
                name.endswith(f".{target}") for target in selected):
            continue
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(
                "stage-scoped LoRA RNG equivalence currently requires "
                f"ordinary Linear targets; {name!r} is "
                f"{type(module).__name__}")
        specs.append((name, int(module.in_features),
                      int(module.out_features)))
    if not specs:
        raise ValueError("canonical full-model LoRA target list is empty")
    return specs


def _consume_linear_lora_init(specs: list[tuple[str, int, int]],
                              rank: int) -> None:
    """Advance CPU RNG exactly as PEFT 0.19.1 initializes Linear LoRA.

    For each target PEFT constructs A and B (one ``nn.Linear`` reset each),
    then ``reset_lora_parameters`` initializes A a second time with the same
    Kaiming law and zeros B.  The zeroing consumes no RNG.  Keeping these
    throwaway tensors one target at a time bounds temporary memory to one
    adapter pair.
    """
    import torch

    for _name, in_features, out_features in specs:
        lora_a = torch.nn.Linear(in_features, rank, bias=False)
        lora_b = torch.nn.Linear(rank, out_features, bias=False)
        torch.nn.init.kaiming_uniform_(
            lora_a.weight, a=math.sqrt(5))
        del lora_a, lora_b


def attach_lora(model, lora_cfg, owned_layers=None):
    from peft import LoraConfig, get_peft_model

    if owned_layers is None:
        # Historical full-model path: preserve its initialization byte for
        # byte.  Stage-scoped equivalence is implemented only in the branch
        # below.
        targets = _target_modules(model)
        prefix_specs = suffix_specs = []
    else:
        owned = set(owned_layers)
        targets = _owned_targets(model, owned)
        canonical = _canonical_target_specs(model)
        canonical_names = [name for name, _, _ in canonical]
        expected_owned = [
            name for name in canonical_names
            if ((match := _LAYER_RE.search(name)) is not None
                and int(match.group(1)) in owned)
        ]
        if targets != expected_owned:
            raise RuntimeError(
                "stage-scoped LoRA targets are not the owned subsequence of "
                "the canonical full-model attach; refusing an RNG-shifted "
                f"run (scoped={targets[:4]!r}, "
                f"canonical_owned={expected_owned[:4]!r})")
        positions = [canonical_names.index(name) for name in targets]
        first, last = positions[0], positions[-1] + 1
        if positions != list(range(first, last)):
            raise RuntimeError(
                "stage-scoped LoRA ownership is not contiguous in canonical "
                "target order; cannot preserve full-model RNG exactly")
        prefix_specs = canonical[:first]
        suffix_specs = canonical[last:]
        _consume_linear_lora_init(prefix_specs, int(lora_cfg.r))

    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    if owned_layers is not None:
        # Leave the process-wide RNG at the same point as a canonical full
        # attach too; later dropout or other stochastic machinery must not
        # learn which loading path was used.
        _consume_linear_lora_init(suffix_specs, int(lora_cfg.r))
    return peft_model
