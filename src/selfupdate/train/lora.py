"""Per-layer LoRA axis: adapters on every block projection, base frozen.

LoRA B matrices initialize to zero, so attaching adapters leaves the forward
pass identical until training moves them — the same-initial-weights alignment
with the teacher cache is preserved exactly.
"""

from __future__ import annotations

import re

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# MLA families (deepseek_v4) name their attention projections by the
# latent decomposition; without these the adapter would only reach o_proj
# and the MLP — a silently weaker attention adapter (plan B8).
MLA_TARGET_MODULES = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa",
                      "kv_b_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
TARGET_LEAVES = tuple(set(TARGET_MODULES) | set(MLA_TARGET_MODULES))

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
    if model_type in ("deepseek_v4", "deepseek_v3"):
        return MLA_TARGET_MODULES
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


def attach_lora(model, lora_cfg, owned_layers=None):
    from peft import LoraConfig, get_peft_model

    targets = (_owned_targets(model, set(owned_layers))
               if owned_layers is not None else _target_modules(model))
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
    return peft_model
