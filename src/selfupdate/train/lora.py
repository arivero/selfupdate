"""Per-layer LoRA axis: adapters on every block projection, base frozen.

LoRA B matrices initialize to zero, so attaching adapters leaves the forward
pass identical until training moves them — the same-initial-weights alignment
with the teacher cache is preserved exactly.
"""

from __future__ import annotations

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TARGET_LEAVES = tuple(TARGET_MODULES)


def _target_modules(model):
    """Gemma 4's vision tower wraps projections in Gemma4ClippableLinear,
    while the text stack uses ordinary Linear projections. Return exact
    text-stack module names so PEFT does not inject adapters into vision."""
    if getattr(getattr(model, "config", None), "model_type", "") != "gemma4":
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


def attach_lora(model, lora_cfg):
    from peft import LoraConfig, get_peft_model

    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=_target_modules(model),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    return peft_model
