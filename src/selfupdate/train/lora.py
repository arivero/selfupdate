"""Per-layer LoRA axis: adapters on every block projection, base frozen.

LoRA B matrices initialize to zero, so attaching adapters leaves the forward
pass identical until training moves them — the same-initial-weights alignment
with the teacher cache is preserved exactly.
"""

from __future__ import annotations

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def attach_lora(model, lora_cfg):
    from peft import LoraConfig, get_peft_model

    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    return peft_model
