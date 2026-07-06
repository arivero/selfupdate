"""Per-block "activations-in / activations-out" contract.

``BlockStack`` exposes the pieces of a loaded HF causal LM as independently
runnable stages: embedding, rotary position embeddings, each decoder block,
final norm, lm_head. Blocks are 1-based to match the teacher-cache convention
(``h{L}`` = output of block L; ``h{n_layers}`` is post-final-norm).

This interface is the 120B streaming story: a future weight-streaming runner
only needs to implement load()/offload() per block behind the same
``run_block`` signature. At small scale everything is resident and those are
no-ops.

Batch-1, unpadded sequences: attention_mask=None takes the causal sdpa fast
path inside the attention layers.
"""

from __future__ import annotations

import inspect
from collections import UserDict

import torch


class BlockStack:
    def __init__(self, model):
        self.model = model
        inner = model.model
        if not all(hasattr(inner, attr) for attr in ("embed_tokens", "layers", "norm")):
            wrapped = getattr(inner, "language_model", None)
            if wrapped is not None:
                inner = wrapped
        # this module layout is shared by the Qwen/Llama/DeepSeek/GLM HF
        # ports. Gemma4 wraps the text stack under model.language_model.
        # Fail loudly for exotic architectures rather than mis-wiring.
        for attr, owner in (("embed_tokens", inner), ("layers", inner),
                            ("norm", inner), ("lm_head", model)):
            if not hasattr(owner, attr):
                raise NotImplementedError(
                    f"{type(inner).__name__} lacks .{attr}; add an arch adapter "
                    "(see docs/scaling.md)"
                )
        self.embed_tokens = inner.embed_tokens
        self.text_config = getattr(inner, "config", getattr(model.config, "text_config", model.config))
        self.layer_types = list(getattr(self.text_config, "layer_types", []) or [])
        # MLA-style models compute rotary inside attention; rotary_emb is optional
        self.rotary_emb = getattr(inner, "rotary_emb", None)
        self.rotary_needs_layer_type = (
            self.rotary_emb is not None
            and "layer_type" in inspect.signature(self.rotary_emb.forward).parameters
        )
        self.needs_gemma4_masks = (
            getattr(self.text_config, "model_type", "") == "gemma4_text"
            and bool(self.layer_types)
        )
        self.blocks = list(inner.layers)
        self.final_norm = inner.norm
        self.lm_head = model.lm_head
        self.n_layers = len(self.blocks)

    def freeze_non_blocks(self) -> None:
        """Embedding, final norm and lm_head stay at init: block-only training
        keeps the localization readout clean and matches the teacher's frozen
        h{n_layers} (post-norm with the initial norm weights)."""
        self.embed_tokens.requires_grad_(False)
        self.final_norm.requires_grad_(False)
        self.lm_head.requires_grad_(False)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.embed_tokens(input_ids)

    def rope(self, hidden: torch.Tensor, position_ids: torch.Tensor):
        if self.rotary_emb is None:
            return None  # attention computes rotary internally (MLA-style)
        if self.rotary_needs_layer_type:
            bundle = {"position_ids": position_ids}
            if self.needs_gemma4_masks:
                from transformers.masking_utils import (
                    create_causal_mask,
                    create_sliding_window_causal_mask,
                )

                mask_kwargs = {
                    "config": self.text_config,
                    "inputs_embeds": hidden,
                    "attention_mask": None,
                    "past_key_values": None,
                    "position_ids": position_ids,
                }
                bundle["attention_masks"] = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                    "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
                }
                bundle["shared_kv_states"] = UserDict()
            return bundle
        with torch.no_grad():
            return self.rotary_emb(hidden, position_ids)

    def run_block(self, L: int, hidden, position_embeddings, position_ids=None):
        """Forward decoder block L (1-based) on [B, n, H] hidden states."""
        attention_mask = None
        shared_kv_states = None
        if self.rotary_needs_layer_type and isinstance(position_embeddings, dict):
            bundle = position_embeddings
            position_ids = bundle["position_ids"]
            layer_type = (
                self.layer_types[L - 1] if L - 1 < len(self.layer_types)
                else getattr(getattr(self.blocks[L - 1], "self_attn", None),
                             "layer_type", None)
            )
            masks = bundle.get("attention_masks")
            if masks is not None:
                attention_mask = masks[layer_type]
            shared_kv_states = bundle.get("shared_kv_states")
            with torch.no_grad():
                position_embeddings = self.rotary_emb(
                    hidden, position_ids, layer_type=layer_type)
        kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "use_cache": False,
        }
        if shared_kv_states is not None:
            kwargs["shared_kv_states"] = shared_kv_states
        return self.blocks[L - 1](hidden, **kwargs)

    def block_params(self, L: int) -> list[torch.nn.Parameter]:
        return list(self.blocks[L - 1].parameters())

    def loss_view(self, L: int, block_out: torch.Tensor) -> torch.Tensor:
        """What to compare against the cached teacher h{L}: raw block output,
        except the last layer, whose cached target is post-final-norm."""
        return self.final_norm(block_out) if L == self.n_layers else block_out
