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

import torch


class BlockStack:
    def __init__(self, model):
        self.model = model
        inner = model.model
        # this module layout is shared by the Qwen/Llama/DeepSeek/GLM HF
        # ports; fail loudly for exotic architectures rather than mis-wiring
        for attr, owner in (("embed_tokens", inner), ("layers", inner),
                            ("norm", inner), ("lm_head", model)):
            if not hasattr(owner, attr):
                raise NotImplementedError(
                    f"{type(inner).__name__} lacks .{attr}; add an arch adapter "
                    "(see docs/scaling.md)"
                )
        self.embed_tokens = inner.embed_tokens
        # MLA-style models compute rotary inside attention; rotary_emb is optional
        self.rotary_emb = getattr(inner, "rotary_emb", None)
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
        with torch.no_grad():
            return self.rotary_emb(hidden, position_ids)

    def run_block(self, L: int, hidden, position_embeddings, position_ids=None):
        """Forward decoder block L (1-based) on [B, n, H] hidden states."""
        return self.blocks[L - 1](
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )

    def block_params(self, L: int) -> list[torch.nn.Parameter]:
        return list(self.blocks[L - 1].parameters())

    def loss_view(self, L: int, block_out: torch.Tensor) -> torch.Tensor:
        """What to compare against the cached teacher h{L}: raw block output,
        except the last layer, whose cached target is post-final-norm."""
        return self.final_norm(block_out) if L == self.n_layers else block_out
