"""Logit-lens depth profile: at which layer does the memory become readable?

For each layer L, apply the model's final norm + lm_head to the layer-L hidden
state at aligned positions of the *student* input (no context) and measure the
mean log-probability of the gold next token. Before training the profile
should be flat and low; after training the depth at which it rises tells where
recall is assembled. (Raw logit lens; a trained tuned-lens probe per layer is
a planned upgrade — raw readouts of early layers are known to be brittle.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def gold_logprob_by_layer(model, tokenizer, pairs, device="cuda", limit=32,
                          rebase_gap: bool = False, translators=None) -> dict:
    """pairs: list[AlignedPair]. Returns {layer: mean gold-token logprob}.
    ``rebase_gap`` must match the training compaction (stub_gap trains at
    gap-shifted positions; probing at contiguous ones would measure an
    untrained geometry). ``translators``: a tuned-lens ModuleDict
    (train/tuned_lens.py); zero-init translators reproduce the raw lens
    exactly (delta parameterization)."""
    from ..train.tuned_lens import apply_translator

    n_layers = model.config.num_hidden_layers
    inner = model.model
    sums = torch.zeros(n_layers + 1, dtype=torch.float64)
    count = 0
    for pair in pairs[:limit]:
        ids = torch.tensor([pair.student_ids], device=device)
        pos = torch.tensor([pair.student_position_ids(rebase_gap)], device=device)
        out = model(ids, position_ids=pos, output_hidden_states=True, use_cache=False)
        s = pair.s_aligned
        gold = torch.tensor(
            pair.student_ids[s.start + 1: s.stop], device=device
        )  # tokens predicted from positions [s0, s0+A-1)
        for L in range(1, n_layers + 1):
            h = out.hidden_states[L][0, s.start: s.stop - 1]
            if L < n_layers:
                h = apply_translator(translators, L, h)
                h = inner.norm(h)  # hidden_states[n_layers] is already post-norm
            logits = model.lm_head(h)
            lp = F.log_softmax(logits.float(), -1)
            sums[L] += lp.gather(1, gold[:, None]).mean().item()
        count += 1
    return {L: sums[L].item() / count for L in range(1, n_layers + 1)}
