"""Strictly local frozen-base hidden-state preservation on anchor text."""

from __future__ import annotations

from pathlib import Path

import torch

from .losses import hidden_match
from .teacher_source import OnlineTeacherSource


class AnchorBank:
    """Tokenized neighbor-genre fragments (blank-line separated), cycled at
    optimizer-step boundaries for the anti-intrusion anchor."""

    def __init__(self, path, tok, device, max_tokens: int = 96):
        texts = [t.strip() for t in Path(path).read_text(encoding="utf-8").split("\n\n")
                 if t.strip()]
        if not texts:
            raise ValueError(f"no anchor fragments in {path}")
        self.ids = [torch.tensor(tok.encode(t, add_special_tokens=False)[:max_tokens],
                                 device=device) for t in texts]
        self.base_states: list[dict[int, torch.Tensor]] | None = None
        self.i = 0

    @torch.no_grad()
    def precompute_base_states(self, teacher: OnlineTeacherSource):
        """Base-model hidden states per fragment, computed once."""
        st = teacher.stack
        device = self.ids[0].device
        all_states = []
        with teacher._ctx(), torch.autocast(device.type, dtype=torch.bfloat16):
            for ids in self.ids:
                pos = torch.arange(len(ids), device=device)[None]
                h = st.embed(ids[None])
                pe = st.rope(h, pos)
                states = {}
                for L in range(1, st.n_layers + 1):
                    h = st.run_block(L, h, pe)
                    states[L] = st.loss_view(L, h)[0].detach().cpu()
                all_states.append(states)
        self.base_states = all_states

    def next(self):
        j = self.i % len(self.ids)
        self.i += 1
        states = self.base_states[j] if self.base_states is not None else None
        return self.ids[j], states


def anchor_trajectory_step(stack, ids, base_states, w, autocast=True):
    """Depth-uniform frozen-base preservation with strictly local backward."""
    if base_states is None:
        raise ValueError("anchor trajectory needs frozen base hidden states")
    device = ids.device
    pos = torch.arange(len(ids), device=device)[None]
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                         enabled=autocast):
        h = stack.embed(ids[None])
        pos_emb = stack.rope(h, pos)
    losses = []
    for L in range(1, stack.n_layers + 1):
        h = h.detach()
        with torch.autocast(h.device.type, dtype=torch.bfloat16, enabled=autocast):
            h = stack.run_block(L, h, pos_emb)
            view = stack.loss_view(L, h)[0]
            loss = hidden_match(view, base_states[L].to(view.device), "nmse")
        (w * loss).backward()
        losses.append(loss.detach().to(device))
    return torch.stack(losses)


def make_anchor(cfg, tok, teacher=None):
    """Build frozen-base targets for local trajectory preservation."""
    if cfg.train.anchor_hidden_weight <= 0:
        return None
    bank = AnchorBank(cfg.train.anchor_path, tok, cfg.model.device)
    if teacher is None:
        raise ValueError("anchor regularization needs an online teacher for "
                         "frozen base targets: enable train.frozen_teacher_copy "
                         "or LoRA + train.online_teacher")
    bank.precompute_base_states(teacher)
    return bank, cfg.train.anchor_hidden_weight
