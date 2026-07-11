"""Anti-intrusion anchor: frozen-base preservation on neighbor-genre text.

Counters the readout trigger ("poetic Spanish -> recite the poem") exactly
where catastrophic remembering showed it is installed. Both terms are
teacher-sourced (training-target law): the KL anchor matches the frozen
base's logits inside the top readout window; the trajectory anchor matches
the frozen base's per-layer states depth-uniformly with strictly local
backward. Reference-token CE on anchor text is forbidden.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

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
        self.base_logits: list[torch.Tensor] | None = None
        self.base_states: list[dict[int, torch.Tensor]] | None = None
        self.i = 0

    @torch.no_grad()
    def precompute_base_logits(self, teacher: OnlineTeacherSource):
        """Base-model logits per fragment (anchor-KL targets), computed once
        through the frozen teacher (adapters-off or frozen copy)."""
        st = teacher.stack
        device = self.ids[0].device
        outs, all_states = [], []
        with teacher._ctx(), torch.autocast(device.type, dtype=torch.bfloat16):
            for ids in self.ids:
                pos = torch.arange(len(ids), device=device)[None]
                h = st.embed(ids[None])
                pe = st.rope(h, pos)
                states = {}
                for L in range(1, st.n_layers + 1):
                    h = st.run_block(L, h, pe)
                    states[L] = st.loss_view(L, h)[0].detach().cpu()
                outs.append(st.lm_head(st.final_norm(h))[0].detach())
                all_states.append(states)
        self.base_logits = outs
        self.base_states = all_states

    def next(self):
        j = self.i % len(self.ids)
        self.i += 1
        base = self.base_logits[j] if self.base_logits is not None else None
        states = self.base_states[j] if self.base_states is not None else None
        return self.ids[j], base, states


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


def anchor_step(stack, L0, ids, w, base_logits=None, autocast=True):
    """Anti-intrusion anchor on a neighbor-genre fragment, gradient
    confined to the top readout window [L0..n] (input detached below the window,
    frozen norm/head): counters the readout trigger ("poetic Spanish ->
    recite the poem") exactly where catastrophic remembering showed it is
    installed. Returns the unweighted loss value.

    ``base_logits`` is required: KL(base || student) per position means
    "on neighbor-genre input, behave like the teacher/base model"."""
    if base_logits is None:
        raise ValueError("anchor_step requires teacher/base logits; reference-token CE is forbidden")
    device = ids.device
    pos = torch.arange(len(ids), device=device)[None]
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                         enabled=autocast):
        h = stack.embed(ids[None])
        pos_emb = stack.rope(h, pos)
        for L in range(1, L0):
            h = stack.run_block(L, h, pos_emb)
    h = h.detach()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
        for L in range(L0, stack.n_layers + 1):
            h = stack.run_block(L, h, pos_emb)
        full_logits = stack.lm_head(stack.final_norm(h))[0]
        loss = F.kl_div(
            F.log_softmax(full_logits.float(), dim=-1),
            F.log_softmax(base_logits.float(), dim=-1),
            log_target=True, reduction="batchmean",
        )
    (w * loss).backward()
    return loss.detach()


def make_anchor(cfg, tok, teacher=None):
    """Build frozen-base targets for output and/or trajectory preservation."""
    if cfg.train.anchor_kl_weight <= 0 and cfg.train.anchor_hidden_weight <= 0:
        return None
    if cfg.train.anchor_kl_weight > 0 and cfg.train.readout_window_blocks <= 0:
        raise ValueError("anchor weights need readout_window_blocks > 0 "
                         "(the anchor regularizes the top readout window)")
    bank = AnchorBank(cfg.train.anchor_path, tok, cfg.model.device)
    if teacher is None:
        raise ValueError("anchor regularization needs an online teacher for "
                         "frozen base targets: enable train.frozen_teacher_copy "
                         "or LoRA + train.online_teacher")
    bank.precompute_base_logits(teacher)
    return bank, max(cfg.train.anchor_kl_weight, cfg.train.anchor_hidden_weight)
