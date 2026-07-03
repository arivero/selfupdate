"""Distillation losses.

Positions convention (see masking.py): hidden-state losses apply at aligned
positions ``[s0, s0+A)``; logit losses apply at ``[s0, s0+A-1)`` predicting
tokens ``[s0+1, s0+A)`` — the caller passes tensors already sliced to the
aligned span, so here row i of the student always corresponds to row i of the
cached teacher.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) for x < 0, numerically stable."""
    x = x.clamp(max=-1e-7)
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def kd_topk_kl(
    student_logits: torch.Tensor,  # [N, V]
    topk_v: torch.Tensor,  # [N, k] teacher top-k logit values
    topk_i: torch.Tensor,  # [N, k] vocab indices
    logz: torch.Tensor,  # [N] logsumexp of the full teacher row
    T: float = 1.0,
) -> torch.Tensor:
    """KL(teacher || student) over the teacher's top-k plus one lumped tail
    bucket. Exact at T=1 (logz covers the full vocab); at T != 1 the tail is
    approximated as a single atom with pseudo-logit log(sum_tail exp(v)).

    The result is scaled by T^2 (Hinton et al.): softened-KL gradients shrink
    as ~1/T^2, so without the rescale a temperature sweep would also sweep the
    effective learning rate. At T=1 the scale is a no-op.
    """
    topk_v = topk_v.float()
    logz = logz.float()
    # tail pseudo-logit: log(exp(logz) - sum_topk exp(v))
    lse_k = torch.logsumexp(topk_v, dim=-1)
    tail_logit = logz + _log1mexp(lse_k - logz)

    t_aug = torch.cat([topk_v, tail_logit.unsqueeze(-1)], dim=-1) / T
    logp = F.log_softmax(t_aug, dim=-1)  # [N, k+1]

    ls_full = F.log_softmax(student_logits.float() / T, dim=-1)
    ls_k = torch.gather(ls_full, -1, topk_i.long())
    s_tail = _log1mexp(torch.logsumexp(ls_k, dim=-1))
    logq = torch.cat([ls_k, s_tail.unsqueeze(-1)], dim=-1)

    return (T * T) * (logp.exp() * (logp - logq)).sum(-1).mean()


def hidden_match(
    student_h: torch.Tensor,  # [N, H]
    teacher_h: torch.Tensor,  # [N, H]
    kind: str = "nmse",
) -> torch.Tensor:
    """Per-layer hidden-state matching loss.

    nmse — MSE normalized by the teacher's mean squared norm, so layers of
    different scales weigh comparably. l2mse — MSE between L2-normalized
    vectors (PKD-style stabilizer, direction only).
    """
    teacher_h = teacher_h.float()
    student_h = student_h.float()
    if kind == "nmse":
        return F.mse_loss(student_h, teacher_h) / teacher_h.pow(2).mean().clamp_min(1e-8)
    if kind == "l2mse":
        return F.mse_loss(F.normalize(student_h, dim=-1), F.normalize(teacher_h, dim=-1))
    raise ValueError(f"unknown hidden loss kind {kind!r}")


def answer_ce(student_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Auxiliary CE on gold answer tokens (already position-shifted by caller)."""
    return F.cross_entropy(student_logits.float(), target_ids)


def lens_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) over full-vocab rows [N, V] — the per-block
    lens-KL candidate: match the TEACHER's layer-L lens distribution instead
    of the gold one-hot (softer than lens-CE; PKD reinterpreted through the
    readout, so hidden error is weighted by what the head cares about)."""
    t = F.log_softmax(teacher_logits.float(), dim=-1)
    s = F.log_softmax(student_logits.float(), dim=-1)
    return (t.exp() * (t - s)).sum(-1).mean()
