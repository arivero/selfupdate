"""Layerwise forward-distillation losses.

Positions convention (see masking.py): hidden-state losses apply at aligned
positions ``[s0, s0+A)``; local readout auxiliaries apply at ``[s0, s0+A-1)`` predicting
tokens ``[s0+1, s0+A)`` — the caller passes tensors already sliced to the
aligned span, so here row i of the student always corresponds to row i of the
cached teacher.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
