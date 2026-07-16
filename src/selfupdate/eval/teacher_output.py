"""Evaluation-only distances between teacher and student output distributions.

These metrics are deliberately outside ``train.losses``.  They are measured
over every eligible answer token in a complete training-set traversal, but
they are NEVER optimizer objectives: inputs must be detached, the vocabulary
head must be frozen, and the function runs under ``torch.no_grad``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


CE_EVAL_NAME = "CE-eval-loss"
KL_EVAL_NAME = "KL-eval-loss"
EVALUATION_ONLY_OUTPUT_NAMES = frozenset({
    CE_EVAL_NAME,
    KL_EVAL_NAME,
    "ce_eval_loss",
    "kl_eval_loss",
})


@torch.no_grad()
def teacher_output_eval_sums(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_answer_ids: torch.Tensor,
    lm_head,
    *,
    chunk_rows: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return token-summed CE and teacher-to-student KL plus token count.

    ``student_hidden[i]`` and ``teacher_hidden[i]`` are final, post-norm
    states at the position predicting ``teacher_answer_ids[i]``.  Chunking
    bounds vocabulary-logit memory; it does not change the token-weighted
    aggregation.  Returned scalars are detached GPU tensors so callers can
    aggregate an epoch without introducing hot-loop host synchronizations.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if (student_hidden.ndim != 2
            or teacher_hidden.shape != student_hidden.shape):
        raise ValueError("student/teacher final hidden states must match [N,H]")
    if (teacher_answer_ids.ndim != 1
            or teacher_answer_ids.shape[0] != student_hidden.shape[0]):
        raise ValueError("teacher answer ids must have one target per hidden row")
    if student_hidden.requires_grad or teacher_hidden.requires_grad:
        raise RuntimeError(
            "CE-eval-loss/KL-eval-loss require detached states and are NEVER "
            "training losses")
    if any(parameter.requires_grad for parameter in lm_head.parameters()):
        raise RuntimeError(
            "CE-eval-loss/KL-eval-loss require a frozen vocabulary head")

    device = student_hidden.device
    if teacher_answer_ids.device != device:
        teacher_answer_ids = teacher_answer_ids.to(device)
    ce_sum = torch.zeros((), dtype=torch.float32, device=device)
    kl_sum = torch.zeros((), dtype=torch.float32, device=device)
    count = int(student_hidden.shape[0])
    for first in range(0, count, chunk_rows):
        stop = min(first + chunk_rows, count)
        with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=(device.type == "cuda")):
            student_logits = lm_head(student_hidden[first:stop])
            teacher_logits = lm_head(teacher_hidden[first:stop])
        student_logp = F.log_softmax(student_logits.float(), dim=-1)
        teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
        ce_sum.add_(F.nll_loss(
            student_logp, teacher_answer_ids[first:stop], reduction="sum"))
        # torch KL uses input=student log-probability and target=teacher
        # log-probability, hence this is KL(teacher || student).
        kl_sum.add_(F.kl_div(
            student_logp, teacher_logp, reduction="sum", log_target=True))
        del student_logits, teacher_logits, student_logp, teacher_logp

    if ce_sum.requires_grad or kl_sum.requires_grad:
        raise RuntimeError("evaluation-only output metrics acquired a graph")
    return ce_sum, kl_sum, count
