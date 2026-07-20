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
    answer_lengths: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor,
           torch.Tensor | None, torch.Tensor | None]:
    """Return token-summed CE, teacher-to-student KL, token count, the
    argmax-acceptance sums (student and teacher) against the answer ids, and
    (when ``answer_lengths`` is given) the student and teacher per-answer
    exact-match sums.

    ``student_hidden[i]`` and ``teacher_hidden[i]`` are final, post-norm
    states at the position predicting ``teacher_answer_ids[i]``.  Chunking
    bounds vocabulary-logit memory; it does not change the token-weighted
    aggregation.  Returned scalars are detached GPU tensors so callers can
    aggregate an epoch without introducing hot-loop host synchronizations.

    The acceptance sums are the vLLM-reproduction metric (owner 2026-07-19):
    ``teacher_answer_ids`` are the vLLM-generated tokens, so
    ``argmax(frozen_head(h)) == id`` counts positions where this trainer's
    OWN forward — cache/store/relay machinery included — greedily reproduces
    the vLLM draft.  Teacher-side acceptance is adapter-free (valid at any
    epoch); student-side equals it at zero-init/lr-0.  Evaluation-only.

    ``answer_lengths`` (owner 2026-07-20), if given, is the per-answer row
    count in this call's row order (one entry per answer, summing to the
    total row count).  The extra return values are then the counts of answers
    where EVERY student-argmax / teacher-argmax token matched its vLLM id —
    the same reproduction checks as the token-level acceptances, at
    per-answer ("exact-seq") granularity.  Both are ``None`` when
    ``answer_lengths`` is omitted.
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

    if answer_lengths is not None and sum(answer_lengths) != int(
            student_hidden.shape[0]):
        raise ValueError("answer_lengths must sum to the row count")

    device = student_hidden.device
    if teacher_answer_ids.device != device:
        teacher_answer_ids = teacher_answer_ids.to(device)
    ce_sum = torch.zeros((), dtype=torch.float32, device=device)
    kl_sum = torch.zeros((), dtype=torch.float32, device=device)
    student_match_sum = torch.zeros((), dtype=torch.float32, device=device)
    teacher_match_sum = torch.zeros((), dtype=torch.float32, device=device)
    student_match_bits = [] if answer_lengths is not None else None
    teacher_match_bits = [] if answer_lengths is not None else None
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
        ids = teacher_answer_ids[first:stop]
        ce_sum.add_(F.nll_loss(student_logp, ids, reduction="sum"))
        # torch KL uses input=student log-probability and target=teacher
        # log-probability, hence this is KL(teacher || student).
        kl_sum.add_(F.kl_div(
            student_logp, teacher_logp, reduction="sum", log_target=True))
        student_bits = student_logp.argmax(-1) == ids
        student_match_sum.add_(student_bits.sum().float())
        teacher_bits = teacher_logp.argmax(-1) == ids
        teacher_match_sum.add_(teacher_bits.sum().float())
        if teacher_match_bits is not None:
            student_match_bits.append(student_bits)
            teacher_match_bits.append(teacher_bits)
        del student_logits, teacher_logits, student_logp, teacher_logp

    if ce_sum.requires_grad or kl_sum.requires_grad:
        raise RuntimeError("evaluation-only output metrics acquired a graph")

    student_exact_match_sum = None
    teacher_exact_match_sum = None
    if answer_lengths is not None:
        student_full_bits = torch.cat(student_match_bits)
        teacher_full_bits = torch.cat(teacher_match_bits)
        student_exact_match_sum = torch.zeros((), dtype=torch.float32,
                                              device=device)
        teacher_exact_match_sum = torch.zeros((), dtype=torch.float32,
                                              device=device)
        pos = 0
        for length in answer_lengths:
            if length > 0:
                student_exact_match_sum += student_full_bits[
                    pos:pos + length].all().float()
                teacher_exact_match_sum += teacher_full_bits[
                    pos:pos + length].all().float()
            pos += length

    return (ce_sum, kl_sum, count, student_match_sum, teacher_match_sum,
            student_exact_match_sum, teacher_exact_match_sum)
