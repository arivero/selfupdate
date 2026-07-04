"""Layerwise forward-distillation losses.

Positions convention (see masking.py): hidden-state losses apply at aligned
positions ``[s0, s0+A)``; local readout auxiliaries apply at ``[s0, s0+A-1)`` predicting
tokens ``[s0+1, s0+A)`` — the caller passes tensors already sliced to the
aligned span, so here row i of the student always corresponds to row i of the
cached teacher.

Two families of hidden-match kinds:

- geometric (``nmse``, ``l2mse``, ``cosine``, ``huber``): compare hidden
  vectors directly, in the residual-stream metric.
- vocabulary-metric (``vocab_mse``, ``lens_kl``): compare hidden vectors as
  the FROZEN vocabulary sees them (docs/hidden_loss.md, Frozen-Vocabulary
  Principle). ``vocab_mse`` is MSE in logit space, computed cheaply through
  the precomputed Gram matrix M = WᵀW of the unembedding — the local
  (Gaussian) approximation of ``lens_kl``, which is full KL between the
  teacher's and student's logit-lens distributions.

Whitened/CKA-style losses are deliberately absent: items are batch-1
``[A, H]`` slices with A often below H, so per-batch covariances are
rank-deficient; and CKA's invariance to orthogonal rotations of the student
representation contradicts the frozen-readout contract — the student must
land in the teacher's exact geometry because the next block, final norm and
head never move.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

GEOMETRIC_KINDS = ("nmse", "l2mse", "cosine", "huber")
VOCAB_KINDS = ("vocab_mse", "lens_kl", "vocab_fisher")

FISHER_TOPK = 64


def hidden_match(
    student_h: torch.Tensor,  # [N, H]
    teacher_h: torch.Tensor,  # [N, H]
    kind: str = "nmse",
) -> torch.Tensor:
    """Per-layer hidden-state matching loss (geometric kinds).

    nmse — MSE normalized by the teacher's mean squared norm, so layers of
    different scales weigh comparably. l2mse — MSE between L2-normalized
    vectors (PKD-style stabilizer, direction only). cosine — 1 minus mean
    per-position cosine similarity (direction only, linear in angular error
    near the optimum). huber — smooth-L1 in units of the teacher RMS
    (scale-invariant like nmse, robust to heavy-tailed residual rows).
    """
    teacher_h = teacher_h.float()
    student_h = student_h.float()
    if kind == "nmse":
        return F.mse_loss(student_h, teacher_h) / teacher_h.pow(2).mean().clamp_min(1e-8)
    if kind == "l2mse":
        return F.mse_loss(F.normalize(student_h, dim=-1), F.normalize(teacher_h, dim=-1))
    if kind == "cosine":
        return 1.0 - F.cosine_similarity(student_h, teacher_h, dim=-1, eps=1e-8).mean()
    if kind == "huber":
        scale = teacher_h.pow(2).mean().sqrt().clamp_min(1e-8)
        return F.smooth_l1_loss(student_h / scale, teacher_h / scale, beta=1.0)
    raise ValueError(f"unknown hidden loss kind {kind!r}")


class HiddenLoss:
    """Configured hidden-match loss, callable as ``loss_fn(hs, ht, normed=)``.

    Geometric kinds delegate to :func:`hidden_match`. Vocabulary-metric kinds
    additionally carry references to the frozen final norm and LM head; the
    Gram matrix M = WᵀW ([H, H], fp32) is computed once, lazily, in chunks.
    ``normed=True`` means the caller already applied the final norm (the
    L = n_layers convention, see ``BlockStack.loss_view``); otherwise vocab
    kinds apply it here so every layer is measured in the decode geometry.
    The norm and head are frozen, so gradients through them still reach only
    the student block that produced ``student_h`` — locality is untouched.
    """

    def __init__(self, kind: str, final_norm=None, lm_head=None):
        if kind not in GEOMETRIC_KINDS + VOCAB_KINDS:
            raise ValueError(f"unknown hidden loss kind {kind!r}")
        if kind in VOCAB_KINDS and (final_norm is None or lm_head is None):
            raise ValueError(f"hidden loss {kind!r} needs final_norm and lm_head")
        self.kind = kind
        self.final_norm = final_norm
        self.lm_head = lm_head
        self._gram: torch.Tensor | None = None

    def _gram_matrix(self) -> torch.Tensor:
        """M = WᵀW of the frozen unembedding, fp32, chunked over vocab rows."""
        if self._gram is None:
            W = self.lm_head.weight.detach()
            H = W.shape[1]
            M = torch.zeros(H, H, dtype=torch.float32, device=W.device)
            for i in range(0, W.shape[0], 16384):
                w = W[i: i + 16384].float()
                M += w.T @ w
            self._gram = M
        return self._gram

    def __call__(self, student_h: torch.Tensor, teacher_h: torch.Tensor,
                 normed: bool = False) -> torch.Tensor:
        if self.kind in GEOMETRIC_KINDS:
            return hidden_match(student_h, teacher_h, self.kind)
        if not normed:
            student_h = self.final_norm(student_h)
            teacher_h = self.final_norm(teacher_h.to(student_h.dtype))
        if self.kind == "vocab_mse":
            with torch.autocast(student_h.device.type, enabled=False):
                d = student_h.float() - teacher_h.float()
                M = self._gram_matrix()
                q = (d @ M * d).sum(-1).mean()
                t = teacher_h.float()
                denom = (t @ M * t).sum(-1).mean().clamp_min(1e-8)
                return q / denom
        if self.kind == "vocab_fisher":
            # Position-dependent Fisher metric: weight the logit-space error
            # by the TEACHER's layer-L lens distribution, restricted to its
            # top-k support. This is the Gauss-Newton form of lens_kl —
            # vocab_mse is its p-uniform, full-support degeneration. Capacity
            # concentrates on directions that move the tokens the teacher
            # actually predicts at this position, instead of all 151k rows
            # equally. Wk/p are detached (the head is frozen); gradient
            # reaches only student_h.
            with torch.autocast(student_h.device.type, enabled=False):
                with torch.no_grad():
                    # cache targets arrive fp16; the frozen head may hold
                    # fp32 masters — coerce to the head's dtype for the matmul
                    t_logits = self.lm_head(
                        teacher_h.to(self.lm_head.weight.dtype)).float()
                    p, idx = t_logits.softmax(-1).topk(FISHER_TOPK, dim=-1)
                    p = p / p.sum(-1, keepdim=True)
                    Wk = self.lm_head.weight.detach()[idx].float()  # [N, k, H]
                d = (student_h.float() - teacher_h.float())
                proj = torch.einsum("nkh,nh->nk", Wk, d)
                t = teacher_h.float()
                tproj = torch.einsum("nkh,nh->nk", Wk, t)
                num = (p * proj.pow(2)).sum(-1).mean()
                denom = (p * tproj.pow(2)).sum(-1).mean().clamp_min(1e-8)
                return num / denom
        # lens_kl: KL(teacher || student) over the frozen logit lens. The
        # V-sized matmul may run under the caller's autocast (bf16); the
        # softmax/KL run in fp32.
        s_logits = self.lm_head(student_h)
        with torch.no_grad():
            t_logits = self.lm_head(teacher_h)
        return F.kl_div(
            F.log_softmax(s_logits.float(), dim=-1),
            F.log_softmax(t_logits.float(), dim=-1),
            log_target=True, reduction="batchmean",
        )


def answer_ce(student_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Auxiliary CE on gold answer tokens (already position-shifted by caller)."""
    return F.cross_entropy(student_logits.float(), target_ids)
