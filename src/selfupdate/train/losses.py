"""Layerwise forward-distillation losses.

Positions convention (see masking.py): hidden-state losses apply at aligned
positions ``[s0, s0+A)``; local readout auxiliaries apply at ``[s0, s0+A-1)`` predicting
tokens ``[s0+1, s0+A)`` — the caller passes tensors already sliced to the
aligned span, so here row i of the student always corresponds to row i of the
cached teacher.

Four families of hidden-match kinds:

- geometric (``nmse``, ``l2mse``, ``cosine``, ``huber``): compare hidden
  vectors directly, in the residual-stream metric.
- local-increment (``delta_cosine``): compare the current student block's
  update ``student_h - teacher_input`` with the same teacher block's update
  ``teacher_h - teacher_input``.  The input is the detached teacher anchor,
  never a previous student block's output.
- vocabulary-metric (``vocab_mse``, ``vocab_cosine_sampled``, ``lens_kl``, ``lens_js``, ``tuned_lens_kl``): compare hidden vectors as
  the FROZEN vocabulary sees them (docs/hidden_loss.md, Frozen-Vocabulary
  Principle). ``vocab_mse`` is MSE in logit space, computed cheaply through
  the precomputed Gram matrix M = WᵀW of the unembedding — the local
  (Gaussian) approximation of ``lens_kl``, which is full KL between the
  teacher's and student's logit-lens distributions.
- Jacobian-pullback (``jacobian_nmse``, ``jacobian_vocab_mse``,
  ``jacobian_lens_kl``): first transport each layer through a frozen,
  corpus-fitted downstream Jacobian. ``jacobian_nmse`` is the pure induced
  ``JᵀJ`` metric; the other two then use the named frozen-head metric.

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

from ..eval.teacher_output import EVALUATION_ONLY_OUTPUT_NAMES

GEOMETRIC_KINDS = ("nmse", "l2mse", "cosine", "huber", "charbonnier",
                   "clipped_nmse", "contrastive", "relational_state", "zero")
LOCAL_INCREMENT_KINDS = ("delta_cosine",)
VOCAB_KINDS = ("vocab_mse", "vocab_cosine_sampled", "lens_kl", "lens_js",
               "tuned_lens_kl", "vocab_fisher")
SPECIAL_KINDS = ("embedding_mse", "mahalanobis")
JACOBIAN_KINDS = ("jacobian_nmse", "jacobian_vocab_mse", "jacobian_cosine",
                  "jacobian_lens_kl")
JACOBIAN_STATE_FALLBACKS = {
    "jacobian_nmse": "nmse",
    "jacobian_vocab_mse": "vocab_mse",
    "jacobian_cosine": "cosine",
    "jacobian_lens_kl": "lens_kl",
}

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
    if kind == "zero":
        # ablation-only: no hidden matching at all — the "auxiliary is 100%"
        # control for the naming contract (keeps the graph valid and cheap)
        return student_h.sum() * 0.0
    if kind == "nmse":
        return F.mse_loss(student_h, teacher_h) / teacher_h.pow(2).mean().clamp_min(1e-8)
    if kind == "l2mse":
        return F.mse_loss(F.normalize(student_h, dim=-1), F.normalize(teacher_h, dim=-1))
    if kind == "cosine":
        return 1.0 - F.cosine_similarity(student_h, teacher_h, dim=-1, eps=1e-8).mean()
    if kind == "huber":
        scale = teacher_h.pow(2).mean().sqrt().clamp_min(1e-8)
        return F.smooth_l1_loss(student_h / scale, teacher_h / scale, beta=1.0)
    if kind == "charbonnier":
        scale = teacher_h.pow(2).mean().sqrt().clamp_min(1e-8)
        return (torch.sqrt(((student_h - teacher_h) / scale).pow(2) + 1e-6) - 1e-3).mean()
    if kind == "clipped_nmse":
        d2 = (student_h - teacher_h).pow(2).mean(-1)
        cap = teacher_h.pow(2).mean(-1).median().detach().clamp_min(1e-8) * 4
        return d2.clamp_max(cap).mean() / teacher_h.pow(2).mean().clamp_min(1e-8)
    if kind == "contrastive":
        # In-sequence teacher rows are negatives; this is deliberately
        # batch-local and therefore has no stale queue or cross-item state.
        s = F.normalize(student_h, dim=-1)
        with torch.no_grad():
            t = F.normalize(teacher_h, dim=-1)
        if s.shape[0] < 2:
            return (student_h - teacher_h).pow(2).mean() * 0.0
        return F.cross_entropy((s @ t.T) / 0.1,
                               torch.arange(s.shape[0], device=s.device))
    if kind == "relational_state":
        # Rotation-invariant token geometry is deliberately paired with an
        # absolute coordinate term: the next frozen block/head requires the
        # teacher basis, so a Gram-only objective is not a legal standalone
        # loss on this branch.
        absolute = hidden_match(student_h, teacher_h, "nmse")
        s = F.normalize(student_h, dim=-1)
        t = F.normalize(teacher_h, dim=-1)
        relational = F.mse_loss(s @ s.T, t @ t.T)
        return 0.5 * (absolute + relational)
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
    ``delta_cosine`` additionally receives the aligned, detached teacher block
    input.  At the final block it uses absolute cosine because the stored
    target and ``loss_view`` are post-final-norm while the anchor is pre-norm;
    subtracting them would not define a same-coordinate block update.
    """

    @classmethod
    def from_config(cls, train_cfg, stack) -> "HiddenLoss":
        """The one construction path the schedule loops share: every knob
        that selects or parameterizes the hidden objective comes from
        ``cfg.train``; the frozen norm/head/embedding come from the stack."""
        return cls(train_cfg.hidden_loss, stack.final_norm, stack.lm_head,
                   tuned_lens_path=train_cfg.tuned_lens_path,
                   jacobian_lens_path=train_cfg.jacobian_lens_path,
                   input_embedding=stack.embed_tokens,
                   mahalanobis_path=train_cfg.mahalanobis_path,
                   vocab_cosine_samples=train_cfg.vocab_cosine_samples,
                   vocab_cosine_seed=train_cfg.vocab_cosine_seed)

    def __init__(self, kind: str, final_norm=None, lm_head=None,
                 tuned_lens_path: str = "", jacobian_lens_path: str = "",
                 input_embedding=None, mahalanobis_path: str = "",
                 vocab_cosine_samples: int = 0,
                 vocab_cosine_seed: int = 17):
        if kind in EVALUATION_ONLY_OUTPUT_NAMES:
            raise ValueError(
                f"{kind} is an evaluation-only full-training-set metric and "
                "is NEVER a training objective")
        if kind not in (GEOMETRIC_KINDS + LOCAL_INCREMENT_KINDS + VOCAB_KINDS
                        + JACOBIAN_KINDS + SPECIAL_KINDS):
            raise ValueError(f"unknown hidden loss kind {kind!r}")
        if kind in VOCAB_KINDS + ("jacobian_vocab_mse", "jacobian_lens_kl") and (final_norm is None or lm_head is None):
            raise ValueError(f"hidden loss {kind!r} needs final_norm and lm_head")
        if kind == "jacobian_nmse" and lm_head is None:
            raise ValueError("hidden loss 'jacobian_nmse' needs lm_head for width validation")
        if kind == "tuned_lens_kl" and not tuned_lens_path:
            raise ValueError("hidden loss 'tuned_lens_kl' needs train.tuned_lens_path")
        if kind in JACOBIAN_KINDS and not jacobian_lens_path:
            raise ValueError(f"hidden loss {kind!r} needs train.jacobian_lens_path")
        if kind == "embedding_mse" and input_embedding is None:
            raise ValueError("hidden loss 'embedding_mse' needs the frozen input embedding")
        if kind == "mahalanobis" and not mahalanobis_path:
            raise ValueError("hidden loss 'mahalanobis' needs train.mahalanobis_path")
        self.kind = kind
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.input_embedding = input_embedding
        self._precision_cpu: dict[int, torch.Tensor] = {}
        self._precision_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        self._gram: torch.Tensor | None = None
        self.vocab_cosine_samples = int(vocab_cosine_samples)
        self.vocab_cosine_seed = int(vocab_cosine_seed)
        self._sampled_vocab: torch.Tensor | None = None
        if kind == "vocab_cosine_sampled" and self.vocab_cosine_samples <= 1:
            raise ValueError(
                "vocab_cosine_sampled needs train.vocab_cosine_samples > 1")
        self.translators = None
        self._jacobians_cpu: dict[int, torch.Tensor] = {}
        self._jacobian_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
        self._jacobian_trace_cache: dict[tuple[int, str, torch.device], torch.Tensor] = {}
        self.jacobian_metadata: dict[str, object] = {}
        if kind == "tuned_lens_kl":
            from .tuned_lens import load_translators

            self.translators = load_translators(tuned_lens_path, device=lm_head.weight.device)
            self.translators.eval()
            self.translators.requires_grad_(False)
        if kind in JACOBIAN_KINDS:
            artifact = torch.load(jacobian_lens_path, map_location="cpu", weights_only=True)
            required = {"J", "source_layers", "d_model", "n_prompts"}
            if not isinstance(artifact, dict) or not required.issubset(artifact):
                raise ValueError(
                    f"invalid Jacobian lens {jacobian_lens_path!r}: needs keys {sorted(required)}")
            width = int(artifact["d_model"])
            if width != int(lm_head.weight.shape[1]):
                raise ValueError(
                    f"Jacobian lens width {width} != model hidden width {lm_head.weight.shape[1]}")
            matrices = artifact["J"]
            source_layers = [int(x) for x in artifact["source_layers"]]
            if not isinstance(matrices, dict) or set(map(int, matrices)) != set(source_layers):
                raise ValueError("Jacobian lens J/source_layers coverage is inconsistent")
            for source in source_layers:
                J = matrices[source].detach().contiguous()
                if J.shape != (width, width) or not torch.isfinite(J).all():
                    raise ValueError(f"invalid Jacobian matrix for source layer {source}")
                self._jacobians_cpu[source] = J
            self.jacobian_metadata = {
                "path": jacobian_lens_path,
                "d_model": width,
                "n_prompts": int(artifact["n_prompts"]),
                "source_layers": source_layers,
            }
        if kind == "mahalanobis":
            artifact = torch.load(mahalanobis_path, map_location="cpu", weights_only=True)
            matrices = artifact.get("precision") if isinstance(artifact, dict) else None
            if not isinstance(matrices, dict):
                raise ValueError("Mahalanobis artifact needs a per-layer 'precision' mapping")
            width = lm_head.weight.shape[1]
            for layer, P in matrices.items():
                P = P.detach().float().contiguous()
                if P.shape != (width, width) or not torch.isfinite(P).all():
                    raise ValueError(f"invalid Mahalanobis precision for layer {layer}")
                self._precision_cpu[int(layer)] = P

    def _jacobian(self, source: int, like: torch.Tensor) -> torch.Tensor | None:
        """Lazily retain each frozen transport on the layer's device."""
        J = self._jacobians_cpu.get(source)
        if J is None:
            return None
        key = (source, like.device, like.dtype)
        if key not in self._jacobian_cache:
            self._jacobian_cache[key] = J.to(device=like.device, dtype=like.dtype)
        return self._jacobian_cache[key]

    def _precision(self, layer: int, like: torch.Tensor) -> torch.Tensor:
        P = self._precision_cpu.get(layer)
        if P is None:
            raise ValueError(f"Mahalanobis artifact has no precision for layer {layer}")
        key = (layer, like.device)
        if key not in self._precision_cache:
            self._precision_cache[key] = P.to(like.device)
        return self._precision_cache[key]

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

    def _sampled_vocab_matrix(self) -> torch.Tensor:
        """Deterministic centred rows of the frozen unembedding.

        Cosine over these sampled score coordinates is a Johnson-Lindenstrauss-
        style coordinate sketch of centred full-vocabulary score cosine. It
        costs O(N H S), not O(N H V), and the frozen sampled matrix remains a
        measurement device rather than a trainable readout.
        """
        if self._sampled_vocab is None:
            W = self.lm_head.weight.detach()
            count = min(self.vocab_cosine_samples, W.shape[0])
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.vocab_cosine_seed)
            indices = torch.randperm(
                W.shape[0], generator=generator, device="cpu")[:count]
            with torch.autocast(W.device.type, enabled=False):
                mean = W.float().mean(dim=0, keepdim=True)
                self._sampled_vocab = (
                    W.index_select(0, indices.to(W.device)).float() - mean
                ).contiguous()
        return self._sampled_vocab

    def __call__(self, student_h: torch.Tensor, teacher_h: torch.Tensor,
                 normed: bool = False, layer: int | None = None,
                 aligned_input: torch.Tensor | None = None) -> torch.Tensor:
        # Teacher targets may be streamed from CPU or a stage-local store.
        teacher_h = teacher_h.to(student_h.device)
        kind = self.kind
        if kind == "delta_cosine":
            if normed:
                # Final-layer targets/views are post-final-norm, whereas x is
                # pre-norm. Absolute cosine is the explicit, typed fallback.
                return hidden_match(student_h, teacher_h.detach(), "cosine")
            if aligned_input is None:
                raise ValueError(
                    "delta_cosine needs the aligned detached teacher block input")
            anchor = aligned_input.detach().to(student_h.device).float()
            student_delta = student_h.float() - anchor
            with torch.no_grad():
                teacher_delta = teacher_h.float() - anchor
            return 1.0 - F.cosine_similarity(
                student_delta, teacher_delta, dim=-1, eps=1e-8).mean()
        if kind in JACOBIAN_KINDS:
            if layer is None:
                raise ValueError(f"{kind} needs a 1-based layer index")
            source = layer - 1
            J = self._jacobian(source, student_h)
            kind = JACOBIAN_STATE_FALLBACKS[kind]
            if J is not None:
                if self.kind in ("jacobian_nmse", "jacobian_vocab_mse"):
                    return self._jacobian_mse(
                        student_h, teacher_h, J, source,
                        vocab=self.kind == "jacobian_vocab_mse")
                student_h = student_h @ J.T
                with torch.no_grad():
                    teacher_h = teacher_h.to(J.dtype) @ J.T
                normed = False
        if kind in GEOMETRIC_KINDS:
            return hidden_match(student_h, teacher_h, kind)
        if kind == "embedding_mse":
            E = self.input_embedding.weight.detach().float()
            d = student_h.float() - teacher_h.float()
            num = (d @ (E.T @ E) * d).sum(-1).mean()
            den = (teacher_h.float() @ (E.T @ E) * teacher_h.float()).sum(-1).mean().clamp_min(1e-8)
            return num / den
        if kind == "mahalanobis":
            if layer is None:
                raise ValueError("mahalanobis needs a layer index")
            d = student_h.float() - teacher_h.float()
            P = self._precision(layer, d)
            return (d @ P * d).sum(-1).mean() / (teacher_h.float() @ P * teacher_h.float()).sum(-1).mean().clamp_min(1e-8)
        if kind == "tuned_lens_kl":
            if layer is None:
                raise ValueError("tuned_lens_kl needs a layer index")
            lens_dev = next(self.translators.parameters()).device
            student_h = student_h.to(lens_dev)
            teacher_h = teacher_h.to(lens_dev)
            if not normed:
                from .tuned_lens import apply_translator

                student_h = apply_translator(self.translators, layer, student_h)
                teacher_h = apply_translator(self.translators, layer, teacher_h)
        if not normed:
            student_h = self.final_norm(student_h)
            teacher_h = self.final_norm(teacher_h.to(student_h.dtype))
        # vocab kinds decode through the frozen head — compute on its card
        head_dev = self.lm_head.weight.device
        student_h = student_h.to(head_dev)
        teacher_h = teacher_h.to(head_dev)
        if kind == "vocab_mse":
            with torch.autocast(student_h.device.type, enabled=False):
                d = student_h.float() - teacher_h.float()
                M = self._gram_matrix()
                q = (d @ M * d).sum(-1).mean()
                t = teacher_h.float()
                denom = (t @ M * t).sum(-1).mean().clamp_min(1e-8)
                return q / denom
        if kind == "vocab_cosine_sampled":
            with torch.autocast(student_h.device.type, enabled=False):
                sampled = self._sampled_vocab_matrix()
                student_scores = student_h.float() @ sampled.T
                with torch.no_grad():
                    teacher_scores = teacher_h.float() @ sampled.T
                return 1.0 - F.cosine_similarity(
                    student_scores, teacher_scores, dim=-1,
                    eps=1e-8).mean()
        if kind == "vocab_fisher":
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
        # lens_kl / lens_js / tuned_lens_kl: vocabulary divergences over the
        # frozen logit lens.  JS is the bounded symmetric control: the teacher
        # distribution is detached, while gradients flow only through the
        # student distribution and their midpoint.
        # logit lens. tuned_lens_kl first maps each layer through its frozen
        # per-layer affine translator so middle layers are compared in a
        # calibrated decode geometry.
        # V-sized matmul may run under the caller's autocast (bf16); the
        # softmax/KL run in fp32.
        s_logits = self.lm_head(student_h)
        with torch.no_grad():
            t_logits = self.lm_head(teacher_h)
        if kind == "lens_js":
            s_logp = F.log_softmax(s_logits.float(), dim=-1)
            t_logp = F.log_softmax(t_logits.float(), dim=-1)
            s_p = s_logp.exp()
            with torch.no_grad():
                t_p = t_logp.exp()
            log_m = (0.5 * (s_p + t_p)).log()
            teacher_to_mid = F.kl_div(log_m, t_logp, log_target=True,
                                      reduction="batchmean")
            student_to_mid = (s_p * (s_logp - log_m)).sum(-1).mean()
            return 0.5 * (teacher_to_mid + student_to_mid)
        return F.kl_div(
            F.log_softmax(s_logits.float(), dim=-1),
            F.log_softmax(t_logits.float(), dim=-1),
            log_target=True, reduction="batchmean",
        )

    def _jacobian_mse(self, student_h, teacher_h, J, source: int, *, vocab: bool):
        """Scale-calibrated quadratic pullback on ``J (h_s - h_t)``.

        A Jacobian transports a perturbation, not an absolute state.  Applying
        final_norm to ``J h`` invents a missing affine intercept and made the
        sole non-Jacobian final layer dominate the old MSE arms.  The trace
        denominator is the expected transported energy for an isotropic error
        with the teacher state's per-coordinate variance; it removes arbitrary
        layer-to-layer operator scale without erasing J's directional metric.
        """
        with torch.autocast(student_h.device.type, enabled=False):
            d = student_h.float() - teacher_h.to(student_h.device).float()
            Jf = J.float()
            z = d @ Jf.T
            teacher_energy = teacher_h.float().pow(2).mean().to(z.device).clamp_min(1e-8)
            if vocab:
                M = self._gram_matrix().to(z.device)
                numerator = (z @ M * z).sum(-1).mean()
                key = (source, "vocab", z.device)
                if key not in self._jacobian_trace_cache:
                    self._jacobian_trace_cache[key] = (M @ Jf * Jf).sum().detach()
            else:
                numerator = z.pow(2).sum(-1).mean()
                key = (source, "plain", z.device)
                if key not in self._jacobian_trace_cache:
                    self._jacobian_trace_cache[key] = Jf.pow(2).sum().detach()
            return numerator / (teacher_energy * self._jacobian_trace_cache[key].clamp_min(1e-8))
