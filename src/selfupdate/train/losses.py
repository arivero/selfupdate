"""Layerwise forward-distillation losses.

Positions convention (see masking.py): hidden-state losses apply at aligned
positions ``[s0, s0+A)``; local readout auxiliaries apply at ``[s0, s0+A-1)`` predicting
tokens ``[s0+1, s0+A)`` — the caller passes tensors already sliced to the
aligned span, so here row i of the student always corresponds to row i of the
cached teacher.

Three families of hidden-match kinds:

- geometric (``nmse``, ``l2mse``, ``cosine``, ``huber``): compare hidden
  vectors directly, in the residual-stream metric.
- vocabulary-metric (``vocab_mse``, ``lens_kl``, ``lens_js``, ``tuned_lens_kl``): compare hidden vectors as
  the FROZEN vocabulary sees them (docs/hidden_loss.md, Frozen-Vocabulary
  Principle). ``vocab_mse`` is MSE in logit space, computed cheaply through
  the precomputed Gram matrix M = WᵀW of the unembedding — the local
  (Gaussian) approximation of ``lens_kl``, which is full KL between the
  teacher's and student's logit-lens distributions.
- increment (``delta_nmse``, ``delta_cosine``, ``delta_vocab_cos``): compare
  the raw residual update made by an interior block rather than its inherited
  absolute state. Cache boundaries use the paired absolute-state metric.
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

GEOMETRIC_KINDS = ("nmse", "l2mse", "cosine", "huber", "charbonnier",
                   "clipped_nmse", "contrastive", "relational_state", "zero")
VOCAB_KINDS = ("vocab_mse", "lens_kl", "lens_js", "tuned_lens_kl", "vocab_fisher")
# Match what a block *adds* rather than repeatedly charging it for inherited
# state error.  The trainer applies these only to interior raw block outputs
# (2 <= L < n); L=1 and h_n use the paired state fallback below because the
# disk cache has no h0 and h_n is post-final-norm.
DELTA_KINDS = ("delta_nmse", "delta_cosine", "delta_vocab_cos", "flow_nmse",
               "state_delta_nmse", "state_delta_charbonnier")
SPECIAL_KINDS = ("embedding_mse", "mahalanobis", "multi_delta_nmse",
                 "component_nmse")
JACOBIAN_KINDS = ("jacobian_nmse", "jacobian_vocab_mse", "jacobian_cosine",
                  "jacobian_lens_kl")
JACOBIAN_STATE_FALLBACKS = {
    "jacobian_nmse": "nmse",
    "jacobian_vocab_mse": "vocab_mse",
    "jacobian_cosine": "cosine",
    "jacobian_lens_kl": "lens_kl",
}
DELTA_STATE_FALLBACKS = {
    "delta_nmse": "nmse",
    "delta_cosine": "cosine",
    "delta_vocab_cos": "vocab_mse",
    "flow_nmse": "nmse",
    "state_delta_nmse": "nmse",
    "state_delta_charbonnier": "charbonnier",
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
    Increment kinds expose :meth:`delta` for raw interior updates; calling an
    increment kind at a cache boundary selects its documented state fallback.
    """

    def __init__(self, kind: str, final_norm=None, lm_head=None,
                 tuned_lens_path: str = "", jacobian_lens_path: str = "",
                 input_embedding=None, mahalanobis_path: str = "",
                 multi_delta_scales: tuple[int, ...] = (1,)):
        if kind not in GEOMETRIC_KINDS + VOCAB_KINDS + DELTA_KINDS + JACOBIAN_KINDS + SPECIAL_KINDS:
            raise ValueError(f"unknown hidden loss kind {kind!r}")
        if kind in VOCAB_KINDS + ("jacobian_vocab_mse", "jacobian_lens_kl") and (final_norm is None or lm_head is None):
            raise ValueError(f"hidden loss {kind!r} needs final_norm and lm_head")
        if kind == "jacobian_nmse" and lm_head is None:
            raise ValueError("hidden loss 'jacobian_nmse' needs lm_head for width validation")
        if kind == "delta_vocab_cos" and (final_norm is None or lm_head is None):
            raise ValueError("hidden loss 'delta_vocab_cos' needs final_norm and lm_head")
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
        self.multi_delta_scales = tuple(sorted(set(int(k) for k in multi_delta_scales)))
        if not self.multi_delta_scales or self.multi_delta_scales[0] < 1:
            raise ValueError("multi_delta_scales must contain positive offsets")
        self._precision_cpu: dict[int, torch.Tensor] = {}
        self._precision_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        self._gram: torch.Tensor | None = None
        self._centered_gram: torch.Tensor | None = None
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

    def _centered_gram_matrix(self) -> torch.Tensor:
        """``Wᵀ C W`` for vocabulary-mean-centred scores.

        ``C = I - 11ᵀ/V`` removes an otherwise arbitrary vocabulary-wide
        score offset.  We form it from the existing unembedding Gram instead
        of materialising a [positions, vocab] logit tensor.
        """
        if self._centered_gram is None:
            W = self.lm_head.weight.detach()
            with torch.autocast(W.device.type, enabled=False):
                mean = W.float().sum(dim=0)
                self._centered_gram = (
                    self._gram_matrix() - torch.outer(mean, mean) / W.shape[0]
                )
        return self._centered_gram

    @property
    def is_delta(self) -> bool:
        return self.kind in DELTA_KINDS

    @property
    def is_multiscale(self) -> bool:
        return self.kind == "multi_delta_nmse"

    @property
    def state_fallback_kind(self) -> str:
        """Absolute-state metric used where a raw adjacent difference is
        unavailable (the embedding boundary and cached post-norm endpoint)."""
        return DELTA_STATE_FALLBACKS.get(self.kind, self.kind)

    def __call__(self, student_h: torch.Tensor, teacher_h: torch.Tensor,
                 normed: bool = False, layer: int | None = None) -> torch.Tensor:
        # pipeline-parallel guard: targets are produced on the item device
        # (cuda:0) while upper blocks live on cuda:1; .to() is differentiable
        teacher_h = teacher_h.to(student_h.device)
        kind = self.state_fallback_kind
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

    def delta(self, student_h: torch.Tensor, student_prev: torch.Tensor,
              teacher_h: torch.Tensor, teacher_prev: torch.Tensor) -> torch.Tensor:
        """Compare raw successive block contributions.

        The preceding student state is explicitly stop-gradient.  In a
        connected window the normal path through ``student_h`` may still give
        credit to earlier blocks inside that sanctioned window, but this loss
        never creates a second direct gradient through the subtraction.
        Final norm and LM-head bias are intentionally absent: this is a
        residual-update measurement, not a decode of an absolute state.
        """
        if not self.is_delta:
            raise ValueError(f"hidden loss {self.kind!r} has no delta form")
        student_prev = student_prev.detach().to(student_h.device)
        teacher_h = teacher_h.to(student_h.device)
        teacher_prev = teacher_prev.to(student_h.device)
        if self.kind == "delta_nmse":
            return hidden_match(student_h - student_prev,
                                teacher_h - teacher_prev, "nmse")
        if self.kind == "delta_cosine":
            return hidden_match(student_h - student_prev,
                                teacher_h - teacher_prev, "cosine")
        if self.kind == "flow_nmse":
            # Cross-layer token flow: relation between the preceding and new
            # token geometry, not either state in isolation.
            sp = F.normalize(student_prev.float(), dim=-1)
            so = F.normalize(student_h.float(), dim=-1)
            tp = F.normalize(teacher_prev.float(), dim=-1)
            to = F.normalize(teacher_h.float(), dim=-1)
            return F.mse_loss(sp @ so.T, tp @ to.T)
        if self.kind == "state_delta_nmse":
            return 0.5 * (
                hidden_match(student_h, teacher_h, "nmse")
                + hidden_match(student_h - student_prev,
                               teacher_h - teacher_prev, "nmse"))
        if self.kind == "state_delta_charbonnier":
            return 0.5 * (
                hidden_match(student_h, teacher_h, "charbonnier")
                + hidden_match(student_h - student_prev,
                               teacher_h - teacher_prev, "charbonnier"))

        # ``delta_vocab_cos``: cosine of *centred* frozen-vocabulary score
        # changes.  Wᵀ C W avoids a V-wide score tensor while exactly matching
        # ``cos(C W d_s, C W d_t)`` up to fp32 matmul rounding.
        head_dev = self.lm_head.weight.device
        with torch.autocast(head_dev.type, enabled=False):
            ds = (student_h.to(head_dev).float()
                  - student_prev.to(head_dev).float())
            dt = (teacher_h.to(head_dev).float()
                  - teacher_prev.to(head_dev).float())
            M = self._centered_gram_matrix()
            ds_M = ds @ M
            dt_M = dt @ M
            dot = (ds_M * dt).sum(-1)
            ds_norm = (ds_M * ds).sum(-1).clamp_min(0).sqrt()
            dt_norm = (dt_M * dt).sum(-1).clamp_min(0).sqrt()
            return 1.0 - (dot / (ds_norm * dt_norm).clamp_min(1e-8)).mean()

    def multiscale_delta(self, student_h: torch.Tensor, student_history: dict[int, torch.Tensor],
                         teacher_h: torch.Tensor, teacher_history: dict[int, torch.Tensor],
                         layer: int) -> torch.Tensor:
        """Uniform average of available raw k-layer displacements."""
        losses = []
        for k in self.multi_delta_scales:
            anchor = layer - k
            if anchor not in student_history or anchor not in teacher_history:
                continue
            losses.append(hidden_match(student_h - student_history[anchor].detach().to(student_h.device),
                                       teacher_h - teacher_history[anchor].to(student_h.device), "nmse"))
        if not losses:
            return hidden_match(student_h, teacher_h, "nmse")
        return torch.stack(losses).mean()
