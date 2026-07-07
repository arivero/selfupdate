#!/usr/bin/env python3
"""
teacher_student_lens_diagnostics_v2_corrected.py

Corrected single-file PyTorch / HuggingFace example for teacher-student
comparison with lens-defined vocabulary observables.

Important terminology
---------------------

A hidden state is not logits.

    h_l : [batch, seq, d_model]

A lens/readout maps a hidden state or a hidden-vector contribution into
vocabulary scores.

    scores_l = Lambda_l(h_l) : [batch, seq, vocab]

A pseudo-distribution appears only after softmaxing lens scores.

    p_l = softmax(scores_l / temperature)

Only model(...).logits are real model logits. Everything else in this file is
called "scores" unless it is literally the final output logits.

What this file implements
-------------------------

1. final_output_divergence
   Compares real final model logits.

2. native_logit_lens_scores_for_hidden_index
   Computes native logit-lens vocabulary scores:
       lm_head(final_norm(h_l))
   with care to avoid double-normalizing the final hidden state when HF already
   returns it normalized.

3. tail_logit_lens_divergence_matrix
   Computes KL/JS over native-logit-lens pseudo-distributions only in the last
   N hidden states.

4. successive_hidden_delta_score_distance_matrix
   Computes score-vector distances for successive hidden-state deltas:
       delta_h_i = h_{i+1} - h_i
       delta_scores_i = lm_head.weight @ delta_h_i
   This is not KL. It is a vocabulary-facing score contribution comparison.

5. cumulative_hidden_delta_score_distance_matrix
   Computes score-vector distances for:
       h_i - h_0

6. local_transport_jvp_scores_one_layer
   Correctly separates three prompt-local JVP variants:

   A. transport="pre_norm_hidden"
      scores = W_U * (d h_pre_final[target] / d h_layer[source]) * v
      This is the closest implementation of W_U J_l^x h_l.

   B. transport="post_norm_hidden"
      scores = W_U * (d h_post_final[target] / d h_layer[source]) * v
      This includes the local derivative of the final norm if HF hidden_states[-1]
      is post-final-norm.

   C. transport="final_logits"
      scores = (d final_logits[target] / d h_layer[source]) * v
      This is direct final-logit JVP. It should match B up to output-head details.

The file is intended as a starting point for experiments, not a universal wrapper
for every model architecture. The generic HuggingFace hidden_states convention is
not fine-grained enough to isolate every exact residual write in all models; for
exact component DLA, use model-specific hooks or TransformerLens.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class LensConfig:
    """
    tau:
        Temperature for softmaxing vocabulary scores before KL/JS.

    apply_final_norm_for_intermediate_logit_lens:
        Apply final norm before the lm_head when reading intermediate hidden
        states. Usually appropriate for pre-norm decoder models.

    final_hidden_state_is_already_normalized:
        Many HuggingFace decoder models return hidden_states[-1] after the final
        norm. If true, the native logit lens will not apply final norm again
        when reading hidden_states[-1].

    center_scores_for_vector_distances:
        Subtract vocabulary mean before cosine/L2 score-vector comparisons.

    exclude_final_normalized_state_from_delta_metrics:
        If hidden_states[-1] is post-final-norm, then the last successive delta
        h[-1] - h[-2] mixes the final transformer block and final normalization.
        By default we exclude that last delta from delta/cumulative score metrics.
    """

    tau: float = 2.0
    apply_final_norm_for_intermediate_logit_lens: bool = True
    final_hidden_state_is_already_normalized: bool = True
    center_scores_for_vector_distances: bool = True
    exclude_final_normalized_state_from_delta_metrics: bool = True


# =============================================================================
# Model accessors
# =============================================================================


def get_output_head(model) -> torch.nn.Module:
    """
    Return lm_head / output embedding module.
    """
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            return head

    if hasattr(model, "lm_head"):
        return model.lm_head

    raise ValueError("Could not find output head. Add a model-specific accessor.")


def get_output_weight_and_bias(model) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Return output projection weight and bias if available.

    Weight shape is [vocab, d_model].
    """
    head = get_output_head(model)

    if not hasattr(head, "weight"):
        raise ValueError("Output head has no .weight. Add a model-specific projection.")

    weight = head.weight
    bias = getattr(head, "bias", None)
    return weight, bias


def project_vector_to_vocab_scores(
    model,
    x: Tensor,
    *,
    include_bias: bool = False,
) -> Tensor:
    """
    Project a vector or vector contribution to vocabulary scores.

    For hidden-state reads, including the lm_head bias may be appropriate.
    For vector contributions / JVPs / deltas, bias should NOT be included,
    because a bias is not part of a vector contribution.

    x:
        [B, T, D] or [B, D]

    returns:
        [B, T, V] or [B, V]
    """
    weight, bias = get_output_weight_and_bias(model)
    return F.linear(x, weight, bias if include_bias else None)


def get_final_norm(model) -> Optional[torch.nn.Module]:
    """
    Best-effort lookup for final norm module in common decoder-only HF models.
    Verify this for your architecture.
    """
    candidates = [
        "model.norm",                 # LLaMA, Mistral, Qwen, Gemma-like
        "transformer.ln_f",           # GPT-2
        "gpt_neox.final_layer_norm",  # GPT-NeoX
        "model.final_layernorm",      # some OPT-like variants
        "norm",
    ]

    for path in candidates:
        cur = model
        ok = True
        for name in path.split("."):
            if not hasattr(cur, name):
                ok = False
                break
            cur = getattr(cur, name)
        if ok:
            return cur

    return None


def apply_final_norm(model, h: Tensor) -> Tensor:
    norm = get_final_norm(model)
    if norm is None:
        return h
    return norm(h)


def get_transformer_layers(model) -> Sequence[torch.nn.Module]:
    """
    Best-effort lookup for transformer blocks.
    Required only for prompt-local JVP diagnostics.
    """
    candidates = [
        "model.layers",          # LLaMA, Mistral, Qwen, Gemma-like
        "transformer.h",         # GPT-2
        "gpt_neox.layers",       # GPT-NeoX
        "model.decoder.layers",  # OPT/BART-like
    ]

    for path in candidates:
        cur = model
        ok = True
        for name in path.split("."):
            if not hasattr(cur, name):
                ok = False
                break
            cur = getattr(cur, name)
        if ok:
            return cur

    raise ValueError("Could not find transformer layers. Add your model path.")


def assert_same_vocab_size(teacher, student) -> None:
    v_t = get_output_weight_and_bias(teacher)[0].shape[0]
    v_s = get_output_weight_and_bias(student)[0].shape[0]
    if v_t != v_s:
        raise ValueError(
            f"Teacher and student vocab sizes differ: {v_t} vs {v_s}. "
            "KL/JS over vocabulary is not defined without a token transport map."
        )


# =============================================================================
# Forward helpers
# =============================================================================


def forward_with_hiddens(
    model,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    *,
    no_grad: bool = True,
) -> Tuple[Tensor, Tuple[Tensor, ...]]:
    """
    Return:
        real_logits: [B, T, V]
        hidden_states: tuple of [B, T, D]

    HF convention is model-specific, but usually:
        hidden_states[0] = embedding output
        hidden_states[i+1] = hidden after block i, except that hidden_states[-1]
        may be after final norm rather than raw last-block residual.
    """
    def run():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return out.logits, out.hidden_states

    if no_grad:
        with torch.no_grad():
            return run()
    return run()


def normalize_hidden_index(hidden_index: int, n_hidden: int) -> int:
    if hidden_index < 0:
        hidden_index = n_hidden + hidden_index
    if hidden_index < 0 or hidden_index >= n_hidden:
        raise IndexError(f"hidden_index out of range: {hidden_index}, n_hidden={n_hidden}")
    return hidden_index


def should_apply_final_norm_to_hidden_index(
    hidden_index: int,
    n_hidden: int,
    config: LensConfig,
) -> bool:
    """
    Decide whether native logit lens should apply final norm before lm_head.
    """
    if not config.apply_final_norm_for_intermediate_logit_lens:
        return False

    idx = normalize_hidden_index(hidden_index, n_hidden)

    if config.final_hidden_state_is_already_normalized and idx == n_hidden - 1:
        return False

    return True


# =============================================================================
# Lens scores
# =============================================================================


def native_logit_lens_scores_for_hidden_index(
    model,
    hidden_states: Sequence[Tensor],
    hidden_index: int,
    *,
    config: LensConfig = LensConfig(),
) -> Tensor:
    """
    Native logit-lens scores for hidden_states[hidden_index].

    Returns vocabulary scores, not actual logits except at the true final output
    under the correct model convention.
    """
    n_hidden = len(hidden_states)
    idx = normalize_hidden_index(hidden_index, n_hidden)
    h = hidden_states[idx]

    if should_apply_final_norm_to_hidden_index(idx, n_hidden, config):
        h = apply_final_norm(model, h)

    # For a hidden-state read, include output-head bias if the model has one.
    return project_vector_to_vocab_scores(model, h, include_bias=True)


def native_logit_lens_scores_for_tensor(
    model,
    h: Tensor,
    *,
    apply_norm: bool = True,
    include_bias: bool = True,
) -> Tensor:
    """
    Native logit-lens scores for an arbitrary hidden tensor.

    Use this when you are not indexing into HF hidden_states.
    """
    if apply_norm:
        h = apply_final_norm(model, h)
    return project_vector_to_vocab_scores(model, h, include_bias=include_bias)


# =============================================================================
# Divergences and score distances
# =============================================================================


def _maybe_add_time_dim(scores: Tensor) -> Tensor:
    if scores.ndim == 2:
        return scores.unsqueeze(1)
    return scores


def _maybe_add_mask_time_dim(mask: Optional[Tensor], scores: Tensor) -> Optional[Tensor]:
    if mask is None:
        return None
    if scores.ndim == 2 and mask.ndim == 1:
        return mask.unsqueeze(1)
    return mask


def masked_mean(x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Mean over [B,T] positions, optionally masked.
    """
    if mask is None:
        return x.mean()

    mask = mask.to(device=x.device, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)

    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def kl_from_scores(
    scores_p: Tensor,
    scores_q: Tensor,
    *,
    tau: float = 1.0,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    KL(P || Q), P = softmax(scores_p/tau).

    This is meaningful only after scores_p/scores_q have been defined by a lens
    or are actual model logits.
    """
    original_ndim = scores_p.ndim
    scores_p = _maybe_add_time_dim(scores_p)
    scores_q = _maybe_add_time_dim(scores_q)
    mask = _maybe_add_mask_time_dim(mask, scores_p if original_ndim == 2 else scores_p)

    log_p = F.log_softmax(scores_p / tau, dim=-1)
    log_q = F.log_softmax(scores_q / tau, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return masked_mean(kl, mask)


def sym_kl_from_scores(
    scores_p: Tensor,
    scores_q: Tensor,
    *,
    tau: float = 1.0,
    mask: Optional[Tensor] = None,
) -> Tensor:
    return 0.5 * kl_from_scores(scores_p, scores_q, tau=tau, mask=mask) + 0.5 * kl_from_scores(
        scores_q, scores_p, tau=tau, mask=mask
    )


def js_from_scores(
    scores_p: Tensor,
    scores_q: Tensor,
    *,
    tau: float = 1.0,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Jensen-Shannon divergence between pseudo-distributions.
    """
    original_ndim = scores_p.ndim
    scores_p = _maybe_add_time_dim(scores_p)
    scores_q = _maybe_add_time_dim(scores_q)
    mask = _maybe_add_mask_time_dim(mask, scores_p if original_ndim == 2 else scores_p)

    log_p = F.log_softmax(scores_p / tau, dim=-1)
    log_q = F.log_softmax(scores_q / tau, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(1e-30))

    js = 0.5 * (p * (log_p - log_m)).sum(dim=-1) + 0.5 * (q * (log_q - log_m)).sum(dim=-1)
    return masked_mean(js, mask)


def divergence_from_scores(
    scores_p: Tensor,
    scores_q: Tensor,
    *,
    divergence: str = "js",
    tau: float = 1.0,
    mask: Optional[Tensor] = None,
) -> Tensor:
    if divergence == "kl":
        return kl_from_scores(scores_p, scores_q, tau=tau, mask=mask)
    if divergence == "reverse_kl":
        return kl_from_scores(scores_q, scores_p, tau=tau, mask=mask)
    if divergence == "sym_kl":
        return sym_kl_from_scores(scores_p, scores_q, tau=tau, mask=mask)
    if divergence == "js":
        return js_from_scores(scores_p, scores_q, tau=tau, mask=mask)
    raise ValueError("divergence must be one of: kl, reverse_kl, sym_kl, js")


def center_vocab_scores(scores: Tensor) -> Tensor:
    return scores - scores.mean(dim=-1, keepdim=True)


def cosine_distance_between_score_vectors(
    scores_a: Tensor,
    scores_b: Tensor,
    *,
    mask: Optional[Tensor] = None,
    center: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """
    Mean 1-cosine between vocabulary score vectors.
    """
    if center:
        scores_a = center_vocab_scores(scores_a)
        scores_b = center_vocab_scores(scores_b)

    scores_a = F.normalize(scores_a, dim=-1, eps=eps)
    scores_b = F.normalize(scores_b, dim=-1, eps=eps)

    dist = 1.0 - (scores_a * scores_b).sum(dim=-1)
    return masked_mean(dist, mask)


def l2_distance_between_score_vectors(
    scores_a: Tensor,
    scores_b: Tensor,
    *,
    mask: Optional[Tensor] = None,
    center: bool = True,
) -> Tensor:
    if center:
        scores_a = center_vocab_scores(scores_a)
        scores_b = center_vocab_scores(scores_b)

    dist = (scores_a - scores_b).pow(2).mean(dim=-1)
    return masked_mean(dist, mask)


def score_vector_distance(
    scores_a: Tensor,
    scores_b: Tensor,
    *,
    metric: str = "cosine",
    mask: Optional[Tensor] = None,
    center: bool = True,
) -> Tensor:
    if metric == "cosine":
        return cosine_distance_between_score_vectors(scores_a, scores_b, mask=mask, center=center)
    if metric == "l2":
        return l2_distance_between_score_vectors(scores_a, scores_b, mask=mask, center=center)
    raise ValueError("metric must be 'cosine' or 'l2'")


# =============================================================================
# Final output divergence: real logits
# =============================================================================


def final_output_divergence(
    teacher,
    student,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    *,
    config: LensConfig = LensConfig(),
    divergence: str = "kl",
    no_grad: bool = True,
) -> Tensor:
    """
    Compare real final model logits.

    This is not a lens; it is the ordinary teacher-student output comparison.
    """
    assert_same_vocab_size(teacher, student)

    teacher.eval()
    if no_grad:
        student.eval()

    teacher_logits, _ = forward_with_hiddens(teacher, input_ids, attention_mask, no_grad=True)
    student_logits, _ = forward_with_hiddens(student, input_ids, attention_mask, no_grad=no_grad)

    return divergence_from_scores(
        teacher_logits,
        student_logits,
        divergence=divergence,
        tau=config.tau,
        mask=attention_mask,
    )


# =============================================================================
# Native logit-lens divergence
# =============================================================================


def logit_lens_hidden_divergence(
    teacher,
    student,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    *,
    teacher_hidden_index: int,
    student_hidden_index: int,
    config: LensConfig = LensConfig(),
    divergence: str = "js",
    no_grad: bool = True,
) -> Tensor:
    """
    Compare two hidden states through their native logit lenses.

    The resulting KL/JS is lens-induced; it is not an intrinsic layer metric.
    """
    assert_same_vocab_size(teacher, student)

    teacher.eval()
    if no_grad:
        student.eval()

    _, t_h = forward_with_hiddens(teacher, input_ids, attention_mask, no_grad=True)
    _, s_h = forward_with_hiddens(student, input_ids, attention_mask, no_grad=no_grad)

    t_scores = native_logit_lens_scores_for_hidden_index(
        teacher, t_h, teacher_hidden_index, config=config
    )
    s_scores = native_logit_lens_scores_for_hidden_index(
        student, s_h, student_hidden_index, config=config
    )

    return divergence_from_scores(
        t_scores,
        s_scores,
        divergence=divergence,
        tau=config.tau,
        mask=attention_mask,
    )


def logit_lens_divergence_matrix(
    teacher,
    student,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    *,
    config: LensConfig = LensConfig(),
    divergence: str = "js",
    include_embedding_state: bool = False,
) -> Tensor:
    """
    Full teacher-hidden-index x student-hidden-index divergence matrix.

    Warning:
        Early/middle raw logit-lens entries may mostly measure lens failure.
    """
    assert_same_vocab_size(teacher, student)

    teacher.eval()
    student.eval()

    _, t_h = forward_with_hiddens(teacher, input_ids, attention_mask, no_grad=True)
    _, s_h = forward_with_hiddens(student, input_ids, attention_mask, no_grad=True)

    t_indices = list(range(0 if include_embedding_state else 1, len(t_h)))
    s_indices = list(range(0 if include_embedding_state else 1, len(s_h)))

    t_scores_list = [
        native_logit_lens_scores_for_hidden_index(teacher, t_h, i, config=config)
        for i in t_indices
    ]
    s_scores_list = [
        native_logit_lens_scores_for_hidden_index(student, s_h, j, config=config)
        for j in s_indices
    ]

    mat = torch.empty(len(t_scores_list), len(s_scores_list), device=input_ids.device)

    for i, t_scores in enumerate(t_scores_list):
        for j, s_scores in enumerate(s_scores_list):
            mat[i, j] = divergence_from_scores(
                t_scores,
                s_scores,
                divergence=divergence,
                tau=config.tau,
                mask=attention_mask,
            )

    return mat


def tail_logit_lens_divergence_matrix(
    teacher,
    student,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    *,
    teacher_tail: int = 8,
    student_tail: int = 8,
    config: LensConfig = LensConfig(),
    divergence: str = "js",
) -> Tensor:
    """
    Tail-only native-logit-lens divergence matrix.
    """
    assert_same_vocab_size(teacher, student)

    teacher.eval()
    student.eval()

    _, t_h = forward_with_hiddens(teacher, input_ids, attention_mask, no_grad=True)
    _, s_h = forward_with_hiddens(student, input_ids, attention_mask, no_grad=True)

    t_indices = list(range(1, len(t_h)))[-teacher_tail:]
    s_indices = list(range(1, len(s_h)))[-student_tail:]

    t_scores_list = [
        native_logit_lens_scores_for_hidden_index(teacher, t_h, i, config=config)
        for i in t_indices
    ]
    s_scores_list = [
        native_logit_lens_scores_for_hidden_index(student, s_h, j, config=config)
        for j in s_indices
    ]

    mat = torch.empty(len(t_scores_list), len(s_scores_list), device=input_ids.device)

    for i, t_scores in enumerate(t_scores_list):
        for j, s_scores in enumerate(s_scores_list):
            mat[i, j] = divergence_from_scores(
                t_scores,
                s_scores,
                divergence=divergence,
                tau=config.tau,
                mask=attention_mask,
            )

    return mat


# =============================================================================
# Successive hidden deltas and cumulative hidden deltas
# =============================================================================


def hidden_indices_for_delta_metrics(
    hidden_states: Sequence[Tensor],
    *,
    config: LensConfig = LensConfig(),
) -> List[int]:
    """
    Hidden-state indices included in cumulative metrics.

    If final hidden state is already normalized and config excludes it, omit the
    final hidden state to avoid mixing final normalization into vector-delta
    diagnostics.
    """
    last = len(hidden_states) - 1
    end_exclusive = last if (
        config.final_hidden_state_is_already_normalized
        and config.exclude_final_normalized_state_from_delta_metrics
    ) else last + 1

    # exclude embedding state 0
    return list(range(1, end_exclusive))


def successive_hidden_deltas(
    hidden_states: Sequence[Tensor],
    *,
    config: LensConfig = LensConfig(),
) -> List[Tensor]:
    """
    Compute successive hidden-state deltas h[i] - h[i-1].

    These are exact deltas between HuggingFace-exposed hidden states, not
    necessarily exact transformer-block writes for every architecture.
    """
    indices = hidden_indices_for_delta_metrics(hidden_states, config=config)
    return [hidden_states[i] - hidden_states[i - 1] for i in indices]


def cumulative_hidden_deltas(
    hidden_states: Sequence[Tensor],
    *,
    config: LensConfig = LensConfig(),
) -> List[Tensor]:
    """
    Compute h[i] - h[0] for included hidden indices.
    """
    indices = hidden_indices_for_delta_metrics(hidden_states, config=config)
    h0 = hidden_states[0]
    return [hidden_states[i] - h0 for i in indices]


def successive_hidden_delta_score_distance_matrix(
    teacher,
    student,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    *,
    config: LensConfig = LensConfig(),
    metric: str = "cosine",
) -> Tensor:
    """
    Compare vocabulary-facing score contributions of successive hidden deltas.

    This is not KL. It compares score vectors:
        W_U^T delta_h_T  vs  W_U^S delta_h_S
    """
    assert_same_vocab_size(teacher, student)

    teacher.eval()
    student.eval()

    _, t_h = forward_with_hiddens(teacher, input_ids, attention_mask, no_grad=True)
    _, s_h = forward_with_hiddens(student, input_ids, attention_mask, no_grad=True)

    t_deltas = successive_hidden_deltas(t_h, config=config)
    s_deltas = successive_hidden_deltas(s_h, config=config)

    # Vector contributions: no lm_head bias.
    t_scores_list = [project_vector_to_vocab_scores(teacher, d, include_bias=False) for d in t_deltas]
    s_scores_list = [project_vector_to_vocab_scores(student, d, include_bias=False) for d in s_deltas]

    mat = torch.empty(len(t_scores_list), len(s_scores_list), device=input_ids.device)

    for i, t_scores in enumerate(t_scores_list):
        for j, s_scores in enumerate(s_scores_list):
            mat[i, j] = score_vector_distance(
                t_scores,
                s_scores,
                metric=metric,
                mask=attention_mask,
                center=config.center_scores_for_vector_distances,
            )

    return mat


def cumulative_hidden_delta_score_distance_matrix(
    teacher,
    student,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    *,
    config: LensConfig = LensConfig(),
    metric: str = "cosine",
) -> Tensor:
    """
    Compare vocabulary-facing score contributions of cumulative hidden deltas.

    This is not KL.
    """
    assert_same_vocab_size(teacher, student)

    teacher.eval()
    student.eval()

    _, t_h = forward_with_hiddens(teacher, input_ids, attention_mask, no_grad=True)
    _, s_h = forward_with_hiddens(student, input_ids, attention_mask, no_grad=True)

    t_cums = cumulative_hidden_deltas(t_h, config=config)
    s_cums = cumulative_hidden_deltas(s_h, config=config)

    t_scores_list = [project_vector_to_vocab_scores(teacher, c, include_bias=False) for c in t_cums]
    s_scores_list = [project_vector_to_vocab_scores(student, c, include_bias=False) for c in s_cums]

    mat = torch.empty(len(t_scores_list), len(s_scores_list), device=input_ids.device)

    for i, t_scores in enumerate(t_scores_list):
        for j, s_scores in enumerate(s_scores_list):
            mat[i, j] = score_vector_distance(
                t_scores,
                s_scores,
                metric=metric,
                mask=attention_mask,
                center=config.center_scores_for_vector_distances,
            )

    return mat


# =============================================================================
# Layer matching
# =============================================================================


def student_to_teacher_depth_pairs(n_teacher_layers: int, n_student_layers: int) -> List[Tuple[int, int]]:
    """
    For each student block index j, choose teacher block index i by relative depth.
    Returns zero-based block index pairs (teacher_i, student_j).
    """
    pairs = []
    for j in range(n_student_layers):
        if n_student_layers <= 1:
            i = n_teacher_layers - 1
        else:
            i = round(j * (n_teacher_layers - 1) / (n_student_layers - 1))
        pairs.append((i, j))
    return pairs


def monotone_alignment_path(distance_matrix: Tensor) -> List[Tuple[int, int]]:
    """
    Dynamic-time-warping-style monotone path through a distance matrix.
    """
    D = distance_matrix
    n, m = D.shape
    cost = torch.empty_like(D)
    back = torch.empty(n, m, 2, dtype=torch.long, device=D.device)

    cost[0, 0] = D[0, 0]
    back[0, 0] = torch.tensor([-1, -1], device=D.device)

    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + D[i, 0]
        back[i, 0] = torch.tensor([i - 1, 0], device=D.device)

    for j in range(1, m):
        cost[0, j] = cost[0, j - 1] + D[0, j]
        back[0, j] = torch.tensor([0, j - 1], device=D.device)

    for i in range(1, n):
        for j in range(1, m):
            candidates = torch.stack([cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]])
            k = int(torch.argmin(candidates).item())

            if k == 0:
                prev = (i - 1, j)
            elif k == 1:
                prev = (i, j - 1)
            else:
                prev = (i - 1, j - 1)

            cost[i, j] = D[i, j] + candidates[k]
            back[i, j] = torch.tensor(prev, device=D.device)

    path = []
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        path.append((i, j))
        pi, pj = back[i, j].tolist()
        if pi < 0:
            break
        i, j = pi, pj

    path.reverse()
    return path


# =============================================================================
# Differentiable training-loss examples
# =============================================================================


def tail_logit_lens_distillation_loss(
    teacher,
    student,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    *,
    teacher_hidden_indices: Sequence[int],
    student_hidden_indices: Sequence[int],
    config: LensConfig = LensConfig(),
    divergence: str = "js",
) -> Tensor:
    """
    Differentiable student loss using native-logit-lens pseudo-distributions.

    Use only for hidden states where the native logit lens is empirically
    meaningful, typically the final/tail states.
    """
    assert_same_vocab_size(teacher, student)

    if len(teacher_hidden_indices) != len(student_hidden_indices):
        raise ValueError("teacher_hidden_indices and student_hidden_indices must match length.")

    teacher.eval()
    student.train()

    with torch.no_grad():
        _, t_h = forward_with_hiddens(teacher, input_ids, attention_mask, no_grad=True)

    _, s_h = forward_with_hiddens(student, input_ids, attention_mask, no_grad=False)

    losses = []
    for ti, si in zip(teacher_hidden_indices, student_hidden_indices):
        with torch.no_grad():
            t_scores = native_logit_lens_scores_for_hidden_index(teacher, t_h, ti, config=config)

        s_scores = native_logit_lens_scores_for_hidden_index(student, s_h, si, config=config)

        losses.append(
            divergence_from_scores(
                t_scores,
                s_scores,
                divergence=divergence,
                tau=config.tau,
                mask=attention_mask,
            )
        )

    return torch.stack(losses).mean()


def successive_delta_score_matching_loss(
    teacher,
    student,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    *,
    teacher_delta_indices: Sequence[int],
    student_delta_indices: Sequence[int],
    config: LensConfig = LensConfig(),
    metric: str = "cosine",
) -> Tensor:
    """
    Differentiable score-vector matching loss for successive hidden deltas.

    This is not a probability/KL loss.
    """
    assert_same_vocab_size(teacher, student)

    if len(teacher_delta_indices) != len(student_delta_indices):
        raise ValueError("teacher_delta_indices and student_delta_indices must match length.")

    teacher.eval()
    student.train()

    with torch.no_grad():
        _, t_h = forward_with_hiddens(teacher, input_ids, attention_mask, no_grad=True)
        t_deltas = successive_hidden_deltas(t_h, config=config)
        t_scores_list = [
            project_vector_to_vocab_scores(teacher, d, include_bias=False) for d in t_deltas
        ]

    _, s_h = forward_with_hiddens(student, input_ids, attention_mask, no_grad=False)
    s_deltas = successive_hidden_deltas(s_h, config=config)
    s_scores_list = [project_vector_to_vocab_scores(student, d, include_bias=False) for d in s_deltas]

    losses = []
    for ti, si in zip(teacher_delta_indices, student_delta_indices):
        losses.append(
            score_vector_distance(
                t_scores_list[ti].detach(),
                s_scores_list[si],
                metric=metric,
                mask=attention_mask,
                center=config.center_scores_for_vector_distances,
            )
        )

    return torch.stack(losses).mean()


def combined_distillation_loss(
    teacher,
    student,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    *,
    config: LensConfig = LensConfig(),
    final_weight: float = 1.0,
    tail_lens_weight: float = 0.1,
    delta_score_weight: float = 0.1,
    teacher_tail_hidden_indices: Sequence[int] = (),
    student_tail_hidden_indices: Sequence[int] = (),
    teacher_delta_indices: Sequence[int] = (),
    student_delta_indices: Sequence[int] = (),
) -> Tensor:
    """
    Example combined objective.

    final-output KL:
        real model logits

    tail lens JS:
        native-logit-lens pseudo-distributions, only for trusted tail states

    delta score matching:
        vocabulary-facing score-vector matching, not KL
    """
    terms = []

    if final_weight:
        terms.append(
            final_weight
            * final_output_divergence(
                teacher,
                student,
                input_ids,
                attention_mask,
                config=config,
                divergence="kl",
                no_grad=False,
            )
        )

    if tail_lens_weight and teacher_tail_hidden_indices and student_tail_hidden_indices:
        terms.append(
            tail_lens_weight
            * tail_logit_lens_distillation_loss(
                teacher,
                student,
                input_ids,
                attention_mask,
                teacher_hidden_indices=teacher_tail_hidden_indices,
                student_hidden_indices=student_tail_hidden_indices,
                config=config,
                divergence="js",
            )
        )

    if delta_score_weight and teacher_delta_indices and student_delta_indices:
        terms.append(
            delta_score_weight
            * successive_delta_score_matching_loss(
                teacher,
                student,
                input_ids,
                attention_mask,
                teacher_delta_indices=teacher_delta_indices,
                student_delta_indices=student_delta_indices,
                config=config,
                metric="cosine",
            )
        )

    if not terms:
        raise ValueError("No loss terms enabled.")

    return torch.stack(terms).sum()


# =============================================================================
# Prompt-local JVP lenses
# =============================================================================


class HiddenStateReplacer:
    """
    Forward hook that replaces a transformer block output with a supplied tensor.

    This handles the common tensor and tuple-output cases.
    """

    def __init__(self, module: torch.nn.Module, replacement: Tensor):
        self.module = module
        self.replacement = replacement
        self.handle = None

    def hook(self, module, inputs, output):
        if torch.is_tensor(output):
            return self.replacement

        if isinstance(output, tuple):
            return (self.replacement,) + output[1:]

        raise TypeError(f"Unsupported block output type: {type(output)!r}")

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self.hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.remove()


class FinalNormInputCapture:
    """
    Captures the input to the final norm module.

    This is used to obtain h_pre_final in models where hidden_states[-1] is
    post-final-norm.
    """

    def __init__(self, norm_module: Optional[torch.nn.Module]):
        self.norm_module = norm_module
        self.handle = None
        self.value = None

    def hook(self, module, inputs):
        self.value = inputs[0]

    def __enter__(self):
        if self.norm_module is not None:
            self.handle = self.norm_module.register_forward_pre_hook(self.hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.remove()


def replace_single_position(
    full_hidden: Tensor,
    replacement_vec: Tensor,
    *,
    source_pos: int,
) -> Tensor:
    """
    Differentiably replace full_hidden[:, source_pos, :] by replacement_vec.

    full_hidden:
        [B, T, D], usually detached constant

    replacement_vec:
        [B, D], differentiable JVP input
    """
    B, T, D = full_hidden.shape
    pos = source_pos % T

    pieces = []
    if pos > 0:
        pieces.append(full_hidden[:, :pos, :])
    pieces.append(replacement_vec.unsqueeze(1))
    if pos + 1 < T:
        pieces.append(full_hidden[:, pos + 1 :, :])

    return torch.cat(pieces, dim=1)


def local_transport_jvp_scores_one_layer(
    model,
    input_ids: Tensor,
    attention_mask: Optional[Tensor],
    *,
    layer_index: int,
    source_pos: int = -1,
    target_pos: int = -1,
    transport: str = "pre_norm_hidden",
    direction: str = "activation",
) -> Tensor:
    """
    Prompt-local JVP lens scores for one layer and one source/target position.

    layer_index:
        zero-based transformer block index.

    source_pos:
        position in the selected layer output whose hidden vector is perturbed.

    target_pos:
        final position whose hidden/logit output is read.

    transport:

        "pre_norm_hidden":
            Compute
                W_U * J_pre * v
            where
                J_pre = d h_pre_final[target_pos] / d h_layer[source_pos].
            This is the closest implemented object to W_U J_l^x h_l.
            Output-head bias is NOT included.

        "post_norm_hidden":
            Compute
                W_U * J_post * v
            where J_post maps to post-final-norm hidden state.
            This includes the local derivative of final norm.
            Output-head bias is NOT included.

        "final_logits":
            Compute
                d final_logits[target_pos] / d h_layer[source_pos] * v.
            This is direct final-logit JVP, not hidden transport.

    direction:

        "activation":
            v = h_layer[source_pos]

        "unit":
            v = normalized h_layer[source_pos]

    returns:
        scores: [B, V]
    """
    model.eval()
    layers = get_transformer_layers(model)

    with torch.no_grad():
        _, hidden_states = forward_with_hiddens(model, input_ids, attention_mask, no_grad=True)
        full_h_l = hidden_states[layer_index + 1].detach()  # output after block layer_index
        source_vec = full_h_l[:, source_pos, :].detach()

    if direction == "activation":
        v = source_vec
    elif direction == "unit":
        v = F.normalize(source_vec, dim=-1)
    else:
        raise ValueError("direction must be 'activation' or 'unit'.")

    final_norm = get_final_norm(model)

    def f_pre_norm_hidden(replacement_vec: Tensor) -> Tensor:
        replacement_full = replace_single_position(
            full_h_l,
            replacement_vec,
            source_pos=source_pos,
        )

        with HiddenStateReplacer(layers[layer_index], replacement_full):
            with FinalNormInputCapture(final_norm) as capture:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

        if capture.value is not None:
            h_pre_final = capture.value
        else:
            # No final norm found; fall back to final hidden.
            h_pre_final = out.hidden_states[-1]

        return h_pre_final[:, target_pos, :]

    def f_post_norm_hidden(replacement_vec: Tensor) -> Tensor:
        replacement_full = replace_single_position(
            full_h_l,
            replacement_vec,
            source_pos=source_pos,
        )

        with HiddenStateReplacer(layers[layer_index], replacement_full):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        return out.hidden_states[-1][:, target_pos, :]

    def f_final_logits(replacement_vec: Tensor) -> Tensor:
        replacement_full = replace_single_position(
            full_h_l,
            replacement_vec,
            source_pos=source_pos,
        )

        with HiddenStateReplacer(layers[layer_index], replacement_full):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )

        return out.logits[:, target_pos, :]

    if transport == "pre_norm_hidden":
        _, jvp_hidden = torch.autograd.functional.jvp(
            f_pre_norm_hidden,
            inputs=source_vec,
            v=v,
            create_graph=False,
            strict=False,
        )
        return project_vector_to_vocab_scores(model, jvp_hidden, include_bias=False)

    if transport == "post_norm_hidden":
        _, jvp_hidden = torch.autograd.functional.jvp(
            f_post_norm_hidden,
            inputs=source_vec,
            v=v,
            create_graph=False,
            strict=False,
        )
        return project_vector_to_vocab_scores(model, jvp_hidden, include_bias=False)

    if transport == "final_logits":
        _, jvp_scores = torch.autograd.functional.jvp(
            f_final_logits,
            inputs=source_vec,
            v=v,
            create_graph=False,
            strict=False,
        )
        return jvp_scores

    raise ValueError("transport must be pre_norm_hidden, post_norm_hidden, or final_logits")


def local_transport_jvp_layer_divergence(
    teacher,
    student,
    input_ids: Tensor,
    attention_mask: Optional[Tensor],
    *,
    teacher_layer_index: int,
    student_layer_index: int,
    source_pos: int = -1,
    target_pos: int = -1,
    transport: str = "pre_norm_hidden",
    direction: str = "activation",
    config: LensConfig = LensConfig(),
    divergence: str = "js",
) -> Tensor:
    """
    Compare teacher/student through prompt-local JVP lens scores.

    Expensive. Diagnostic, not default training loss.
    """
    assert_same_vocab_size(teacher, student)

    teacher_scores = local_transport_jvp_scores_one_layer(
        teacher,
        input_ids,
        attention_mask,
        layer_index=teacher_layer_index,
        source_pos=source_pos,
        target_pos=target_pos,
        transport=transport,
        direction=direction,
    )

    student_scores = local_transport_jvp_scores_one_layer(
        student,
        input_ids,
        attention_mask,
        layer_index=student_layer_index,
        source_pos=source_pos,
        target_pos=target_pos,
        transport=transport,
        direction=direction,
    )

    return divergence_from_scores(
        teacher_scores,
        student_scores,
        divergence=divergence,
        tau=config.tau,
        mask=None,
    )


# =============================================================================
# Combined diagnostic
# =============================================================================


def teacher_student_diagnostic(
    teacher,
    student,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    *,
    config: LensConfig = LensConfig(),
    tail: int = 8,
) -> Dict[str, Tensor]:
    """
    Basic diagnostic bundle.

    Returns:
        final_kl:
            KL over real final logits.

        tail_logit_lens_js:
            JS matrix over native-logit-lens pseudo-distributions in the tail.

        successive_delta_score_cosine:
            score-vector distance matrix for successive hidden-state deltas.

        cumulative_delta_score_cosine:
            score-vector distance matrix for h_i - h_0.
    """
    return {
        "final_kl": final_output_divergence(
            teacher,
            student,
            input_ids,
            attention_mask,
            config=config,
            divergence="kl",
            no_grad=True,
        ),
        "tail_logit_lens_js": tail_logit_lens_divergence_matrix(
            teacher,
            student,
            input_ids,
            attention_mask,
            teacher_tail=tail,
            student_tail=tail,
            config=config,
            divergence="js",
        ),
        "successive_delta_score_cosine": successive_hidden_delta_score_distance_matrix(
            teacher,
            student,
            input_ids,
            attention_mask,
            config=config,
            metric="cosine",
        ),
        "cumulative_delta_score_cosine": cumulative_hidden_delta_score_distance_matrix(
            teacher,
            student,
            input_ids,
            attention_mask,
            config=config,
            metric="cosine",
        ),
    }


# =============================================================================
# CLI demo
# =============================================================================


def load_models_and_tokenizer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_name = args.tokenizer or args.teacher
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=dtype,
        device_map=args.device_map if args.device_map != "none" else None,
    )
    student = AutoModelForCausalLM.from_pretrained(
        args.student,
        torch_dtype=dtype,
        device_map=args.device_map if args.device_map != "none" else None,
    )

    if args.device_map == "none":
        teacher.to(args.device)
        student.to(args.device)

    return teacher, student, tokenizer


def run_cli_demo(args) -> None:
    teacher, student, tokenizer = load_models_and_tokenizer(args)

    batch = tokenizer(
        [args.prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )

    if args.device_map == "none":
        batch = {k: v.to(args.device) for k, v in batch.items()}

    config = LensConfig(
        tau=args.tau,
        apply_final_norm_for_intermediate_logit_lens=not args.no_final_norm,
        final_hidden_state_is_already_normalized=not args.final_hidden_not_normalized,
        center_scores_for_vector_distances=True,
        exclude_final_normalized_state_from_delta_metrics=True,
    )

    metrics = teacher_student_diagnostic(
        teacher,
        student,
        batch["input_ids"],
        batch.get("attention_mask"),
        config=config,
        tail=args.tail,
    )

    print("\n=== Teacher-student lens diagnostics v2 corrected ===")
    print(f"teacher: {args.teacher}")
    print(f"student: {args.student}")
    print(f"prompt:  {args.prompt!r}")
    print(f"tau:     {args.tau}")
    print()

    for key, value in metrics.items():
        value = value.detach().float().cpu()
        print(f"{key}: shape={tuple(value.shape)}")
        print(value)
        print()

    if args.run_local_jvp:
        d = local_transport_jvp_layer_divergence(
            teacher,
            student,
            batch["input_ids"],
            batch.get("attention_mask"),
            teacher_layer_index=args.local_teacher_layer,
            student_layer_index=args.local_student_layer,
            source_pos=-1,
            target_pos=-1,
            transport=args.local_transport,
            config=config,
            divergence="js",
        )
        print(f"local_transport_jvp_layer_divergence({args.local_transport}):")
        print(d.detach().float().cpu())


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Corrected teacher-student lens diagnostics."
    )
    p.add_argument("--teacher", type=str, default="gpt2-medium")
    p.add_argument("--student", type=str, default="gpt2")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--prompt", type=str, default="The capital of France is")
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--tail", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--device-map", type=str, default="none", help="'none' or e.g. 'auto'")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--no-final-norm", action="store_true")
    p.add_argument("--final-hidden-not-normalized", action="store_true")
    p.add_argument("--run-local-jvp", action="store_true")
    p.add_argument("--local-teacher-layer", type=int, default=0)
    p.add_argument("--local-student-layer", type=int, default=0)
    p.add_argument(
        "--local-transport",
        type=str,
        default="pre_norm_hidden",
        choices=["pre_norm_hidden", "post_norm_hidden", "final_logits"],
    )
    return p


# =============================================================================
# Documentation and references
# =============================================================================


DOCUMENTATION_AND_REFERENCES = r"""
===============================================================================
DOCUMENTATION
===============================================================================

1. Hidden states, scores, pseudo-distributions, logits
------------------------------------------------------

This file enforces the following terminology.

Hidden state:
    h_l in R^d

Real model logits:
    final_logits = model(input).logits

Lens-induced vocabulary scores:
    scores_l = Lambda_l(h_l)

Lens-induced pseudo-distribution:
    p_l = softmax(scores_l / tau)

Intermediate hidden states do not have logits. Any intermediate KL/JS is a
property of the hidden state plus the chosen lens.


2. Native logit lens
--------------------

For a hidden state h_l:

    scores_l = lm_head(final_norm(h_l))

These are called scores, not actual logits. The final hidden state may already be
normalized in HuggingFace hidden_states[-1], so the code avoids double norming if
LensConfig.final_hidden_state_is_already_normalized is true.

Use tail_logit_lens_divergence_matrix when the native logit lens is readable only
near the end of the network.


3. Vector contribution scores
-----------------------------

For a vector contribution v, such as a hidden-state delta or JVP vector:

    scores_v = W_U v

The lm_head bias is intentionally excluded. A bias is part of a state readout,
not part of a vector contribution. These scores should normally be compared with
cosine or centered L2, not KL.


4. Successive hidden deltas
---------------------------

The generic HuggingFace hidden_states tuple is not always a perfect list of
residual-stream block outputs. In many decoder models hidden_states[-1] is after
the final norm. Therefore the final successive delta may mix the last transformer
block and final normalization. The default config excludes that final normalized
state from delta metrics.

For exact component-level DLA, use architecture-specific hooks or TransformerLens.


5. Prompt-local JVP variants
----------------------------

local_transport_jvp_scores_one_layer separates three objects:

A. pre_norm_hidden

    W_U * (d h_pre_final[target] / d h_layer[source]) * v

This is closest to the proposed corpus-free local hidden-transport lens:

    W_U J_l^x h_l

B. post_norm_hidden

    W_U * (d h_post_final[target] / d h_layer[source]) * v

This includes the local derivative of the final norm.

C. final_logits

    (d final_logits[target] / d h_layer[source]) * v

This is direct final-logit JVP. It is not the same object as A.

All three are prompt-local and expensive. They are better as diagnostics than as
main training losses.


6. Tokenizer requirement
------------------------

KL/JS over vocabulary assumes teacher and student share the same vocabulary and
token ordering. If they do not, you need a token/string transport map.


===============================================================================
REFERENCES
===============================================================================

Logit lens:
    nostalgebraist, "interpreting GPT: the logit lens", 2020.
    https://www.alignmentforum.org/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

Tuned lens:
    Belrose et al., "Eliciting Latent Predictions from Transformers with the
    Tuned Lens", arXiv:2303.08112, 2023.
    https://arxiv.org/abs/2303.08112

DistillLens:
    Dhakal et al., "DistillLens: Symmetric Knowledge Distillation Through Logit
    Lens", arXiv:2602.13567, 2026.
    https://arxiv.org/abs/2602.13567

Anthropic Jacobian lens / global workspace:
    "Verbalizable Representations Form a Global Workspace in Language Models",
    Transformer Circuits Thread, 2026.
    https://transformer-circuits.pub/2026/workspace/index.html

Direct Logit Attribution caveat:
    Janiak et al., "An Adversarial Example for Direct Logit Attribution:
    Memory Management in GELU-4L", arXiv:2310.07325, 2023.
    https://arxiv.org/abs/2310.07325

IG-Lens / probability attribution:
    Nguyen, "IG-Lens: Exact Additive Probability Attribution Across Transformer
    Layers via Telescoping Integrated Gradients", arXiv:2606.29693, 2026.
    https://arxiv.org/abs/2606.29693

TransformerLens:
    https://github.com/TransformerLensOrg/TransformerLens
"""


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    run_cli_demo(args)
