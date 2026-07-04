"""Tuned lens: per-layer affine translators onto the frozen unembedding.

Raw logit-lens readouts of early layers are brittle (representation drift
across depth makes the final norm+head the wrong decoder for layer L).
The tuned lens (Belrose et al.) fixes this with one affine probe per
layer trained to match the FINAL distribution: minimize
KL(p_final || p_lens(L)) on neutral text.

Design constraints inherited from the program:
- The vocabulary stack is FROZEN (embed / final norm / head). Translators
  are separate parameters; the model gets no gradient — hidden states are
  detached and only translator params sit in the optimizer.
- Delta parameterization: lens(h) = h + A h + b with A, b zero-initialized,
  so the untrained tuned lens IS the raw logit lens (identity), and
  training only learns the correction. This also makes the
  identity-equivalence test exact rather than approximate.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def make_translators(hidden_size: int, n_layers: int, device="cuda") -> torch.nn.ModuleDict:
    """Zero-init affine deltas for layers 1..n_layers-1 (the final layer is
    decoded by the real norm+head and needs no translator)."""
    d = torch.nn.ModuleDict()
    for L in range(1, n_layers):
        lin = torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float32)
        torch.nn.init.zeros_(lin.weight)
        torch.nn.init.zeros_(lin.bias)
        d[str(L)] = lin
    return d.to(device)


def apply_translator(translators, L: int, h: torch.Tensor) -> torch.Tensor:
    """h + A_L h + b_L in fp32, returned in h's dtype. Missing layer (final)
    passes through."""
    key = str(L)
    if translators is None or key not in translators:
        return h
    hf = h.float()
    return (hf + translators[key](hf)).to(h.dtype)


def save_translators(translators, path, meta: dict | None = None):
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for key, lin in translators.items():
        tensors[f"layer{key}.weight"] = lin.weight.detach().cpu().contiguous()
        tensors[f"layer{key}.bias"] = lin.bias.detach().cpu().contiguous()
    save_file(tensors, str(path))
    if meta is not None:
        (path.parent / "meta.json").write_text(json.dumps(meta, indent=1))


def load_translators(path, device="cuda") -> torch.nn.ModuleDict:
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    layers = sorted({int(k[len("layer"):k.index(".")]) for k in tensors})
    d = torch.nn.ModuleDict()
    for L in layers:
        w = tensors[f"layer{L}.weight"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0], dtype=torch.float32)
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(tensors[f"layer{L}.bias"])
        d[str(L)] = lin
    return d.to(device)


def lens_kl_step(model, translators, ids: torch.Tensor,
                 chunk: int = 256) -> dict[int, float]:
    """One training step's forward+backward: mean-over-positions
    KL(final || lens_L) per layer, position-chunked so full-vocab logits
    never exceed [chunk, V]. Model is used under no_grad; the graph starts
    at the detached hidden states, so only translators receive gradient.
    Returns per-layer KL (detached floats). Caller owns optimizer step."""
    inner = model.model
    n_layers = model.config.num_hidden_layers
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
    hs = [h[0].detach() for h in out.hidden_states]  # [T, d] per depth
    T = hs[0].shape[0]
    per_layer = {L: 0.0 for L in range(1, n_layers)}
    n_chunks = 0
    for c0 in range(0, T, chunk):
        sl = slice(c0, min(c0 + chunk, T))
        with torch.no_grad():
            # hidden_states[-1] is already post-final-norm
            t_logp = F.log_softmax(model.lm_head(hs[n_layers][sl]).float(), -1)
            t_p = t_logp.exp()
        losses = []
        for L in range(1, n_layers):
            h = apply_translator(translators, L, hs[L][sl])
            l_logp = F.log_softmax(model.lm_head(inner.norm(h)).float(), -1)
            kl = (t_p * (t_logp - l_logp)).sum(-1).mean()
            losses.append(kl)
            per_layer[L] += kl.item()
        sum(losses).backward()
        n_chunks += 1
    return {L: v / n_chunks for L, v in per_layer.items()}
