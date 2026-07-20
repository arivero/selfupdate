"""Frozen tuned-lens translators used as a local measurement device.

Raw logit-lens readouts of early layers are brittle (representation drift
across depth makes the final norm+head the wrong decoder for layer L).
The artifact loader and affine application remain because ``tuned_lens_kl``
may use an already-fitted, frozen translator.  Artifact fitting is not part of
the v4 trainer: this module contains no optimizer or backward path.
"""

from __future__ import annotations

import torch


def apply_translator(translators, L: int, h: torch.Tensor) -> torch.Tensor:
    """h + A_L h + b_L in fp32, returned in h's dtype. Missing layer (final)
    passes through."""
    key = str(L)
    if translators is None or key not in translators:
        return h
    hf = h.float()
    return (hf + translators[key](hf)).to(h.dtype)


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
