"""Layer grafting and ablation: causal test of where the memory lives.

Graft: copy trained block L into a fresh base model — does recitation improve?
Ablate: revert block L of the trained model to its initial weights — does
recitation break? Both directions per layer give a causal localization curve
that per-layer delta norms alone cannot (cf. Hase et al. 2023: storage and
editability can disagree).
"""

from __future__ import annotations

import torch

from .recite import recite_eval


def _copy_block(dst_model, src_state: dict, layer_idx0: int) -> dict[str, torch.Tensor]:
    """Overwrite block ``layer_idx0`` of dst with weights from src_state.
    Returns the overwritten tensors for undo."""
    prefix = f"model.layers.{layer_idx0}."
    undo = {}
    sd = dst_model.state_dict()
    for name, tensor in src_state.items():
        if name.startswith(prefix):
            undo[name] = sd[name].clone()
            sd[name].copy_(tensor.to(sd[name].dtype).to(sd[name].device))
    assert undo, f"no tensors matched {prefix}"
    return undo


def _restore(dst_model, undo: dict[str, torch.Tensor]) -> None:
    sd = dst_model.state_dict()
    for name, tensor in undo.items():
        sd[name].copy_(tensor)


@torch.no_grad()
def swap_curves(
    base_model,
    trained_state: dict,
    base_state: dict,
    trained_model,
    tokenizer,
    records: list[dict],
    limit: int = 8,
    layers: list[int] | None = None,
) -> list[dict]:
    """For each layer L (1-based): graft CER (base+trained block L) and
    ablate CER (trained model with block L reverted)."""
    n = base_model.config.num_hidden_layers
    rows = []
    for L in layers or range(1, n + 1):
        undo = _copy_block(base_model, trained_state, L - 1)
        graft = recite_eval(base_model, tokenizer, records, limit=limit)
        _restore(base_model, undo)

        undo = _copy_block(trained_model, base_state, L - 1)
        ablate = recite_eval(trained_model, tokenizer, records, limit=limit)
        _restore(trained_model, undo)

        rows.append({
            "layer": L,
            "graft_cer": graft["cer"], "graft_line_exact": graft["line_exact"],
            "ablate_cer": ablate["cer"], "ablate_line_exact": ablate["line_exact"],
        })
        print(f"layer {L:2d}: graft CER {graft['cer']:.3f}  ablate CER {ablate['cer']:.3f}")
    return rows
