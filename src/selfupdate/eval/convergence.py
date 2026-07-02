"""Cross-method convergence: do different trainings store the memory alike?

Given two runs' weight deltas (theta_run - theta_base, flattened per layer):
- per-layer cosine similarity of the delta vectors,
- Spearman rank correlation of the per-layer delta-norm profiles.
"""

from __future__ import annotations

import pandas as pd
import torch

from .weight_deltas import LAYER_RE


def per_layer_delta_vectors(base_state: dict, trained_state: dict) -> dict[int, torch.Tensor]:
    per_layer: dict[int, list[torch.Tensor]] = {}
    for name, w0 in base_state.items():
        m = LAYER_RE.search(name)
        if m is None or name not in trained_state:
            continue
        layer = int(m.group(1)) + 1
        d = (trained_state[name].float() - w0.float()).flatten()
        per_layer.setdefault(layer, []).append(d)
    return {L: torch.cat(ds) for L, ds in per_layer.items()}


def layer_cosines(base_state: dict, run_a: dict, run_b: dict) -> pd.DataFrame:
    da = per_layer_delta_vectors(base_state, run_a)
    db = per_layer_delta_vectors(base_state, run_b)
    rows = []
    for L in sorted(set(da) & set(db)):
        cos = torch.nn.functional.cosine_similarity(da[L], db[L], dim=0).item()
        rows.append({"layer": L, "cosine": cos,
                     "norm_a": da[L].norm().item(), "norm_b": db[L].norm().item()})
    return pd.DataFrame(rows)


def profile_spearman(df: pd.DataFrame) -> float:
    """Spearman correlation between the two runs' per-layer delta-norm profiles."""
    return df["norm_a"].rank().corr(df["norm_b"].rank(), method="pearson")
