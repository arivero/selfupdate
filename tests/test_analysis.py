"""CPU unit tests for the M4 analysis math (synthetic state dicts)."""

import torch

from selfupdate.eval.convergence import layer_cosines, per_layer_delta_vectors, profile_spearman
from selfupdate.eval.weight_deltas import full_ft_deltas, lora_deltas, per_layer_profile


def _state(n_layers=4, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for i in range(n_layers):
        for mod in ("self_attn.q_proj", "mlp.down_proj"):
            sd[f"model.layers.{i}.{mod}.weight"] = torch.randn(dim, dim, generator=g)
    sd["model.embed_tokens.weight"] = torch.randn(16, dim, generator=g)  # ignored
    return sd


def test_full_ft_deltas_localized():
    base = _state()
    trained = {k: v.clone() for k, v in base.items()}
    # perturb only layer 2 (0-based) => reported as layer 3 (1-based)
    trained["model.layers.2.self_attn.q_proj.weight"] += 0.5
    df = full_ft_deltas(base, trained)
    prof = per_layer_profile(df)
    assert prof.idxmax() == 3
    assert prof[3] > 10 * prof.drop(3).max()
    assert not any("embed_tokens" in str(m) for m in df["module"])


def test_lora_deltas_zero_B_is_zero():
    base = _state()
    adapter = {
        "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight": torch.randn(4, 8),
        "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight": torch.zeros(8, 4),
    }
    df = lora_deltas(base, adapter, scaling=2.0)
    assert len(df) == 1
    assert df.iloc[0]["layer"] == 2
    assert df.iloc[0]["rel_delta"] == 0.0


def test_layer_cosines_identical_runs():
    base = _state()
    run = {k: v + 0.1 for k, v in base.items()}
    df = layer_cosines(base, run, run)
    assert (df["cosine"] > 0.9999).all()
    assert profile_spearman(df) > 0.99


def test_layer_cosines_orthogonal_perturbations():
    base = _state(dim=64)
    key = "model.layers.0.self_attn.q_proj.weight"
    d1 = torch.zeros(64, 64)
    d1[0, :] = 1.0
    d2 = torch.zeros(64, 64)
    d2[1, :] = 1.0  # disjoint rows -> orthogonal deltas
    run_a = {k: (v + d1 if k == key else v.clone()) for k, v in base.items()}
    run_b = {k: (v + d2 if k == key else v.clone()) for k, v in base.items()}
    df = layer_cosines(base, run_a, run_b)
    row = df[df.layer == 1].iloc[0]
    assert abs(row["cosine"]) < 1e-6


def test_delta_vectors_cover_all_layers():
    base = _state(n_layers=5)
    run = {k: v + 0.01 for k, v in base.items()}
    vecs = per_layer_delta_vectors(base, run)
    assert sorted(vecs) == [1, 2, 3, 4, 5]


def test_strip_think():
    from selfupdate.eval.recite import strip_think

    assert strip_think("<think>razono...</think>\nEn la tierra") == "\nEn la tierra"
    assert strip_think("En la tierra") == "En la tierra"          # no block: untouched
    assert strip_think("  <think>sin cierre") == ""               # unclosed = failure
    assert strip_think("mid <think>x</think>") == "mid <think>x</think>"  # only leading
