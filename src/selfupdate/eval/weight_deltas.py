"""Per-layer, per-module weight deltas: where did training move the weights?

Full fine-tune: relative Frobenius norm ||theta - theta0||_F / ||theta0||_F per
(layer, module). LoRA: ||B@A||_F / ||W0||_F read directly from the adapter.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import torch

LAYER_RE = re.compile(r"layers\.(\d+)\.(.+?)\.weight$")


def _layer_module(name: str) -> tuple[int, str] | None:
    m = LAYER_RE.search(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def full_ft_deltas(base_state: dict, trained_state: dict) -> pd.DataFrame:
    rows = []
    for name, w0 in base_state.items():
        lm = _layer_module(name)
        if lm is None or name not in trained_state:
            continue
        layer, module = lm
        w1 = trained_state[name].float()
        w0 = w0.float()
        rows.append({
            "layer": layer + 1,  # 1-based, matching the cache convention
            "module": module,
            "rel_delta": ((w1 - w0).norm() / w0.norm().clamp_min(1e-12)).item(),
        })
    return pd.DataFrame(rows)


def lora_deltas(base_state: dict, adapter_state: dict, scaling: float) -> pd.DataFrame:
    """adapter keys look like ...layers.N.<module>.lora_A.weight / lora_B.weight"""
    rows = []
    a_keys = [k for k in adapter_state if "lora_A" in k]
    for ka in a_keys:
        kb = ka.replace("lora_A", "lora_B")
        target = ka.split(".lora_A")[0]
        m = LAYER_RE.search(target + ".weight")
        if m is None or kb not in adapter_state:
            continue
        layer, module = int(m.group(1)), m.group(2)
        ba = (adapter_state[kb].float() @ adapter_state[ka].float()) * scaling
        base_key = next(
            (k for k in base_state if k.endswith(f"layers.{layer}.{module}.weight")), None
        )
        w0n = base_state[base_key].float().norm().item() if base_key else float("nan")
        rows.append({
            "layer": layer + 1,
            "module": module,
            "rel_delta": ba.norm().item() / max(w0n, 1e-12),
        })
    return pd.DataFrame(rows)


def per_layer_profile(df: pd.DataFrame) -> pd.Series:
    """Aggregate module deltas into one number per layer (RMS over modules)."""
    return df.groupby("layer")["rel_delta"].apply(lambda x: (x**2).mean() ** 0.5)


def load_state(path: str | Path) -> dict:
    """State dict of a HF checkpoint dir or single safetensors file (CPU)."""
    from safetensors.torch import load_file

    p = Path(path)
    if p.is_dir():
        state = {}
        for shard in sorted(p.glob("*.safetensors")):
            state.update(load_file(str(shard)))
        return state
    return load_file(str(p))
