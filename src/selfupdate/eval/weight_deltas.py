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


def _frobenius_norm_fp32(tensor: torch.Tensor,
                         chunk_elements: int = 4 * 1024 * 1024) -> float:
    """Stable CPU norm without materializing a full fp32 model matrix."""
    flat = tensor.reshape(-1)
    total = 0.0
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start:start + chunk_elements].float()
        total += float(chunk.square().sum(dtype=torch.float64).item())
    return total ** 0.5


def _lora_product_norm(a: torch.Tensor, b: torch.Tensor,
                       scaling: float) -> float:
    """Exact Frobenius norm of ``scaling * B @ A`` in LoRA rank space.

    ``||BA||_F^2 = tr((AA^T)(B^TB))`` avoids constructing the often
    hundreds-of-megabytes dense update merely to reduce it to one scalar.
    """
    a = a.float()
    b = b.float()
    gram_a = a @ a.T
    gram_b = b.T @ b
    squared = (gram_a * gram_b).sum(dtype=torch.float64).clamp_min(0)
    return abs(float(scaling)) * float(squared.sqrt().item())


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
        base_key = next(
            (k for k in base_state if k.endswith(f"layers.{layer}.{module}.weight")), None
        )
        w0n = (_frobenius_norm_fp32(base_state[base_key])
               if base_key else float("nan"))
        delta_norm = _lora_product_norm(adapter_state[ka], adapter_state[kb], scaling)
        rows.append({
            "layer": layer + 1,
            "module": module,
            "rel_delta": delta_norm / max(w0n, 1e-12),
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
