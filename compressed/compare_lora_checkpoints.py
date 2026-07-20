"""Compare two LoRA checkpoints without loading their base model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys


import torch
from safetensors import safe_open


LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _load(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key).float() for key in handle.keys()}


def _summary(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor],
             keys: list[str]) -> dict:
    diff2 = left2 = right2 = dot = 0.0
    max_abs = 0.0
    count = 0
    for key in keys:
        a, b = left[key].double(), right[key].double()
        d = a - b
        diff2 += d.square().sum().item()
        left2 += a.square().sum().item()
        right2 += b.square().sum().item()
        dot += (a * b).sum().item()
        max_abs = max(max_abs, d.abs().max().item())
        count += a.numel()
    return {
        "elements": count,
        "relative_l2": math.sqrt(diff2) / max(math.sqrt(left2), 1e-30),
        "cosine": dot / max(math.sqrt(left2 * right2), 1e-30),
        "max_absolute": max_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    left, right = _load(args.left), _load(args.right)
    if left.keys() != right.keys():
        raise ValueError({
            "only_left": sorted(left.keys() - right.keys()),
            "only_right": sorted(right.keys() - left.keys()),
        })
    keys = sorted(left)
    by_layer: dict[int, list[str]] = {}
    for key in keys:
        match = LAYER_RE.search(key)
        if match:
            by_layer.setdefault(int(match.group(1)) + 1, []).append(key)
    result = {
        "left": str(args.left),
        "right": str(args.right),
        "global": _summary(left, right, keys),
        "per_layer": {
            str(layer): _summary(left, right, layer_keys)
            for layer, layer_keys in sorted(by_layer.items())
        },
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
