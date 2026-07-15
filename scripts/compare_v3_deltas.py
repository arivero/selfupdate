"""Compare exact trainable-parameter deltas from two v3 smoke probes."""

from __future__ import annotations

import argparse
import json
import re

import torch
from safetensors.torch import load_file


LAYER_RE = re.compile(r"^layer(\d+)\.")


def _stats(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]):
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        raise ValueError(f"delta keys differ: missing={missing}, extra={extra}")
    accum: dict[int, list[float]] = {}
    overall = [0.0, 0.0, 0.0, 0.0]
    for key in sorted(reference):
        match = LAYER_RE.match(key)
        if match is None:
            raise ValueError(f"unrecognized delta key {key!r}")
        layer = int(match.group(1))
        ref = reference[key].double().reshape(-1)
        got = candidate[key].double().reshape(-1)
        if ref.shape != got.shape:
            raise ValueError(f"shape differs for {key}: {ref.shape} != {got.shape}")
        values = accum.setdefault(layer, [0.0, 0.0, 0.0, 0.0])
        ref_sq = float(torch.dot(ref, ref))
        got_sq = float(torch.dot(got, got))
        diff_sq = float(torch.dot(got - ref, got - ref))
        dot = float(torch.dot(ref, got))
        for dest in (values, overall):
            dest[0] += ref_sq
            dest[1] += got_sq
            dest[2] += diff_sq
            dest[3] += dot
    return accum, overall


def _render(values: list[float]) -> dict[str, float | None]:
    ref_sq, got_sq, diff_sq, dot = values
    ref_norm = ref_sq ** 0.5
    got_norm = got_sq ** 0.5
    diff_norm = diff_sq ** 0.5
    denom = ref_norm * got_norm
    return {
        "reference_l2": ref_norm,
        "candidate_l2": got_norm,
        "divergence_l2": diff_norm,
        "relative_divergence_to_reference": (
            diff_norm / ref_norm if ref_norm else None),
        "cosine": (dot / denom if denom else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    reference = load_file(args.reference, device="cpu")
    candidate = load_file(args.candidate, device="cpu")
    by_layer, overall = _stats(reference, candidate)
    print(json.dumps({
        "reference": args.reference,
        "candidate": args.candidate,
        "overall": _render(overall),
        "per_layer": {
            str(layer): _render(values)
            for layer, values in sorted(by_layer.items())
        },
    }, indent=2))


if __name__ == "__main__":
    main()
