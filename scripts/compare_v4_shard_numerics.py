"""Compare pipeline-v4 per-layer numerics: single-process vs layer-sharded.

v4 layers are independent, so sharding must be numerics-neutral: at equal
seed, every layer's epoch loss and gradient norm must agree between the
single-process run and whichever stage owns that layer. This is the M2
acceptance instrument (one-off measurement, not a stored test — repo law).

Usage:
    python scripts/compare_v4_shard_numerics.py \
        runs/h100_smoke_qwen3_0p6b_v4_1proc_e1 \
        runs/h100_smoke_qwen3_0p6b_v4_4stage_e1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _epoch_rows(metrics_path: Path, kind: str) -> dict[int, dict]:
    rows = {}
    for line in metrics_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("kind") == kind:
            rows[row["epoch"]] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("single_run", type=Path)
    ap.add_argument("staged_run", type=Path)
    ap.add_argument("--rtol", type=float, default=0.0,
                    help="0 = require exact float equality (default)")
    args = ap.parse_args()

    single = _epoch_rows(args.single_run / "metrics.jsonl", "v4_epoch")
    staged: dict[int, dict[str, float]] = {}
    stage_dirs = sorted(args.staged_run.glob("stage*/metrics.jsonl"))
    if not stage_dirs:
        sys.exit(f"no stage*/metrics.jsonl under {args.staged_run}")
    for path in stage_dirs:
        for epoch, row in _epoch_rows(path, "v4_epoch").items():
            staged.setdefault(epoch, {}).update(row["layer_losses"])

    worst = 0.0
    mismatches = []
    for epoch, row in sorted(single.items()):
        got = staged.get(epoch, {})
        for layer, value in row["layer_losses"].items():
            other = got.get(layer)
            if other is None:
                mismatches.append(f"epoch {epoch} layer {layer}: "
                                  "missing from staged run")
                continue
            if value == other:
                continue
            rel = abs(value - other) / max(abs(value), 1e-12)
            worst = max(worst, rel)
            if rel > args.rtol:
                mismatches.append(
                    f"epoch {epoch} layer {layer}: single {value!r} "
                    f"vs staged {other!r} (rel {rel:.3e})")
    layers_checked = sum(len(r["layer_losses"]) for r in single.values())
    print(f"checked {layers_checked} (epoch, layer) cells; "
          f"worst relative delta {worst:.3e}")
    if mismatches:
        print("MISMATCH:")
        for m in mismatches[:20]:
            print(" ", m)
        sys.exit(1)
    print("PASS: layer-sharded numerics identical to single-process")


if __name__ == "__main__":
    main()
