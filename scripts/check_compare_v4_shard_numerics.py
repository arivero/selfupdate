#!/usr/bin/env python3
"""Disposable synthetic self-check for the v4 numerics comparator."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "scripts" / "compare_v4_shard_numerics.py"


def _base(kind: str, epoch: int, stage: int, splits: list[int]) -> dict:
    return {
        "kind": kind, "epoch": epoch, "source_commit": "fresh-commit",
        "runtime_dirty": False, "runtime_diff_sha256": None,
        "runtime_untracked": [], "loss_kind": "delta_cosine",
        "v4_stage": stage, "v4_stage_splits": splits,
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _cert(layers: list[int]) -> dict:
    keys = [str(layer) for layer in layers]
    return {
        "kind": "locality_certification", "passed": True,
        "source_commit": "fresh-commit", "loss_kind": "delta_cosine",
        "runtime_dirty": False, "runtime_diff_sha256": None,
        "runtime_untracked": [],
        "checked_layers": keys,
        "per_layer": {key: {"finite_positive": True} for key in keys},
    }


def _run(single: Path, staged: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(COMPARE), str(single), str(staged),
         "--strict-current"], text=True, capture_output=True)


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        single, staged = root / "single", root / "staged"
        loss = _base("v4_epoch", 1, -1, [])
        loss["layer_losses"] = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0}
        grad = _base("v4_gradient_norm", 1, -1, [])
        grad["grad_norms"] = {"1": 10.0, "2": 20.0, "3": 30.0, "4": 40.0}
        _write(single / "metrics.jsonl", [loss, grad, _cert([1, 2, 3, 4])])
        for stage, layers in enumerate(([1, 2], [3, 4])):
            s_loss = _base("v4_epoch", 1, stage, [2])
            s_loss["layer_losses"] = {
                str(layer): loss["layer_losses"][str(layer)] for layer in layers}
            s_grad = _base("v4_gradient_norm", 1, stage, [2])
            s_grad["grad_norms"] = {
                str(layer): grad["grad_norms"][str(layer)] for layer in layers}
            _write(staged / f"stage{stage}" / "metrics.jsonl",
                   [s_loss, s_grad, _cert(list(layers))])

        passed = _run(single, staged)
        assert passed.returncode == 0, passed.stdout + passed.stderr
        assert "28" not in passed.stdout  # no campaign-specific fixture
        assert "4 loss cells and 4 gradient cells" in passed.stdout

        # An extra/overlapping layer and a gradient mismatch must each make
        # strict admission fail; restore between independent probes.
        stage1 = staged / "stage1" / "metrics.jsonl"
        rows = [json.loads(line) for line in stage1.read_text().splitlines()]
        rows[0]["layer_losses"]["2"] = 2.0
        _write(stage1, rows)
        failed = _run(single, staged)
        assert failed.returncode == 1 and "ownership" in failed.stdout

        rows[0]["layer_losses"].pop("2")
        rows[1]["grad_norms"]["4"] = 41.0
        _write(stage1, rows)
        failed = _run(single, staged)
        assert failed.returncode == 1 and "grad_norms" in failed.stdout

        rows[1]["grad_norms"]["4"] = 40.0
        rows[0]["runtime_untracked"] = ["scratch.txt"]
        _write(stage1, rows)
        failed = _run(single, staged)
        assert failed.returncode == 1 and "runtime_untracked" in failed.stdout

        rows[0]["runtime_untracked"] = []
        rows[2]["skipped"] = "synthetic_debt"
        _write(stage1, rows)
        failed = _run(single, staged)
        assert failed.returncode == 1 and "locality passed" in failed.stdout
    print("v4 shard comparator synthetic self-check: PASS")


if __name__ == "__main__":
    main()
