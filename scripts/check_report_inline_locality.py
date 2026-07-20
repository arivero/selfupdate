#!/usr/bin/env python3
"""Focused CPU check for report-v2 inline-locality row adaptation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_v2 import _inline_locality_signal


def _stage(stage: int, first: int, last: int, *, skipped: bool = False):
    rows = [{
        "_stage": stage,
        "kind": "pipeline_v4_contract",
        "owned_blocks": [first, last],
    }]
    per_layer = {
        str(layer): {
            "local_grad_norm": float(layer),
            "cross_block_grad_norm": 0.0,
            "frozen_vocab_grad_norm": 0.0,
        }
        for layer in range(first, last + 1)
    }
    rows.append({
        "_stage": stage,
        "kind": "locality_certification",
        "passed": not skipped,
        "skipped": "historical_debt" if skipped else None,
        "items": last - first + 1,
        "checked_layers": list(range(first, last + 1)),
        "per_layer": per_layer,
        "local_grad_norm": 1.0,
        "cross_block_leak_grad_norm": 0.0,
        "frozen_vocab_grad_norm": 0.0,
    })
    return rows


def main() -> None:
    complete = _stage(0, 1, 2) + _stage(1, 3, 4)
    signal = _inline_locality_signal(complete, expected_stages=2)
    assert signal.get("passed") is True, signal
    assert set(map(int, signal["per_block"])) == {1, 2, 3, 4}, signal
    assert all(row["max_foreign_grad_norm"] == 0.0
               for row in signal["per_block"].values())

    skipped = _stage(0, 1, 2, skipped=True) + _stage(1, 3, 4)
    assert _inline_locality_signal(skipped) == {}
    incomplete = complete[:-1]
    assert _inline_locality_signal(incomplete) == {}
    bad_coverage = _stage(0, 1, 2)
    bad_coverage[-1]["checked_layers"] = [1]
    assert _inline_locality_signal(bad_coverage) == {}
    assert _inline_locality_signal(
        _stage(0, 1, 2), expected_stages=2) == {}
    overlap = _stage(0, 1, 2) + _stage(1, 2, 3)
    assert _inline_locality_signal(overlap, expected_stages=2) == {}
    gap = _stage(0, 1, 2) + _stage(1, 4, 5)
    assert _inline_locality_signal(gap, expected_stages=2) == {}
    print("report-v2 inline locality adapter CPU self-check: PASS")


if __name__ == "__main__":
    main()
