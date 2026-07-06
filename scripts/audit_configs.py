"""Audit active configs for branch-law violations.

This script is intentionally raw-YAML first: dataclass construction ignores
unknown keys, so old keys must be caught before ``load_config`` can silently
drop them.

Usage:
    python scripts/audit_configs.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.config import load_config  # noqa: E402
from selfupdate.train.layerwise import _validate_knob_schedule  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "configs/base.yaml"
EXPERIMENTS = ROOT / "configs/experiments"

OLD_KEYS = {
    "tail_ce_blocks",
    "tail_ce_weight",
    "tail_ce_kind",
    "tail_hidden_weight",
    "last_block_ce_weight",
    "lens_ce_weight",
    "lens_ce_from",
    "answer_ce_weight",
    "last_block_" + "task" + "_label_weight",
    "lens_" + "task" + "_label_weight",
    "anchor_" + "ce_weight",
    "lens_" + "from_layer",
}
FORBIDDEN_REFERENCE_SOURCE = "task" + "_label"


@dataclass
class Issue:
    path: Path
    message: str


def _load_raw(path: Path) -> tuple[dict, list[Issue]]:
    try:
        return yaml.safe_load(path.read_text()) or {}, []
    except Exception as e:  # noqa: BLE001
        return {}, [Issue(path, f"YAML parse error: {type(e).__name__}: {e}")]


def audit_one(path: Path, base: Path = BASE) -> list[Issue]:
    issues: list[Issue] = []
    raw, raw_issues = _load_raw(path)
    issues.extend(raw_issues)
    if raw_issues:
        return issues
    train = raw.get("train", {}) or {}
    old = sorted(k for k in OLD_KEYS if k in train)
    if old:
        issues.append(Issue(path, "old banned train keys present: " + ", ".join(old)))
    if train.get("readout_source") == FORBIDDEN_REFERENCE_SOURCE:
        issues.append(Issue(path, "forbidden reference-text readout source"))
    if path == base and "readout_source" in train:
        issues.append(Issue(path, "base.yaml must not set readout_source"))

    try:
        cfg = load_config(base, None if path == base else path)
    except Exception as e:  # noqa: BLE001
        issues.append(Issue(path, f"load_config failed: {type(e).__name__}: {e}"))
        return issues

    needs_source = (
        cfg.train.readout_window_blocks > 0
        or cfg.train.readout_weight > 0
        or train.get("readout_window_blocks", 0) > 0
        or train.get("readout_weight", 0) > 0
    )
    if needs_source and "readout_source" not in train:
        issues.append(Issue(path, "readout_source must be explicit in the experiment file"))
    if cfg.train.run_class == "legacy_archive":
        issues.append(Issue(path, "legacy_archive configs do not belong in active configs/experiments"))

    try:
        _validate_knob_schedule(cfg)
    except Exception as e:  # noqa: BLE001
        issues.append(Issue(path, f"trainer validation failed: {e}"))
    return issues


def audit_all(base: Path = BASE, experiments: Path = EXPERIMENTS) -> list[Issue]:
    paths = [base] + sorted(experiments.glob("*.yaml"))
    issues: list[Issue] = []
    run_names: dict[str, list[Path]] = {}
    for path in paths:
        issues.extend(audit_one(path, base))
        if path == base:
            continue
        raw, raw_issues = _load_raw(path)
        if raw_issues:
            continue
        if "run_name" not in raw:
            issues.append(Issue(path, "experiment files must pin run_name explicitly"))
            continue
        run = raw.get("run_name")
        if isinstance(run, str) and run:
            run_names.setdefault(run, []).append(path)
    for run, owners in sorted(run_names.items()):
        if len(owners) > 1:
            joined = ", ".join(str(p) for p in owners)
            issues.append(Issue(owners[0], f"duplicate run_name {run!r}: {joined}"))
    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--experiments", type=Path, default=EXPERIMENTS)
    args = ap.parse_args()

    issues = audit_all(args.base, args.experiments)
    if issues:
        for issue in issues:
            print(f"{issue.path}: {issue.message}", file=sys.stderr)
        return 1
    print("config audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
