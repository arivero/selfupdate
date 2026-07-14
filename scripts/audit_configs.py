"""Audit active configs for branch-law violations.

This script is intentionally raw-YAML first: dataclass construction ignores
unknown keys, so old keys must be caught before ``load_config`` can silently
drop them.

Usage:
    python scripts/audit_configs.py
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.config import load_config  # noqa: E402
from selfupdate.train.validate import validate_knob_schedule as _validate_knob_schedule  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "configs/base.yaml"
EXPERIMENTS = ROOT / "configs/experiments"

OLD_KEYS = {
    "readout_window_blocks",
    "readout_weight",
    "readout_source",
    "anchor_kl_weight",
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

# Structural companion to the name-based OLD_KEYS list (2026-07-10 review:
# a RENAMED banned knob slips past an exact-name blacklist).  The banned
# concepts have a shape — cross-entropy terms, label targeting, the purged
# word — so any train key matching these patterns is flagged for explicit
# review rather than silently accepted.
_BANNED_KEY_PATTERNS = (
    re.compile(r"(^|_)ce(_|$)"),  # ..._ce_..., ce_weight, tail_ce
    re.compile("label"),
    re.compile("gold"),           # purged lexicon (owner 2026-07-05)
)


def _pattern_banned_keys(train: dict) -> list[str]:
    return sorted(
        k for k in train
        if k not in OLD_KEYS and any(p.search(k) for p in _BANNED_KEY_PATTERNS)
    )


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
    shaped = _pattern_banned_keys(train)
    if shaped:
        issues.append(Issue(
            path, "train keys match banned-concept patterns (ce/label/gold) "
                  "and need explicit review: " + ", ".join(shaped)))
    try:
        cfg = load_config(base, None if path == base else path)
    except Exception as e:  # noqa: BLE001
        issues.append(Issue(path, f"load_config failed: {type(e).__name__}: {e}"))
        return issues

    if cfg.train.run_class == "legacy_archive":
        issues.append(Issue(path, "legacy_archive configs do not belong in active configs/experiments"))

    try:
        _validate_knob_schedule(cfg)
    except Exception as e:  # noqa: BLE001
        issues.append(Issue(path, f"trainer validation failed: {e}"))
    return issues


def _pending_queue_experiment_paths(root: Path = ROOT) -> set[Path]:
    """--experiment paths referenced by rows the scheduler could still
    dispatch: done_file not yet satisfied and not `#`-disabled
    (gpu_scheduler.sh: `[ -e "$done" ] && continue` /
    `case "$done" in \\#*) continue;; esac`). A row whose done_file already
    exists is permanently skipped regardless of what its config contains —
    flagging it would just make the audit fail on harmless history."""
    paths: set[Path] = set()
    for qpath in sorted(root.glob("scripts/queue*.tsv")):
        for line in qpath.read_text().splitlines()[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            done_file, _need, _after, cmd = parts[0], parts[1], parts[2], parts[3]
            if done_file.startswith("#") or (root / done_file).exists():
                continue
            m = re.search(r"--experiment (\S+)", cmd)
            if m:
                paths.add(root / m.group(1))
    return paths


def audit_queue_snapshots(experiments: Path = EXPERIMENTS,
                          root: Path = ROOT) -> list[Issue]:
    """Queue TSVs can point --experiment at a run-dir config SNAPSHOT
    (runs/<run>/config.yaml) instead of a configs/experiments/*.yaml file —
    outside audit_all's scan perimeter. Restricted to rows the scheduler
    could still dispatch (see _pending_queue_experiment_paths); the queue
    is mostly a completed-campaign history and flagging harmless done rows
    would just break the audit gate. load_config already fail-loud rejects
    unknown keys at dispatch time (config.py), but that means a queued job
    discovers a stale/renamed knob only when the scheduler actually reaches
    it, mid-campaign. Catch it offline instead."""
    issues: list[Issue] = []
    for path in sorted(_pending_queue_experiment_paths(root)):
        try:
            if experiments in path.parents:
                continue  # already covered by audit_all's own scan
        except ValueError:
            pass
        if not path.exists():
            issues.append(Issue(path, "queue references a missing --experiment path"))
            continue
        raw, raw_issues = _load_raw(path)
        issues.extend(raw_issues)
        if raw_issues:
            continue
        train = raw.get("train", {}) or {}
        old = sorted(k for k in OLD_KEYS if k in train)
        if old:
            issues.append(Issue(
                path, "queue-referenced snapshot carries old banned train "
                     "keys (will fail loud at dispatch): " + ", ".join(old)))
    return issues


def audit_all(base: Path = BASE, experiments: Path = EXPERIMENTS) -> list[Issue]:
    paths = [base] + sorted(experiments.rglob("*.yaml"))
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
    issues.extend(audit_queue_snapshots(experiments))
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
