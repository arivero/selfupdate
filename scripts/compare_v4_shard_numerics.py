#!/usr/bin/env python3
"""Compare pipeline-v4 single-process and layer-sharded numerics.

The default preserves the historical, loss-only comparison. Fresh admission
runs should pass ``--strict-current``: it additionally proves clean/same code
and objective provenance, exact stage ownership, gradient-norm numerics, and
passed non-skipped locality certificates. References remain disposable; this
script stores no expected numerical fingerprint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


def _rows(metrics_path: Path, kind: str) -> list[dict]:
    if not metrics_path.is_file():
        raise ValueError(f"missing metrics file: {metrics_path}")
    found = []
    for lineno, line in enumerate(metrics_path.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{metrics_path}:{lineno}: {exc}") from exc
        if row.get("kind") == kind:
            found.append(row)
    return found


def _epoch_rows(metrics_path: Path, kind: str, *, strict: bool) -> dict[int, dict]:
    rows = {}
    for row in _rows(metrics_path, kind):
        epoch = row.get("epoch")
        if not isinstance(epoch, int):
            raise ValueError(f"{metrics_path}: {kind} row lacks integer epoch")
        if strict and epoch in rows:
            raise ValueError(
                f"{metrics_path}: duplicate {kind} row for epoch {epoch}")
        rows[epoch] = row
    return rows


def _stage_paths(staged_run: Path) -> dict[int, Path]:
    result = {}
    for path in staged_run.glob("stage*/metrics.jsonl"):
        match = re.fullmatch(r"stage(\d+)", path.parent.name)
        if not match:
            continue
        stage = int(match.group(1))
        if stage in result:
            raise ValueError(f"duplicate stage directory for stage {stage}")
        result[stage] = path
    if not result:
        raise ValueError(f"no stage*/metrics.jsonl under {staged_run}")
    return result


def _compare_cells(label: str, expected: dict[int, dict[str, float]],
                   actual: dict[int, dict[str, float]], rtol: float,
                   mismatches: list[str]) -> tuple[int, float]:
    checked, worst = 0, 0.0
    if set(actual) != set(expected):
        mismatches.append(
            f"{label}: epoch set {sorted(actual)} != {sorted(expected)}")
    for epoch, wanted in sorted(expected.items()):
        got = actual.get(epoch, {})
        if set(got) != set(wanted):
            missing = sorted(set(wanted) - set(got), key=int)
            extra = sorted(set(got) - set(wanted), key=int)
            mismatches.append(
                f"{label} epoch {epoch}: layer cells missing={missing} "
                f"extra={extra}")
        for layer, value in wanted.items():
            if layer not in got:
                continue
            checked += 1
            other = got[layer]
            if not (math.isfinite(value) and math.isfinite(other)):
                mismatches.append(
                    f"{label} epoch {epoch} layer {layer}: non-finite "
                    f"single={value!r} staged={other!r}")
                continue
            if value == other:
                continue
            rel = abs(value - other) / max(abs(value), 1e-12)
            worst = max(worst, rel)
            if rel > rtol:
                mismatches.append(
                    f"{label} epoch {epoch} layer {layer}: single "
                    f"{value!r} vs staged {other!r} (rel {rel:.3e})")
    return checked, worst


def _legacy(single_path: Path, stage_paths: dict[int, Path], rtol: float) -> int:
    single_rows = _epoch_rows(single_path, "v4_epoch", strict=False)
    staged: dict[int, dict[str, float]] = {}
    for path in stage_paths.values():
        for epoch, row in _epoch_rows(path, "v4_epoch", strict=False).items():
            staged.setdefault(epoch, {}).update(row["layer_losses"])
    expected = {epoch: row["layer_losses"]
                for epoch, row in single_rows.items()}
    # Historical behavior compared only cells present in the single run; keep
    # it available explicitly by omitting --strict-current.
    projected = {epoch: {layer: staged.get(epoch, {})[layer]
                         for layer in values if layer in staged.get(epoch, {})}
                 for epoch, values in expected.items()}
    mismatches = []
    checked, worst = _compare_cells(
        "layer_losses", expected, projected, rtol, mismatches)
    return _finish(checked, 0, worst, mismatches, strict=False)


def _clean_metadata(row: dict, label: str, mismatches: list[str]) -> None:
    if row.get("runtime_dirty") is not False:
        mismatches.append(f"{label}: runtime_dirty is not false")
    if row.get("runtime_diff_sha256") not in (None, ""):
        mismatches.append(f"{label}: runtime_diff_sha256 is present")
    if row.get("runtime_untracked") != []:
        mismatches.append(
            f"{label}: runtime_untracked={row.get('runtime_untracked')!r}")


def _strict(single_path: Path, stage_paths: dict[int, Path], rtol: float) -> int:
    mismatches: list[str] = []
    single_loss = _epoch_rows(single_path, "v4_epoch", strict=True)
    single_grad = _epoch_rows(single_path, "v4_gradient_norm", strict=True)
    if not single_loss:
        mismatches.append("single run has no v4_epoch rows")
        return _finish(0, 0, 0.0, mismatches, strict=True)
    first_single = single_loss[min(single_loss)]
    expected_layers = set(first_single.get("layer_losses", {}))
    numeric_layers = sorted((int(layer) for layer in expected_layers))
    if not numeric_layers or numeric_layers != list(range(1, max(numeric_layers) + 1)):
        mismatches.append(
            f"single ownership is not contiguous L1..Ln: {numeric_layers}")
    n_layers = max(numeric_layers, default=0)
    all_layers = {str(layer) for layer in range(1, n_layers + 1)}
    for epoch, row in single_loss.items():
        if set(row.get("layer_losses", {})) != all_layers:
            mismatches.append(
                f"single v4_epoch {epoch} does not own exact L1..L{n_layers}")
    for epoch, row in single_grad.items():
        if set(row.get("grad_norms", {})) != all_layers:
            mismatches.append(
                f"single v4_gradient_norm {epoch} does not cover exact L1..L{n_layers}")
    if set(single_grad) != set(single_loss):
        mismatches.append("single gradient epochs differ from loss epochs")

    commits = {first_single.get("source_commit")}
    losses = {first_single.get("loss_kind")}
    for epoch, row in single_loss.items():
        commits.add(row.get("source_commit"))
        losses.add(row.get("loss_kind"))
        _clean_metadata(row, f"single epoch {epoch}", mismatches)
    for epoch, row in single_grad.items():
        commits.add(row.get("source_commit"))
        losses.add(row.get("loss_kind"))
        _clean_metadata(row, f"single gradient epoch {epoch}", mismatches)
    if not first_single.get("source_commit"):
        mismatches.append("single source_commit is empty")
    if not first_single.get("loss_kind"):
        mismatches.append("single loss_kind is empty")

    staged_loss_rows: dict[int, dict[int, dict]] = {}
    staged_grad_rows: dict[int, dict[int, dict]] = {}
    splits = None
    seen_stage_ids = set()
    for directory_stage, path in sorted(stage_paths.items()):
        loss_rows = _epoch_rows(path, "v4_epoch", strict=True)
        grad_rows = _epoch_rows(path, "v4_gradient_norm", strict=True)
        staged_loss_rows[directory_stage] = loss_rows
        staged_grad_rows[directory_stage] = grad_rows
        if not loss_rows:
            mismatches.append(f"stage{directory_stage} has no v4_epoch rows")
            continue
        first = loss_rows[min(loss_rows)]
        reported_stage = first.get("v4_stage")
        if reported_stage != directory_stage:
            mismatches.append(
                f"stage{directory_stage} row reports v4_stage={reported_stage!r}")
        if reported_stage in seen_stage_ids:
            mismatches.append(f"duplicate reported v4_stage={reported_stage}")
        seen_stage_ids.add(reported_stage)
        row_splits = list(first.get("v4_stage_splits") or [])
        if splits is None:
            splits = row_splits
        elif splits != row_splits:
            mismatches.append(
                f"stage{directory_stage} splits {row_splits} != {splits}")
        for epoch, row in loss_rows.items():
            commits.add(row.get("source_commit"))
            losses.add(row.get("loss_kind"))
            _clean_metadata(
                row, f"stage{directory_stage} epoch {epoch}", mismatches)
        for epoch, row in grad_rows.items():
            commits.add(row.get("source_commit"))
            losses.add(row.get("loss_kind"))
            _clean_metadata(
                row, f"stage{directory_stage} gradient epoch {epoch}",
                mismatches)

    splits = splits or []
    expected_stage_ids = set(range(len(splits) + 1))
    if set(stage_paths) != expected_stage_ids:
        mismatches.append(
            f"stage directories {sorted(stage_paths)} != expected "
            f"{sorted(expected_stage_ids)} from splits {splits}")
    valid_splits = not (
        any(not isinstance(cut, int) for cut in splits)
        or splits != sorted(set(splits))
        or any(cut <= 0 or cut >= n_layers for cut in splits)
    )
    if not valid_splits:
        mismatches.append(f"invalid stage splits for {n_layers} layers: {splits}")
    boundaries = [0, *splits, n_layers] if valid_splits else [0, n_layers]
    expected_owned = {
        stage: {str(layer) for layer in range(boundaries[stage] + 1,
                                               boundaries[stage + 1] + 1)}
        for stage in range(len(boundaries) - 1)
    }

    staged_loss: dict[int, dict[str, float]] = {}
    staged_grad: dict[int, dict[str, float]] = {}
    for stage, rows in staged_loss_rows.items():
        owned = expected_owned.get(stage, set())
        if set(rows) != set(single_loss):
            mismatches.append(
                f"stage{stage} loss epochs {sorted(rows)} != single "
                f"{sorted(single_loss)}")
        for epoch, row in rows.items():
            cells = row.get("layer_losses", {})
            if set(cells) != owned:
                mismatches.append(
                    f"stage{stage} epoch {epoch} loss ownership "
                    f"{sorted(cells, key=int)} != {sorted(owned, key=int)}")
            target = staged_loss.setdefault(epoch, {})
            overlap = set(target) & set(cells)
            if overlap:
                mismatches.append(
                    f"epoch {epoch} duplicate loss ownership: "
                    f"{sorted(overlap, key=int)}")
            target.update(cells)
    for stage, rows in staged_grad_rows.items():
        owned = expected_owned.get(stage, set())
        if set(rows) != set(single_grad):
            mismatches.append(
                f"stage{stage} gradient epochs {sorted(rows)} != single "
                f"{sorted(single_grad)}")
        for epoch, row in rows.items():
            cells = row.get("grad_norms", {})
            if set(cells) != owned:
                mismatches.append(
                    f"stage{stage} epoch {epoch} gradient ownership "
                    f"{sorted(cells, key=int)} != {sorted(owned, key=int)}")
            target = staged_grad.setdefault(epoch, {})
            overlap = set(target) & set(cells)
            if overlap:
                mismatches.append(
                    f"epoch {epoch} duplicate gradient ownership: "
                    f"{sorted(overlap, key=int)}")
            target.update(cells)

    _check_locality(single_path, all_layers, "single", mismatches,
                    commits, losses)
    for stage, path in stage_paths.items():
        _check_locality(path, expected_owned.get(stage, set()),
                        f"stage{stage}", mismatches, commits, losses)
    if len(commits) != 1:
        mismatches.append(f"source commits differ: {sorted(map(str, commits))}")
    if len(losses) != 1:
        mismatches.append(f"loss kinds differ: {sorted(map(str, losses))}")

    expected_loss = {epoch: row["layer_losses"]
                     for epoch, row in single_loss.items()}
    expected_grad = {epoch: row["grad_norms"]
                     for epoch, row in single_grad.items()}
    checked_loss, worst_loss = _compare_cells(
        "layer_losses", expected_loss, staged_loss, rtol, mismatches)
    checked_grad, worst_grad = _compare_cells(
        "grad_norms", expected_grad, staged_grad, rtol, mismatches)
    return _finish(checked_loss, checked_grad, max(worst_loss, worst_grad),
                   mismatches, strict=True)


def _check_locality(path: Path, expected: set[str], label: str,
                    mismatches: list[str], commits: set,
                    losses: set) -> None:
    rows = _rows(path, "locality_certification")
    if len(rows) != 1:
        mismatches.append(
            f"{label}: expected exactly one locality_certification, got {len(rows)}")
        return
    cert = rows[0]
    commits.add(cert.get("source_commit"))
    losses.add(cert.get("loss_kind"))
    _clean_metadata(cert, f"{label} locality", mismatches)
    if cert.get("passed") is not True or cert.get("skipped"):
        mismatches.append(
            f"{label}: locality passed={cert.get('passed')!r}, "
            f"skipped={cert.get('skipped')!r}")
    checked = {str(layer) for layer in (cert.get("checked_layers") or [])}
    per_layer = set((cert.get("per_layer") or {}).keys())
    if checked != expected:
        mismatches.append(
            f"{label}: locality checked_layers {sorted(checked, key=int)} != "
            f"expected {sorted(expected, key=int)}")
    if per_layer and per_layer != expected:
        mismatches.append(
            f"{label}: locality per_layer ownership "
            f"{sorted(per_layer, key=int)} != expected "
            f"{sorted(expected, key=int)}")


def _finish(loss_cells: int, grad_cells: int, worst: float,
            mismatches: list[str], *, strict: bool) -> int:
    mode = "strict-current" if strict else "historical loss-only"
    print(f"mode: {mode}")
    print(f"checked {loss_cells} loss cells and {grad_cells} gradient cells; "
          f"worst relative delta {worst:.3e}")
    if mismatches:
        print("MISMATCH:")
        for mismatch in mismatches[:40]:
            print(" ", mismatch)
        if len(mismatches) > 40:
            print(f"  ... {len(mismatches) - 40} more")
        return 1
    if strict:
        print("PASS: layer-sharded numerics and contracts match single-process")
    else:
        print("PASS: historical layer-loss numerics match single-process")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("single_run", type=Path)
    ap.add_argument("staged_run", type=Path)
    ap.add_argument("--rtol", type=float, default=0.0,
                    help="0 = require exact float equality (default)")
    ap.add_argument(
        "--strict-current", action="store_true",
        help="require clean/same provenance, exact ownership, gradient norms, "
             "and passed non-skipped locality certificates")
    args = ap.parse_args()
    if args.rtol < 0:
        ap.error("--rtol must be non-negative")
    try:
        stage_paths = _stage_paths(args.staged_run)
        rc = (_strict(args.single_run / "metrics.jsonl", stage_paths, args.rtol)
              if args.strict_current else
              _legacy(args.single_run / "metrics.jsonl", stage_paths, args.rtol))
    except (ValueError, KeyError, TypeError) as exc:
        print(f"INVALID ARTIFACT: {exc}", file=sys.stderr)
        rc = 2
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
