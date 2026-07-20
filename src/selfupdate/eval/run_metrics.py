"""Shared per-run ``metrics.jsonl`` loader for report tooling.

Two independent axes of variation this module resolves once, so
``report_v2.py`` / ``layer_loss_plots.py`` / any future report consumer
never re-derives them (and never re-introduces the schema/stage bugs this
module was written to fix — see issues.md for the diagnosis):

Run layout
----------
- flat (pre-pipeline-v4, or ``v4_stage < 0``): one process, one
  ``run_dir/metrics.jsonl``.
- stage-scoped (pipeline-v4 PPP-N, N > 1): N OS processes, each writing
  ``run_dir/stageK/metrics.jsonl`` (+ its own ``config.yaml``/``checkpoint``).
  There is no top-level ``run_dir/metrics.jsonl`` for these runs.

Per-layer schema
-----------------
- legacy: ``kind=="train"`` + list field ``per_layer``; ``kind==
  "v3_gradient_norm"`` + list field ``per_layer_mean``. Always single-stage,
  so local list index ``i`` is global layer ``i + 1`` directly.
- v4 (``src/selfupdate/train/online_v4.py``): ``kind=="v4_epoch"`` + dict
  field ``layer_losses``; ``kind=="v4_gradient_norm"`` + dict field
  ``grad_norms``. Both dicts are keyed by the GLOBAL 1-based layer number
  as a string already (see ``online_v4.py`` around line 1734: ``layer_losses
  [str(layer)] = ...`` where ``layer`` comes from ``owned`` — the global
  ``_owned_range()`` — not a local 0-based loop index).
  ``kind=="parameter_delta"`` (``src/selfupdate/train/telemetry.py``,
  shared by both pipelines) carries list fields ``per_layer_absolute_l2`` /
  ``per_layer_relative_l2`` / ``per_layer_parameter_count``; empirically
  (verified against ``runs/h100_g26b_v4_ppp4_e40``) these are already
  full-model-depth lists with zero entries at layers this stage does not
  own, because ``ParameterDeltaTracker`` iterates ``range(1, stack.n_layers
  + 1)`` where ``stack.n_layers`` is the FULL model depth in every stage
  process, not the local owned count.

Every stage's ``metrics.jsonl`` carries exactly one ``kind==
"pipeline_v4_contract"`` row with ``owned_blocks: [first, last]`` — a
one-based inclusive GLOBAL layer range (``online_v4.py`` ``_owned_range()``
docstring: "One-based inclusive block range this process trains"). This
module reads that row once per stage and uses it to place every per-layer
value at its correct global layer, defensively handling both the
already-global-length list shape (current ``parameter_delta`` reality) and
a hypothetical local-length list/dict shape, so a future emission-shape
change does not silently mislabel layers again.

Scalar per-epoch kinds (``eval``, ``standard_eval``, ``teacher_output_eval``,
``done``, ...) need no offset correction: each is emitted by exactly one
stage's process (confirmed empirically — ``eval``/``standard_eval`` only in
stage0, ``teacher_output_eval`` only in the stage owning the frozen
vocabulary head/LM head). Simple concatenation across stages is correct for
these; this module just does that via ``RunMetrics.rows``.
"""

from __future__ import annotations

import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def stage_dirs(run_dir: Path) -> list[Path]:
    """Sorted stageK/ subdirectories that actually hold metrics.jsonl."""
    candidates = [
        p for p in run_dir.glob("stage*")
        if p.is_dir() and (p / "metrics.jsonl").is_file()
        and p.name.removeprefix("stage").isdigit()
    ]
    return sorted(candidates, key=lambda p: int(p.name.removeprefix("stage")))


def is_stage_scoped(run_dir: Path) -> bool:
    return not (run_dir / "metrics.jsonl").is_file() and bool(stage_dirs(run_dir))


def representative_config_path(run_dir: Path) -> Path | None:
    """The config.yaml to use as the report's single identity snapshot.

    Flat runs: run_dir/config.yaml. Stage-scoped runs: stage0's config.yaml —
    every stage shares the same experiment-level knobs (lr, schedule,
    hidden_loss, ...), differing only in v4_stage/v4_stage_devices/
    owned_blocks (CLAUDE.md Training Runtime section).
    """
    flat = run_dir / "config.yaml"
    if flat.is_file():
        return flat
    dirs = stage_dirs(run_dir)
    if dirs:
        candidate = dirs[0] / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


def _owned_from_contract(rows: list[dict]) -> tuple[int, int] | None:
    for row in rows:
        if row.get("kind") == "pipeline_v4_contract":
            ob = row.get("owned_blocks")
            if ob and len(ob) == 2:
                return int(ob[0]), int(ob[1])
    return None


class RunMetrics:
    """Loaded, stage-merged metrics.jsonl for one run.

    ``rows`` is every row from every stage (or the single flat file), each
    tagged with an injected ``_stage`` (int index, or None for a flat run)
    and ``_owned_blocks`` (that stage's ``(first, last)`` global layer range,
    or None) key. Scalar per-epoch kinds can be scanned directly off
    ``rows`` exactly like a flat run's row list always could — see module
    docstring. Per-layer kinds must go through ``loss_rows()`` /
    ``gradient_rows()`` / ``parameter_delta_rows()`` below, which resolve
    each stage's contribution to its correct GLOBAL 1-based layer position
    before merging across stages.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.stage_scoped = is_stage_scoped(self.run_dir)
        self.stage_dirs = stage_dirs(self.run_dir) if self.stage_scoped else []
        self.rows: list[dict] = []
        self.n_layers = 0

        if self.stage_scoped:
            for stage_idx, stage_dir in enumerate(self.stage_dirs):
                stage_rows = _read_jsonl(stage_dir / "metrics.jsonl")
                owned = _owned_from_contract(stage_rows)
                if owned is not None:
                    self.n_layers = max(self.n_layers, owned[1])
                for row in stage_rows:
                    row["_stage"] = stage_idx
                    row["_owned_blocks"] = owned
                self.rows.extend(stage_rows)
        else:
            flat_path = self.run_dir / "metrics.jsonl"
            flat_rows = _read_jsonl(flat_path) if flat_path.is_file() else []
            owned = _owned_from_contract(flat_rows)
            for row in flat_rows:
                row["_stage"] = None
                row["_owned_blocks"] = owned
            self.rows = flat_rows
            if owned is not None:
                self.n_layers = owned[1]
            else:
                # Legacy runs carry no contract row at all: infer depth from
                # whichever per-layer field is widest.
                for row in flat_rows:
                    for key in ("per_layer", "per_layer_mean",
                                "per_layer_absolute_l2"):
                        values = row.get(key)
                        if isinstance(values, list):
                            self.n_layers = max(self.n_layers, len(values))

    # -- placement helpers -------------------------------------------------

    def _owned_positions(self, n: int,
                         owned: tuple[int, int] | None) -> list[tuple[int, int]]:
        """(local_index, global_layer) pairs this stage AUTHORITATIVELY owns.

        Two list shapes are handled: a LOCAL array of exactly this stage's
        owned span (every entry belongs to this stage — offset by
        ``owned[0]``); and an already-GLOBAL, full-model-depth array that is
        zero-filled outside this stage's own range (current
        ``parameter_delta`` reality, since ``ParameterDeltaTracker`` iterates
        the full ``stack.n_layers`` in every stage process — see
        ``telemetry.py``). In the global case only this stage's own slice is
        authoritative: the zero entries elsewhere are placeholders for
        blocks this stage does not train, not real values, and the stage
        that DOES own them supplies its own (non-zero) row for that slice.
        Returning every stage's full 30-entry array here would make every
        global layer appear once per stage instead of once total.
        No owned-range contract (flat legacy run): every entry belongs to
        the single process, local index i is global layer i + 1 directly.
        """
        if owned is not None:
            start, stop = owned
            span = stop - start + 1
            if n == span:
                return [(i, start + i) for i in range(n)]
            if n == self.n_layers:
                return [(i, i + 1) for i in range(start - 1, stop)]
        return [(i, i + 1) for i in range(n)]

    def _place_dict(self, mapping: dict, owned: tuple[int, int] | None) -> dict[int, float]:
        """Global-layer -> value for a DICT field keyed by layer (as str/int).

        Current v4 telemetry already keys these dicts by the global layer
        number (see module docstring). Handled defensively in case a future
        emission keys by local 0-based index instead.
        """
        try:
            int_keys = {int(k): v for k, v in mapping.items()}
        except (TypeError, ValueError):
            return {}
        if owned is not None:
            start, stop = owned
            if all(start <= k <= stop for k in int_keys):
                return int_keys
            span = stop - start + 1
            if all(0 <= k < span for k in int_keys):
                return {start + k: v for k, v in int_keys.items()}
        return int_keys

    # -- per-layer accessors -------------------------------------------------

    def loss_rows(self) -> list[dict]:
        """[{epoch, layer, loss}], global layer, completed-epoch numbering.

        Legacy ``train``/``per_layer``: epoch is logged 0-based -> +1.
        v4 ``v4_epoch``/``layer_losses``: epoch is already the completed
        count (``online_v4.py`` logs ``epoch=epoch + 1``).
        """
        out = []
        for row in self.rows:
            if row.get("kind") == "train" and row.get("per_layer"):
                epoch = int(row.get("epoch", 0)) + 1
                for i, value in enumerate(row["per_layer"]):
                    out.append({"epoch": epoch, "layer": i + 1,
                                "loss": float(value)})
            elif row.get("kind") == "v4_epoch" and row.get("layer_losses"):
                epoch = int(row.get("epoch", 0))
                placed = self._place_dict(row["layer_losses"], row.get("_owned_blocks"))
                for layer, value in placed.items():
                    out.append({"epoch": epoch, "layer": layer,
                                "loss": float(value)})
        return out

    def gradient_rows(self) -> list[dict]:
        """[{epoch, layer, gradient_l2}], global layer.

        Legacy ``v3_gradient_norm``/``per_layer_mean``: epoch as logged (no
        offset — matches the historical report's behavior, which never
        added one for this kind).  v4 ``v4_gradient_norm``/``grad_norms``:
        epoch already the completed count.
        """
        out = []
        for row in self.rows:
            if row.get("kind") == "v3_gradient_norm":
                epoch = int(row.get("epoch", 0))
                for i, value in enumerate(row.get("per_layer_mean", []) or []):
                    out.append({"epoch": epoch, "layer": i + 1,
                                "gradient_l2": float(value)})
            elif row.get("kind") == "v4_gradient_norm" and row.get("grad_norms"):
                epoch = int(row.get("epoch", 0))
                placed = self._place_dict(row["grad_norms"], row.get("_owned_blocks"))
                for layer, value in placed.items():
                    out.append({"epoch": epoch, "layer": layer,
                                "gradient_l2": float(value)})
        return out

    def parameter_delta_rows(self) -> list[dict]:
        """[{epoch, layer, absolute_l2, relative_l2, parameter_count,
        representation}], global layer, for ``kind=="parameter_delta"``.

        Shared by legacy and v4 pipelines (``telemetry.py``). Each stage
        contributes only the slice of its array it authoritatively owns
        (``_owned_positions``), so merging stages yields exactly one row
        per (epoch, global layer) rather than one per (epoch, layer, stage)
        — the latter would both duplicate keys AND silently prefer
        whichever stage's zero placeholder happened to sort last.
        """
        out = []
        for row in self.rows:
            if row.get("kind") != "parameter_delta":
                continue
            relative = row.get("per_layer_relative_l2") or []
            absolute = row.get("per_layer_absolute_l2") or []
            counts = row.get("per_layer_parameter_count") or []
            epoch = int(row.get("epoch", 0))
            representation = row.get("representation", "unknown")
            for i, layer in self._owned_positions(len(relative), row.get("_owned_blocks")):
                out.append({
                    "epoch": epoch, "layer": layer,
                    "absolute_l2": absolute[i] if i < len(absolute) else None,
                    "relative_l2": relative[i],
                    "parameter_count": counts[i] if i < len(counts) else None,
                    "representation": representation,
                })
        return out

    # -- completion ----------------------------------------------------------

    def has_checkpoint(self) -> bool:
        if self.stage_scoped:
            return bool(self.stage_dirs) and all(
                (d / "checkpoint").exists() for d in self.stage_dirs)
        return (self.run_dir / "checkpoint").exists()
