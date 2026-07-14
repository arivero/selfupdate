"""Time constant-area answer x token tiles through the real forward layer walk.

This is a mechanics probe, not a convergence experiment.  It uses inexpensive
L2-normalized MSE hidden matching, no readout, and still executes every
block's local backward plus a real nonzero-learning-rate optimizer step.
Default geometries trace constant-area 16-, 32-, and 64-cell diagonals.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.utils.env import cap_cpu_threads

cap_cpu_threads()

import torch

from selfupdate.config import load_config
from selfupdate.data.dataset import collate_padded_items, iter_batch_grid_tiles
from selfupdate.train.layerwise import _make_dataset, _summed_batch
from selfupdate.train.losses import HiddenLoss
from selfupdate.train.runtime import OptimizerPlan, TrainingRuntime
from selfupdate.train.validate import validate_knob_schedule


DEFAULT_PAIRS = (
    (1, 64), (2, 32), (4, 16), (8, 8), (16, 4), (32, 2), (64, 1),
    (1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1),
    (1, 16), (2, 8), (4, 4), (8, 2), (16, 1),
)


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _parse_pairs(text: str) -> list[tuple[int, int]]:
    if not text:
        return list(DEFAULT_PAIRS)
    pairs = []
    for field in text.split(","):
        b, k = field.lower().split("x", 1)
        pairs.append((int(b), int(k)))
    if any(b <= 0 or k <= 0 for b, k in pairs):
        raise ValueError("all BxK coordinates must be positive")
    return pairs


def _representative_indices(ds, count: int, min_aligned: int) -> list[int]:
    """Nested, median-length examples so B comparisons share their rows."""
    eligible = [
        i for i, pair in enumerate(ds.pairs)
        if pair.aligned_len >= min_aligned
    ]
    if len(eligible) < count:
        raise ValueError(
            f"need {count} examples with A >= {min_aligned}, found {len(eligible)}")
    eligible.sort(key=lambda i: len(ds.pairs[i].student_ids))
    middle = len(eligible) // 2
    # Alternate around the median, then restore deterministic length order.
    picked = []
    radius = 0
    while len(picked) < count:
        for j in (middle - radius, middle + radius):
            if 0 <= j < len(eligible) and eligible[j] not in picked:
                picked.append(eligible[j])
                if len(picked) == count:
                    break
        radius += 1
    return sorted(picked, key=lambda i: len(ds.pairs[i].student_ids))


def _configure(cfg, *, device: str) -> None:
    cfg.model.device = device
    cfg.train.pipeline_version = 2
    cfg.train.update_granularity = "grid"
    cfg.train.answers_per_update = 8
    cfg.train.tokens_per_answer_update = 8
    cfg.train.update_reduction = "token_mean"
    cfg.train.micro_batch = 8
    cfg.train.grad_accum = 1
    cfg.train.batching = "padded"
    cfg.train.run_class = "control"
    cfg.train.hidden_loss = "l2mse"
    cfg.train.conn_window = 1
    cfg.train.conn_stride = 1
    cfg.train.lr = 1.0e-5
    cfg.train.epochs = 1
    cfg.train.online_teacher = False
    cfg.train.frozen_teacher_copy = False


def benchmark(args) -> dict:
    pairs = _parse_pairs(args.pairs)
    cfg = load_config(args.config, args.experiment)
    _configure(cfg, device=args.device)
    validate_knob_schedule(cfg)

    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    runtime = TrainingRuntime(cfg).load()
    load_seconds = time.perf_counter() - load_started
    stack, tok = runtime.stack, runtime.tokenizer
    cache = runtime.load_cache()
    n_layers = stack.n_layers
    ds = _make_dataset(cfg, cache, tok, list(range(1, n_layers + 1)))
    max_b = max(b for b, _ in pairs)
    max_k = max(k for _, k in pairs)
    indices = _representative_indices(ds, max_b, max_k)
    # Materialize once from the RAM-backed cache; timing below begins after all
    # host target reads and collation.
    items = [ds[i] for i in indices]
    loss_fn = HiddenLoss.from_config(cfg.train, stack)
    plan = OptimizerPlan.build(stack, cfg)

    rows = []
    for B, K in pairs:
        source = collate_padded_items(items[:B])
        tile = iter_batch_grid_tiles(source, K)[0]
        batch = tile.batch
        selected = tile.aligned_token_count
        if selected != B * K:
            raise RuntimeError(
                f"{B}x{K} produced {selected} cells; representative rows are too short")
        cfg.train.answers_per_update = B
        cfg.train.tokens_per_answer_update = K
        cfg.train.micro_batch = B
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        timings = []
        status = "ok"
        error = ""
        try:
            for iteration in range(args.warmup + args.repeats):
                _sync(args.device)
                started = time.perf_counter()
                _summed_batch(cfg, stack, loss_fn, batch, batch.hidden, args.device)
                plan.step()
                _sync(args.device)
                elapsed = time.perf_counter() - started
                if iteration >= args.warmup:
                    timings.append(elapsed)
        except torch.cuda.OutOfMemoryError as exc:
            status = "oom"
            error = str(exc).splitlines()[0]
            stack.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            status = "error"
            error = str(exc).splitlines()[0]
            stack.model.zero_grad(set_to_none=True)
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()

        mean_s = statistics.mean(timings) if timings else None
        median_s = statistics.median(timings) if timings else None
        sequence_tokens = int(batch.lengths.sum())
        row = {
            "B": B,
            "K": K,
            "selected_answer_token_cells": selected,
            "layer_loss_cells": selected * n_layers,
            "sequence_tokens": sequence_tokens,
            "causal_token_layer_cells": sequence_tokens * n_layers,
            "max_sequence_tokens": int(batch.lengths.max()),
            "mean_tile_seconds": mean_s,
            "median_tile_seconds": median_s,
            "tiles_per_second": (1.0 / mean_s if mean_s else None),
            "selected_cells_per_second": (selected / mean_s if mean_s else None),
            "layer_loss_cells_per_second": (
                selected * n_layers / mean_s if mean_s else None),
            "causal_tokens_per_second": (
                sequence_tokens / mean_s if mean_s else None),
            "peak_allocated_gb": (
                torch.cuda.max_memory_allocated() / 2**30
                if torch.device(args.device).type == "cuda" else None),
            "peak_reserved_gb": (
                torch.cuda.max_memory_reserved() / 2**30
                if torch.device(args.device).type == "cuda" else None),
            "status": status,
            "error": error,
        }
        rows.append(row)
        print(
            f"B={B:>2} K={K:>2} status={status} "
            + (f"median={median_s:.4f}s peak={row['peak_reserved_gb']:.2f}GiB"
               if median_s else error),
            flush=True,
        )

    return {
        "model": cfg.model.name,
        "dataset": cfg.data.examples_path,
        "loss": "l2mse (inexpensive nonzero mechanics probe)",
        "final_logit_training": False,
        "learning_rate": cfg.train.lr,
        "layer_order": "forward",
        "causal_context": "full_prefix",
        "n_layers": n_layers,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "load_seconds": load_seconds,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "representative_examples": [items[i].example_id for i in range(len(items))],
        "rows": rows,
    }


def write_outputs(result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows = result["rows"]
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path = out.with_suffix(".md")
    columns = ("B", "K", "status", "median_tile_seconds",
               "selected_cells_per_second", "causal_tokens_per_second",
               "peak_reserved_gb")
    lines = [
        "# Constant-area B × K tile timing", "",
        result.get("table_description", (
            "Rows trace constant-area 16-, 32-, and 64-cell diagonals and walk "
            "all layers forward with full causal prefixes. Loss is "
            "L2-normalized MSE, readout is disabled, and AdamW uses learning "
            "rate 1e-5.")), "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_outputs(paths: list[Path]) -> dict:
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not sources:
        raise ValueError("merge needs at least one JSON result")
    rows = {}
    for source in sources:
        for row in source["rows"]:
            rows[(int(row["B"]), int(row["K"]))] = row
    result = {key: value for key, value in sources[0].items() if key != "rows"}
    result["rows"] = [rows[key] for key in sorted(rows)]
    result["merged_sources"] = [str(path) for path in paths]
    result["table_description"] = (
        "Combined constant-area 16/32/64 diagonals plus the completed "
        "fast-side rectangle B={1,2,4,8} × K={8,16,32,64}. Every tile walks "
        "all layers forward with full causal prefixes. Loss is L2-normalized "
        "MSE, readout is disabled, and AdamW uses learning rate 1e-5."
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", nargs="*", default=[], metavar="RESULT_JSON",
                    help="merge existing benchmark JSON files; do not run a model")
    args = ap.parse_args()
    if args.merge:
        result = merge_outputs([Path(path) for path in args.merge])
    else:
        if not args.experiment:
            ap.error("--experiment is required unless --merge is used")
        result = benchmark(args)
    write_outputs(result, Path(args.out))


if __name__ == "__main__":
    main()
