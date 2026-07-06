"""Build a machine-readable corpus index for run artifacts.

The index is intentionally conservative: it records old keys and evidence
warnings instead of trying to normalize them away. Reports and conclusion
ledgers should filter on ``evidence_status == method_clean`` for method claims.

Usage:
    python scripts/build_corpus_index.py --out runs/corpus.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
CONFIGS = ROOT / "configs/experiments"

OLD_KEYS = {
    "tail_ce_blocks", "tail_ce_weight", "tail_ce_kind", "tail_hidden_weight",
    "last_block_ce_weight", "lens_ce_weight", "lens_ce_from", "answer_ce_weight",
    "last_block_" + "task" + "_label_weight",
    "lens_" + "task" + "_label_weight",
    "anchor_" + "ce_weight", "lens_" + "from_layer",
}
FORBIDDEN_REFERENCE_SOURCE = "task" + "_label"

FIELDS = [
    "run", "run_class", "evidence_status", "warnings", "active_config",
    "model", "schedule", "hidden_loss", "lora", "online_teacher",
    "frozen_teacher_copy", "examples_path", "mask_mode", "compaction",
    "readout_source", "readout_window", "readout_weight",
    "window_hidden_weight", "conn_window", "conn_stride", "legacy_keys",
    "epochs", "lr", "seed", "items_seen", "train_logs", "loss_first",
    "loss_final", "last_eval_cer", "last_eval_line_exact", "full_eval_cer",
    "full_eval_line_exact", "general_ce", "forgetting_delta_ce",
    "epoch0_cer", "epoch0_general_ce", "epoch0_source",
    "last_epoch", "last_epoch_cer", "last_epoch_ce", "last_epoch_forgetting_ce",
    "best_epoch", "best_epoch_cer", "best_epoch_ce", "best_epoch_forgetting_ce",
    "final_forgetting_ce",
    "destruction_json", "signal_attribution_json", "hidden_share",
    "vram_gb", "vram_reserved_gb", "train_min",
]


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception as e:  # noqa: BLE001
        return {"_parse_error": f"{type(e).__name__}: {e}"}


def _status(train: dict) -> tuple[str, list[str]]:
    warnings: list[str] = []
    old = sorted(k for k in OLD_KEYS if k in train)
    if old:
        warnings.append("legacy_keys=" + ",".join(old))
    source = train.get("readout_source", train.get("tail_ce_kind", "UNSET"))
    if source == FORBIDDEN_REFERENCE_SOURCE:
        warnings.append("forbidden_reference_text_training_signal")
    blocks = train.get("readout_window_blocks", train.get("tail_ce_blocks", 0)) or 0
    if blocks:
        if source == "UNSET":
            warnings.append("readout_source_unset")
        if train.get("conn_window", 0) != blocks or train.get("conn_stride", 0) != 1:
            warnings.append("not_sanctioned_sliding_window")
    run_class = train.get("run_class", "method")
    if run_class != "method":
        return run_class, warnings
    if warnings:
        return "confounded", warnings
    return "method_clean", warnings


def _active_config_map() -> dict[str, str]:
    out = {}
    for path in sorted(CONFIGS.glob("*.yaml")):
        cfg = _read_yaml(path)
        run = cfg.get("run_name")
        if run:
            out[run] = str(path.relative_to(ROOT))
    return out


def _model_label(model: object) -> str:
    m = str(model or "")
    if "Qwen3-0.6B" in m:
        return "Qwen3-0.6B"
    if "Qwen3-1.7B" in m:
        return "Qwen3-1.7B"
    if "Qwen3-4B" in m:
        return "Qwen3-4B"
    if "Qwen3-8B" in m:
        return "Qwen3-8B"
    if "Qwen3-14B" in m:
        return "Qwen3-14B"
    if "Qwen3.6-27B" in m:
        return "Qwen3.6-27B"
    if "gemma-4-26B-A4B" in m or "Gemma-4-26B-A4B" in m:
        return "Gemma-4-26B-A4B"
    if "gemma-4-31B" in m or "Gemma-4-31B" in m:
        return "Gemma-4-31B"
    if "Mistral-7B" in m:
        return "Mistral-7B"
    if "gpt-oss-20b" in m or "gpt-oss-20B" in m:
        return "gpt-oss-20B"
    if "Llama-3.1-8B" in m:
        return "Llama-8B"
    if "Phi-4-mini" in m:
        return "Phi-4-mini"
    return m.rsplit("/", 1)[-1] if m else "unknown"


def _teacher_reference_index(runs: Path) -> dict[str, dict]:
    """Native/no-RAG epoch-zero references keyed by normalized model label."""
    refs: dict[str, dict] = {}
    candidates = [runs / "base-eval-full/recite.json"]
    candidates += sorted(runs.glob("baseline_native_*/recite.json"))
    candidates += sorted(runs.glob("teacher_ref_native_*/recite.json"))
    for path in candidates:
        ref = _read_json(path)
        if not ref:
            continue
        model = ref.get("model") or "Qwen/Qwen3-0.6B"
        label = _model_label(model)
        source = path.parent.name
        if label in refs and not source.startswith("teacher_ref_native_"):
            continue
        refs[label] = {
            "cer": ref.get("cer"),
            "general_ce": (ref.get("general") or {}).get("mean_ce"),
            "source": source,
        }
    return refs


def _forget(ce, epoch0_ce):
    if ce is None or epoch0_ce is None:
        return None
    return round(float(ce) - float(epoch0_ce), 6)


def _metric_summary(run_dir: Path) -> dict:
    p = run_dir / "metrics.jsonl"
    first = []
    last_reversed = []
    items_seen = 0
    last_eval = None
    evals = []
    done = None
    if not p.exists():
        return {
            "first_losses": first, "last_losses": [], "train_logs": "",
            "items_seen": 0, "last_eval": None, "last_epoch_eval": None,
            "best_epoch_eval": None, "done": None,
        }
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = m.get("kind")
            if kind == "eval" and "epoch" in m:
                evals.append(m)
            if kind == "train" and "loss" in m:
                if len(first) < 20:
                    first.append(m["loss"])
            elif kind == "stage" and "loss" in m and not first:
                if len(first) < 20:
                    first.append(m["loss"])
    tail = _tail_lines(p)
    for line in reversed(tail):
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = m.get("kind")
        if kind == "train" and "loss" in m:
            if len(last_reversed) < 20:
                last_reversed.append(m["loss"])
            items_seen = max(items_seen, m.get("items_seen", 0))
        elif kind == "stage" and "loss" in m and not last_reversed:
            if len(last_reversed) < 20:
                last_reversed.append(m["loss"])
            items_seen = max(items_seen, m.get("steps", 0))
        elif kind == "eval" and last_eval is None:
            last_eval = m
        elif kind == "done" and done is None:
            done = m
        if len(last_reversed) >= 20 and last_eval is not None and done is not None:
            break
    last_epoch_eval = evals[-1] if evals else None
    best_epoch_eval = min(evals, key=lambda m: m.get("cer", float("inf"))) if evals else None
    return {
        "first_losses": first,
        "last_losses": list(reversed(last_reversed)),
        "train_logs": "",
        "items_seen": items_seen,
        "last_eval": last_eval,
        "last_epoch_eval": last_epoch_eval,
        "best_epoch_eval": best_epoch_eval,
        "done": done,
    }


def _tail_lines(path: Path, max_bytes: int = 1 << 20) -> list[str]:
    size = path.stat().st_size
    with path.open("rb") as f:
        start = max(0, size - max_bytes)
        f.seek(start)
        data = f.read().decode("utf-8", errors="ignore")
    lines = data.splitlines()
    if start > 0 and lines:
        lines = lines[1:]  # first line may be partial
    return lines


def build_rows(runs: Path = RUNS) -> list[dict]:
    active = _active_config_map()
    rows: list[dict] = []
    epoch0_refs = _teacher_reference_index(runs)

    for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            continue
        cfg = _read_yaml(cfg_path)
        if cfg.get("_parse_error"):
            train = {}
            warnings = [cfg["_parse_error"]]
            status = "unreadable"
        else:
            train = cfg.get("train", {}) or {}
            status, warnings = _status(train)
        data = cfg.get("data", {}) or {}
        mask = cfg.get("mask", {}) or {}
        model = cfg.get("model", {}) or {}
        epoch0 = epoch0_refs.get(_model_label(model.get("name", "")), {})
        epoch0_ce = epoch0.get("general_ce")
        metrics = _metric_summary(run_dir)
        full = _read_json(run_dir / "eval/recite.json")
        signal = _read_json(run_dir / "eval/signal_attribution.json")
        destruction = run_dir / "eval/destruction.json"
        general_ce = full.get("general", {}).get("mean_ce") if full else None
        forget = _forget(general_ce, epoch0_ce)
        loss_first = (round(sum(metrics["first_losses"]) / len(metrics["first_losses"]), 6)
                      if metrics["first_losses"] else None)
        loss_final = (round(sum(metrics["last_losses"]) / len(metrics["last_losses"]), 6)
                      if metrics["last_losses"] else None)
        last_eval = metrics["last_eval"]
        last_epoch_eval = metrics["last_epoch_eval"]
        best_epoch_eval = metrics["best_epoch_eval"]
        done = metrics["done"]
        row = {
            "run": run_dir.name,
            "run_class": train.get("run_class", "method"),
            "evidence_status": status,
            "warnings": ";".join(warnings),
            "active_config": active.get(run_dir.name, ""),
            "model": model.get("name", ""),
            "schedule": train.get("schedule", ""),
            "hidden_loss": train.get("hidden_loss", ""),
            "lora": train.get("lora", {}).get("enabled", ""),
            "online_teacher": train.get("online_teacher", ""),
            "frozen_teacher_copy": train.get("frozen_teacher_copy", ""),
            "examples_path": data.get("examples_path", ""),
            "mask_mode": mask.get("mode", ""),
            "compaction": mask.get("compaction", ""),
            "readout_source": train.get("readout_source", train.get("tail_ce_kind", "UNSET")),
            "readout_window": train.get("readout_window_blocks", train.get("tail_ce_blocks", 0)),
            "readout_weight": train.get("readout_weight", train.get("tail_ce_weight", 0.0)),
            "window_hidden_weight": train.get("window_hidden_weight", train.get("tail_hidden_weight", 1.0)),
            "conn_window": train.get("conn_window", 0),
            "conn_stride": train.get("conn_stride", 0),
            "legacy_keys": ",".join(k for k in sorted(OLD_KEYS) if k in train),
            "epochs": train.get("epochs", ""),
            "lr": train.get("lr", ""),
            "seed": train.get("seed", ""),
            "items_seen": metrics["items_seen"],
            "train_logs": metrics["train_logs"],
            "loss_first": loss_first,
            "loss_final": loss_final,
            "last_eval_cer": last_eval.get("cer") if last_eval else None,
            "last_eval_line_exact": last_eval.get("line_exact") if last_eval else None,
            "full_eval_cer": full.get("cer") if full else None,
            "full_eval_line_exact": full.get("line_exact") if full else None,
            "general_ce": general_ce,
            "forgetting_delta_ce": forget,
            "epoch0_cer": epoch0.get("cer"),
            "epoch0_general_ce": epoch0_ce,
            "epoch0_source": epoch0.get("source"),
            "last_epoch": last_epoch_eval.get("epoch") if last_epoch_eval else None,
            "last_epoch_cer": last_epoch_eval.get("cer") if last_epoch_eval else None,
            "last_epoch_ce": last_epoch_eval.get("gen_ce") if last_epoch_eval else None,
            "last_epoch_forgetting_ce": _forget(
                last_epoch_eval.get("gen_ce") if last_epoch_eval else None, epoch0_ce),
            "best_epoch": best_epoch_eval.get("epoch") if best_epoch_eval else None,
            "best_epoch_cer": best_epoch_eval.get("cer") if best_epoch_eval else None,
            "best_epoch_ce": best_epoch_eval.get("gen_ce") if best_epoch_eval else None,
            "best_epoch_forgetting_ce": _forget(
                best_epoch_eval.get("gen_ce") if best_epoch_eval else None, epoch0_ce),
            "final_forgetting_ce": forget,
            "destruction_json": destruction.exists(),
            "signal_attribution_json": signal is not None,
            "hidden_share": signal.get("hidden_share") if signal else None,
            "vram_gb": done.get("vram_gb") if done else None,
            "vram_reserved_gb": done.get("vram_reserved_gb") if done else None,
            "train_min": (last_eval.get("minutes") if last_eval else None)
            or (done.get("minutes") if done else None),
        }
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=RUNS / "corpus.csv")
    args = ap.parse_args()

    rows = build_rows(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
