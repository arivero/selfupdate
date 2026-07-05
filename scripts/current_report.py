#!/usr/bin/env python3
"""Write a compact live report for active KD runs.

The long PDF report is final-artifact oriented. This report reads metrics.jsonl
directly so active trainings show recall, general-NLL drift, memory, and layer
localization before final eval artifacts exist.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import yaml


RUNS = Path("runs")
ACTIVE_RUNS = [
    "kd_lora_kl_hi_e40_v3_qwen36_27b_rag",
    "kd_lora_kl_hi_e40_v3_qwen3_30ba3b_inst2507_rag",
    "kd_lora_kl_hi_e60_v3_14b_rag",
]


def load_metrics(run: str) -> list[dict]:
    path = RUNS / run / "metrics.jsonl"
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def model_name(run: str) -> str:
    path = RUNS / run / "config.yaml"
    if not path.exists():
        return ""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cfg.get("model", {}).get("name", "")


def top_layers(run: str, limit: int = 5) -> tuple[int | None, str]:
    path = RUNS / run / "eval" / "lora_layer_deltas_by_epoch.csv"
    if not path.exists():
        return None, ""
    by_epoch: dict[int, list[tuple[float, int]]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                epoch = int(float(row["epoch"]))
                layer = int(float(row["layer"]))
                score = float(row["adapter_update_rms"])
            except (KeyError, ValueError):
                continue
            by_epoch.setdefault(epoch, []).append((score, layer))
    if not by_epoch:
        return None, ""
    epoch = max(by_epoch)
    tops = sorted(by_epoch[epoch], reverse=True)[:limit]
    return epoch, ", ".join(f"L{layer}:{score:.3g}" for score, layer in tops)


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def row_for(run: str) -> dict:
    metrics = load_metrics(run)
    trains = [m for m in metrics if m.get("kind") == "train"]
    evals = [m for m in metrics if m.get("kind") == "eval"]
    done = [m for m in metrics if m.get("kind") == "done"]
    setup = next((m for m in metrics if m.get("kind") == "setup"), {})
    layer_epoch, layers = top_layers(run)

    last_eval = evals[-1] if evals else {}
    first_eval = evals[0] if evals else {}
    best_eval = min(evals, key=lambda e: e.get("cer", float("inf"))) if evals else {}
    gen_last = last_eval.get("gen_ce")
    gen_first = first_eval.get("gen_ce")
    gen_drift = gen_last - gen_first if gen_last is not None and gen_first is not None else None
    losses = [m["loss"] for m in trains[-200:] if "loss" in m]

    trainable = setup.get("trainable_params")
    total = setup.get("total_params")
    trainable_pct = 100 * trainable / total if trainable and total else None

    return {
        "run": run,
        "model": model_name(run),
        "status": "done" if done else "training" if trains else "queued",
        "epoch": trains[-1].get("epoch") if trains else None,
        "items": len(trains) if trains else None,
        "last_loss_mean200": statistics.mean(losses) if losses else None,
        "last_cer": last_eval.get("cer"),
        "last_exact": last_eval.get("line_exact"),
        "best_cer": best_eval.get("cer"),
        "best_epoch": best_eval.get("epoch"),
        "gen_nll_first": gen_first,
        "gen_nll_last": gen_last,
        "gen_nll_drift": gen_drift,
        "vram_gb": last_eval.get("vram_gb") or (done[-1].get("vram_gb") if done else None),
        "trainable_pct": trainable_pct,
        "layer_epoch": layer_epoch,
        "top_layers": layers,
    }


def write_report() -> Path:
    rows = [row_for(run) for run in ACTIVE_RUNS]
    out = RUNS / "current_report.md"
    lines = [
        "# Current KD Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "Training objective is teacher-logit KL only. Recall is in-training recitation eval. "
        "Forgetting is tracked here as general-text NLL (`gen_nll`) and drift from the first "
        "probe of the same run; lower is better. This is not yet a clean pretrained-baseline "
        "delta for Qwen30/Qwen3.6 because their base general probes have not been run.",
        "",
        "| run | model | status | epoch | items | mean KL last200 | last CER | best CER @ epoch | line exact | gen NLL first | gen NLL last | drift | VRAM GiB | trainable % | top modified layers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {run} | {model} | {status} | {epoch} | {items} | {loss} | "
            "{last_cer} | {best_cer} @ {best_epoch} | {exact} | {g0} | {g1} | "
            "{drift} | {vram} | {tpct} | e{layer_epoch}: {layers} |".format(
                run=r["run"],
                model=r["model"],
                status=r["status"],
                epoch=fmt(r["epoch"], 0),
                items=fmt(r["items"], 0),
                loss=fmt(r["last_loss_mean200"], 4),
                last_cer=fmt(r["last_cer"], 4),
                best_cer=fmt(r["best_cer"], 4),
                best_epoch=fmt(r["best_epoch"], 0),
                exact=fmt(r["last_exact"], 4),
                g0=fmt(r["gen_nll_first"], 4),
                g1=fmt(r["gen_nll_last"], 4),
                drift=fmt(r["gen_nll_drift"], 4),
                vram=fmt(r["vram_gb"], 2),
                tpct=fmt(r["trainable_pct"], 3),
                layer_epoch=fmt(r["layer_epoch"], 0),
                layers=r["top_layers"] or "—",
            )
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Qwen30-A3B has reached perfect probe recitation several times; its latest "
        "available forgetting drift is modest in absolute terms but positive.",
        "- Qwen3.6-27B is early; its recall improved sharply by epoch 3 but has not "
        "stabilized yet.",
        "- Qwen3-14B finished 60 epochs; recall is strong but noisier than Qwen30-A3B, "
        "and its general-NLL drift is larger.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    print(f"wrote {write_report()}")


if __name__ == "__main__":
    main()
