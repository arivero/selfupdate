"""Aggregate recall + retention evidence across the three checkouts.

The final report compares the layerwise-forward-distillation method (this
checkout) against the classical-KD baselines (../selfupdate_kd) and the
multigpu big-model layerwise runs (../selfupdate_multi).  This script scans all
three `runs/` trees and emits one tidy table, runs/retention_index.csv, that
both the trajectory plots and cross_report.py consume.

Axes of interest (all bounded/stable, per the 2026-07-07 metric switch):
  recall     : memorization of the reference corpus.
               recall_cer  = recite.json CER (lower better)
               recall      = 1 - min(1, recall_cer)          (higher better)
               plus exact-match recall probes from retention.json
  retention  : capability preserved vs the epoch-0 teacher of the same base.
               arc_acc     = ARC-Easy accuracy (higher better)
               arc_retained= arc_acc / teacher arc_acc
               wikitext_ppl_ratio (lower better; ~1 == undamaged)

Usage: python scripts/retention_index.py [--out runs/retention_index.csv]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent

# checkout dir -> (source label, method family shown in the report)
CHECKOUTS = {
    REPO: "layerwise",
    REPO.parent / "selfupdate_kd": "classical-kd",
    REPO.parent / "selfupdate_multi": "multigpu",
}


def _model_label(model: str | None) -> str:
    if not model:
        return "unknown"
    return (model.replace("Qwen/", "")
                 .replace("openai/", "")
                 .replace("BSC-LT/", "")
                 .replace("google/", ""))


def _corpus_family(run: str, cfg: dict) -> str:
    data = ((cfg.get("data") or {}).get("examples") or "").lower()
    name = run.lower()
    text = f"{name} {data}"
    if "combined" in text or "v4_ch1" in text:
        return "Machado+Quijote"
    if "q_ch" in text or "quijote" in text or "cervantes" in text or "ch1" in text:
        return "Quijote"
    return "Machado"


def _loss_kind(run: str, train: dict) -> str:
    loss = str(train.get("hidden_loss") or "").lower()
    name = run.lower()
    if train.get("method") == "kd":
        return "teacher_kl"
    for key in ("lens_kl", "vocab_mse", "l2mse", "nmse", "huber", "cosine"):
        if key in loss or key in name:
            return key
    if "kl" in name:
        return "teacher_kl"
    return loss or "unknown"


def _window_kind(run: str, train: dict) -> str:
    name = run.lower()
    blocks = (train.get("readout_window_blocks") or train.get("tail_ce_blocks")
              or train.get("conn_window") or 0)
    try:
        blocks = int(blocks)
    except (TypeError, ValueError):
        blocks = 0
    if blocks > 0:
        return f"k{blocks}"
    if "k1local" in name or "strict" in name:
        return "strict/local"
    for k in (8, 6, 4, 3, 2):
        if f"slide{k}" in name or f"_k{k}" in name:
            return f"k{k}"
    return "full/top" if train.get("method") == "kd" else "strict/local"


def _lens_kind(loss_kind: str, train: dict) -> str:
    if train.get("method") == "kd":
        return "logit_kd"
    if loss_kind == "lens_kl":
        return "lens_kl"
    if loss_kind == "vocab_mse":
        return "frozen_vocab"
    if loss_kind in {"nmse", "l2mse", "huber", "cosine"}:
        return "hidden_match"
    return loss_kind


def _read_json(p: Path):
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


def _cfg(run_dir: Path) -> dict:
    p = run_dir / "config.yaml"
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:  # noqa: BLE001
        return {}


def _row(run_dir: Path, source: str) -> dict | None:
    cfg = _cfg(run_dir)
    if not cfg:
        return None
    train = cfg.get("train", {}) or {}
    model = (cfg.get("model") or {}).get("name")
    recite = _read_json(run_dir / "eval" / "recite.json") or {}
    ret = _read_json(run_dir / "eval" / "retention.json") or {}
    tasks = ret.get("tasks", {}) or {}
    retained = ret.get("retained", {}) or {}

    cer = recite.get("cer")
    loss_kind = _loss_kind(run_dir.name, train)
    ppl_ratio = retained.get("wikitext_ppl_ratio")
    row = {
        "source": source,
        "run": run_dir.name,
        "checkpoint_kind": "trained",
        "model": model,
        "model_label": _model_label(model),
        "corpus_family": _corpus_family(run_dir.name, cfg),
        "method": train.get("method"),
        "schedule": train.get("schedule"),
        "loss_kind": loss_kind,
        "window_kind": _window_kind(run_dir.name, train),
        "lens_kind": _lens_kind(loss_kind, train),
        "is_lora": bool(ret.get("is_lora")) if "is_lora" in ret else None,
        "mode": train.get("mode"),
        # recall
        "recall_cer": cer,
        "recall": (1.0 - min(1.0, cer)) if isinstance(cer, (int, float)) else None,
        "recite_n": recite.get("n"),
        # retention (destruction)
        "arc_acc": tasks.get("arc_easy", {}).get("accuracy"),
        "arc_retained": retained.get("arc_retained"),
        "wikitext_ppl": tasks.get("wikitext2_ppl", {}).get("ppl"),
        "wikitext_ppl_ratio": ppl_ratio,
        "wikitext_log_ppl_ratio": math.log(ppl_ratio) if isinstance(ppl_ratio, (int, float)) and ppl_ratio > 0 else None,
        # exact-match recall probes
        "recall_cont_machado": tasks.get("recall_cont_machado", {}).get("exact_acc"),
        "recall_cloze_machado": tasks.get("recall_cloze_machado", {}).get("exact_acc"),
        "recall_cont_cervantes": tasks.get("recall_cont_cervantes", {}).get("exact_acc"),
        "recall_censor_cervantes": tasks.get("recall_censor_cervantes", {}).get("exact_acc"),
        # teacher (base) references for the same base model
        "teacher_arc_acc": (ret.get("teacher") or {}).get("arc_easy", {}).get("accuracy")
        if ret.get("teacher") else None,
        "has_retention": bool(tasks),
        "has_recite": bool(recite),
        "run_dir": str(run_dir),
    }
    return row


def _teacher_row(row: dict, ret: dict) -> dict | None:
    teacher = ret.get("teacher") or {}
    if not teacher:
        return None
    tasks = {
        "arc_easy": teacher.get("arc_easy", {}),
        "wikitext2_ppl": teacher.get("wikitext2_ppl", {}),
        "recall_cont_machado": teacher.get("recall_cont_machado", {}),
        "recall_cloze_machado": teacher.get("recall_cloze_machado", {}),
        "recall_cont_cervantes": teacher.get("recall_cont_cervantes", {}),
        "recall_censor_cervantes": teacher.get("recall_censor_cervantes", {}),
    }
    return {
        **row,
        "run": f"epoch0::{row['model_label']}::{row['corpus_family']}",
        "checkpoint_kind": "epoch0",
        "method": "epoch0",
        "schedule": "epoch0",
        "loss_kind": "epoch0",
        "window_kind": "epoch0",
        "lens_kind": "epoch0",
        "is_lora": False,
        "mode": None,
        "recall_cer": None,
        "recall": None,
        "recite_n": None,
        "arc_acc": tasks["arc_easy"].get("accuracy"),
        "arc_retained": 1.0,
        "wikitext_ppl": tasks["wikitext2_ppl"].get("ppl"),
        "wikitext_ppl_ratio": 1.0,
        "wikitext_log_ppl_ratio": 0.0,
        "recall_cont_machado": tasks["recall_cont_machado"].get("exact_acc"),
        "recall_cloze_machado": tasks["recall_cloze_machado"].get("exact_acc"),
        "recall_cont_cervantes": tasks["recall_cont_cervantes"].get("exact_acc"),
        "recall_censor_cervantes": tasks["recall_censor_cervantes"].get("exact_acc"),
        "teacher_arc_acc": tasks["arc_easy"].get("accuracy"),
        "has_retention": True,
        "has_recite": False,
        "run_dir": "",
    }


def build(out: Path) -> pd.DataFrame:
    rows = []
    teacher_rows = {}
    for root, source in CHECKOUTS.items():
        runs = root / "runs"
        if not runs.exists():
            continue
        for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
            if not (run_dir / "config.yaml").exists():
                continue
            r = _row(run_dir, source)
            if r:
                rows.append(r)
                ret = _read_json(run_dir / "eval" / "retention.json") or {}
                trow = _teacher_row(r, ret)
                if trow:
                    teacher_rows[(trow["source"], trow["model"], trow["corpus_family"])] = trow
    rows.extend(teacher_rows.values())
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/retention_index.csv")
    args = ap.parse_args()
    df = build(Path(args.out))
    trained = df[df["checkpoint_kind"] != "epoch0"] if "checkpoint_kind" in df else df
    n_ret = int(trained["has_retention"].sum()) if "has_retention" in trained else 0
    n_rec = int(trained["has_recite"].sum()) if "has_recite" in trained else 0
    n_epoch0 = int((df["checkpoint_kind"] == "epoch0").sum()) if "checkpoint_kind" in df else 0
    print(f"wrote {args.out}: {len(trained)} trained runs + {n_epoch0} epoch-0 baselines across {df['source'].nunique()} checkouts")
    print(f"  trained with retention.json: {n_ret}   with recite.json: {n_rec}")
    if n_ret:
        cols = ["source", "run", "model", "recall", "arc_acc", "arc_retained"]
        show = df[df["has_retention"]][cols].head(15)
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
