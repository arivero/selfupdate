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
    row = {
        "source": source,
        "run": run_dir.name,
        "model": model,
        "method": train.get("method"),
        "schedule": train.get("schedule"),
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
        "wikitext_ppl_ratio": retained.get("wikitext_ppl_ratio"),
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


def build(out: Path) -> pd.DataFrame:
    rows = []
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
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/retention_index.csv")
    args = ap.parse_args()
    df = build(Path(args.out))
    n_ret = int(df["has_retention"].sum()) if "has_retention" in df else 0
    n_rec = int(df["has_recite"].sum()) if "has_recite" in df else 0
    print(f"wrote {args.out}: {len(df)} runs across {df['source'].nunique()} checkouts")
    print(f"  with retention.json: {n_ret}   with recite.json: {n_rec}")
    if n_ret:
        cols = ["source", "run", "model", "recall", "arc_acc", "arc_retained"]
        show = df[df["has_retention"]][cols].head(15)
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
