"""Validate runs/conclusions.yaml against runs/corpus.csv.

Every claim must name runs that exist, are not confounded, and carry a
full-corpus eval; a claim whose evidence degrades (run reclassified,
eval missing) fails loudly here instead of surviving as stale narrative
(recommendations.md "Conclusion Ledger").

Usage:
    python compressed/conclusion_check.py [--ledger runs/conclusions.yaml]
        [--corpus runs/corpus.csv]

Exit nonzero on any ERROR; WARNs are advisory.
"""

import argparse
import csv
import sys
from pathlib import Path


import yaml

STATUSES = {"proven", "replicated", "single_seed", "confounded", "open",
            "retracted", "teacher_reference_only", "archived"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="runs/conclusions.yaml")
    ap.add_argument("--corpus", default="runs/corpus.csv")
    args = ap.parse_args()

    ledger = yaml.safe_load(Path(args.ledger).read_text())
    corpus = {r["run"]: r for r in csv.DictReader(Path(args.corpus).open())}

    errors, warns = [], []
    seen_ids = set()
    for entry in ledger:
        cid = entry.get("id", "<missing id>")
        if cid in seen_ids:
            errors.append(f"{cid}: duplicate id")
        seen_ids.add(cid)
        if entry.get("status") not in STATUSES:
            errors.append(f"{cid}: bad status {entry.get('status')!r}")
        if not entry.get("claim"):
            errors.append(f"{cid}: empty claim")
        runs = entry.get("required_runs") or []
        if not runs and entry.get("status") in ("proven", "replicated"):
            warns.append(f"{cid}: status {entry['status']} with no required_runs")
        for run in runs:
            row = corpus.get(run)
            if row is None:
                errors.append(f"{cid}: required run {run} not in corpus")
                continue
            if row.get("run_class") == "confounded":
                errors.append(f"{cid}: required run {run} is CONFOUNDED")
            cer = (row.get("full_eval_cer") or "").strip()
            if not cer:
                warns.append(f"{cid}: {run} lacks full-corpus eval")
            elif float(cer) == 0.0:
                warns.append(f"{cid}: {run} reports full_eval_cer == 0.0 "
                             "exactly — verify it is not an artifact")
        if entry.get("status") in ("proven", "replicated") and \
                entry.get("blocking_gaps"):
            warns.append(f"{cid}: {entry['status']} but has blocking_gaps — "
                         "downgrade or resolve")

    for w in warns:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print(f"{len(ledger)} claims checked: {len(errors)} errors, {len(warns)} warnings")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
