"""Build the one-off 2026-07-10 checkpoint coverage queue.

Every checkpoint gets the same three fixed standard benchmark subsets and
every model family gets an epoch-zero reference. Incomplete corpus-recall
artifacts are re-run with the schema-v2 evaluator.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


import yaml


RUNS = Path("runs")
OUT = Path("scripts/queue_coverage_20260710.tsv")
TASK_ARGS = "--tasks arc_easy arc_challenge hellaswag --limit 100"


def safe_model(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def resources(model: str) -> tuple[int, int, str, int]:
    """need_mb, batch size, flags, n_gpus."""
    if "ALIA-40b" in model:
        return 38000, 4, "--load-4bit", 1
    if "gpt-oss-20b" in model:
        return 42000, 4, "--auto-map", 2
    if "14B" in model:
        return 40000, 6, "", 1
    if "8B" in model or "Mistral-7B" in model or "Llama-3.1-8B" in model:
        return 28000, 8, "", 1
    if "Phi-4-mini" in model or "4B" in model:
        return 19000, 12, "", 1
    if "1.7B" in model:
        return 11000, 16, "", 1
    return 6500, 16, "", 1


def rung(path) -> str | None:
    """'quijote_ch8' from '.../raw_ch8.txt' — rung-level corpus keys, as in
    tasks_report.py: chapter rungs are distinct recall targets."""
    m = re.search(r"ch(\d+)", str(path or "").lower())
    return f"quijote_ch{m.group(1)}" if m else None


def measured_corpus(path) -> str:
    path = str(path or "").lower()
    if "quijote" in path:
        return rung(path) or "quijote_ch1"
    return "machado"


def recall_complete(run: Path, raw: dict, cfg: dict) -> bool:
    data = cfg.get("data") or {}
    examples = str(data.get("examples_path") or "").lower()
    poem = str(data.get("poem_path") or "").lower()
    if "combined" in examples:
        expected = {"machado", rung(examples) or "quijote_ch1"}
    elif "quijote" in examples or "quijote" in poem:
        expected = {rung(examples) or rung(poem) or "quijote_ch1"}
    else:
        expected = {"machado"}
    if raw.get("corpora"):
        measured = {measured_corpus(result.get("poem_path")) or key
                    for key, result in raw["corpora"].items()}
    else:
        measured = {measured_corpus(raw.get("poem_path"))}
    return expected <= measured


def main() -> None:
    rows = []
    runs = []
    for run in sorted(RUNS.iterdir()):
        task_path = run / "eval" / "tasks.json"
        if not (run / "checkpoint").exists() or not task_path.exists():
            continue
        raw = json.loads(task_path.read_text())
        cfg_path = run / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        runs.append((run, raw["model"], cfg_path, cfg))

    representatives = {}
    for run, model, cfg_path, _ in runs:
        representatives.setdefault(model, (run, cfg_path))

    for model, (_, cfg_path) in sorted(representatives.items()):
        need, batch, flags, ngpu = resources(model)
        out = RUNS / "standard_damage" / f"teacher_{safe_model(model)}.json"
        cmd = (f".venv/bin/python compressed/standard_destruction_eval.py "
               f"--experiment {cfg_path} --base --out {out} {TASK_ARGS} "
               f"--batch-size {batch} --stage-to-local {flags}").strip()
        rows.append((out, need, "-", cmd, ngpu))

    for run, model, cfg_path, cfg in runs:
        need, batch, flags, ngpu = resources(model)
        base = RUNS / "standard_damage" / f"teacher_{safe_model(model)}.json"
        out = RUNS / "standard_damage" / f"{run.name}.json"
        cmd = (f".venv/bin/python compressed/standard_destruction_eval.py "
               f"--experiment {cfg_path} --checkpoint {run / 'checkpoint'} "
               f"--out {out} {TASK_ARGS} --batch-size {batch} "
               f"--stage-to-local {flags}").strip()
        rows.append((out, need, base, cmd, ngpu))

        task_raw = json.loads((run / "eval" / "tasks.json").read_text())
        if not recall_complete(run, task_raw, cfg):
            marker = run / "eval" / "tasks_corpus_v2.done"
            recall_cmd = (f".venv/bin/python compressed/evaluate.py "
                          f"--checkpoint {run / 'checkpoint'} "
                          f"--out {run / 'eval'} {flags} && touch {marker}")
            rows.append((marker, need, "-", recall_cmd, ngpu))

    lines = ["# done_file\tneed_mb\tafter\tcommand\tn_gpus"]
    lines += ["\t".join(map(str, row)) for row in rows]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {len(representatives)} bases, {len(runs)} checkpoints, "
          f"{sum(str(r[0]).endswith('tasks_corpus_v2.done') for r in rows)} recall repairs")


if __name__ == "__main__":
    main()
