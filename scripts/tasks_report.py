"""Corpus-aware recall and standard-capability report.

Machado and Quijote are different recall targets.  Base references are keyed
by (model, corpus), never by model alone.  Model damage comes from fixed
subsets of standard benchmarks in eval/destruction.json.

    python scripts/tasks_report.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

# Quijote chapter rungs (ch1/ch4/ch8/ch16) are DIFFERENT recall targets with
# different base references; conflating them clobbered every quijote epoch-0
# column with whichever base-tasks dir sorted last (review 2026-07-10).
CORPUS_PATHS = {
    "machado": "data/poem/raw.txt",
    "quijote_ch1": "data/quijote/raw_ch1.txt",
    "quijote_ch4": "data/quijote/raw_ch4.txt",
    "quijote_ch8": "data/quijote/raw_ch8.txt",
    "quijote_ch16": "data/quijote/raw_ch16.txt",
}
CORPUS_LABELS = {
    "machado": "Machado",
    "quijote_ch1": "Quijote ch1",
    "quijote_ch4": "Quijote ch4",
    "quijote_ch8": "Quijote ch8",
    "quijote_ch16": "Quijote ch16",
}


def quijote_rung(path: str | None) -> str | None:
    """'quijote_ch8' from '.../raw_ch8.txt' or '.../examples_ch8.jsonl'."""
    m = re.search(r"ch(\d+)", str(path or "").lower())
    return f"quijote_ch{m.group(1)}" if m else None


def corpus_name(path: str | None) -> str | None:
    path = str(path or "").lower()
    if "quijote" in path:
        return quijote_rung(path) or "quijote_ch1"
    if "poem" in path or path.endswith("raw.txt"):
        return "machado"
    return None


def training_scope(config: dict, measured: set[str]) -> tuple[str, ...]:
    data = config.get("data") or {}
    examples = str(data.get("examples_path") or "").lower()
    poem = str(data.get("poem_path") or "").lower()
    if "combined" in examples:
        return ("machado", quijote_rung(examples) or "quijote_ch1")
    if "quijote" in examples or "quijote" in poem:
        return (quijote_rung(examples) or quijote_rung(poem) or "quijote_ch1",)
    if examples or poem:
        return ("machado",)
    return tuple(sorted(measured)) or ("machado",)


def recall_results(raw: dict) -> dict[str, dict]:
    """Read both schema v2 and historical one-corpus task artifacts.

    Corpora are rekeyed by the poem_path each result was MEASURED on, so a
    historical v2 artifact keyed 'quijote' (evaluate.py once hardcoded
    raw_ch1.txt) lands on 'quijote_ch1' — what was actually scored.
    """
    if raw.get("corpora"):
        return {corpus_name(result.get("poem_path")) or key: result
                for key, result in raw["corpora"].items()}
    corpus = corpus_name(raw.get("poem_path"))
    return {corpus: raw} if corpus else {}


def standard_bases(runs_dir: Path) -> dict[str, dict]:
    """Choose the richest available epoch-zero standard-suite reference."""
    bases = {}
    for path in sorted((runs_dir / "standard_damage").glob("teacher_*.json")):
        if not path.is_file():
            continue
        raw = json.loads(path.read_text())
        bench = {name: result for name, result in (raw.get("tasks") or {}).items()
                 if result.get("accuracy") is not None}
        if raw.get("model") and bench:
            bases[raw["model"]] = bench
    for path in sorted((runs_dir / "destruction").glob("*/destruction.json")):
        if not path.is_file():
            continue
        raw = json.loads(path.read_text())
        model, bench = raw.get("model"), raw.get("benchmarks") or {}
        # "richest available" literally: a 5-bench destruction base beats a
        # 3-task teacher file (the old first-wins order preferred whichever
        # loop ran first, which inverted the docstring for models with both).
        if model and len(bench) > len(bases.get(model, {})):
            bases[model] = bench
    return bases


def damage_result(run_dir: Path, model: str, bases: dict[str, dict]) -> dict:
    standard_path = (run_dir.parent / "standard_damage" /
                     f"{run_dir.name}.json")
    path = standard_path if standard_path.exists() else run_dir / "eval" / "destruction.json"
    base = bases.get(model) or {}
    base_scores = [v.get("accuracy") for v in base.values()
                   if v.get("accuracy") is not None]
    base_summary = {
        "base_accuracy": (sum(base_scores) / len(base_scores)
                          if base_scores else None),
        "base_n_tasks": len(base_scores),
    }
    if not path.exists():
        return base_summary
    raw = json.loads(path.read_text())
    bench = (raw.get("benchmarks") or
             {name: result for name, result in (raw.get("tasks") or {}).items()
              if result.get("accuracy") is not None})
    common = sorted(set(bench) & set(base))
    # Only paired tasks belong in the delta. Historical artifacts contain
    # different suite widths; averaging unmatched tasks would introduce a
    # second, subtler version of the corpus-conflation bug.
    pairs = [(name, bench[name].get("accuracy"), base[name].get("accuracy"))
             for name in common
             if bench[name].get("n") == base[name].get("n")]
    pairs = [(n, a, b) for n, a, b in pairs
             if a is not None and b is not None]
    if not pairs:
        return {**base_summary, "suite": sorted(bench), "n_common": 0}
    deltas = [(name, acc - ref) for name, acc, ref in pairs]
    worst_name, worst_delta = min(deltas, key=lambda x: x[1])
    return {
        "accuracy": sum(a for _, a, _ in pairs) / len(pairs),
        "base_accuracy": sum(b for _, _, b in pairs) / len(pairs),
        "base_n_tasks": len(base_scores),
        "delta": sum(a - b for _, a, b in pairs) / len(pairs),
        "worst_name": worst_name,
        "worst_delta": worst_delta,
        "n_common": len(pairs),
        "suite": [n for n, _, _ in pairs],
    }


def collect(runs_dir: Path = Path("runs")) -> tuple[list[dict], dict, list[str]]:
    rows, bases = [], {}
    std_bases = standard_bases(runs_dir)
    unevaluated = []
    for d in sorted(runs_dir.iterdir()):
        f = d / "eval" / "tasks.json" if not d.name.startswith("base-tasks-") \
            else d / "tasks.json"
        if not f.exists():
            # Denominators must not be self-referential: a real run whose
            # battery never ran would otherwise just vanish from every
            # table. Name it loudly instead.
            if (d / "config.yaml").exists() and (
                    (d / "checkpoint").exists()
                    or (d / "CHECKPOINT_PRUNED.md").exists()
            ) and not d.name.startswith("certify_"):
                # certify_* dirs are train_certify.py instrument variants —
                # never battery-evaluated, not missing science.
                unevaluated.append(d.name)
            continue
        r = json.loads(f.read_text())
        measured = recall_results(r)
        entry = {"run": d.name, "model": r.get("model", "?"),
                 "recall": measured}
        if d.name.startswith("base-tasks-"):
            for corpus, result in measured.items():
                bases[(entry["model"], corpus)] = result
        else:
            cfg = d / "config.yaml"
            c = {}
            if cfg.exists():
                c = yaml.safe_load(cfg.read_text())
                entry["schedule"] = (c.get("train") or {}).get("schedule", "?")
            entry["scope"] = training_scope(c, set(measured))
            entry["damage"] = damage_result(d, entry["model"], std_bases)
            rows.append(entry)
    if unevaluated:
        print(f"WARNING: {len(unevaluated)} run(s) with a config and "
              "checkpoint but NO tasks.json are excluded from every table: "
              + ", ".join(sorted(unevaluated)), file=sys.stderr)
    return rows, bases, sorted(unevaluated)


def result_cell(row: dict, corpus: str, bases: dict) -> tuple[str, str, str]:
    result = row["recall"].get(corpus)
    base = bases.get((row["model"], corpus), {}).get("overall_word_acc")
    score = result.get("overall_word_acc") if result else None
    delta = None if score is None or base is None else score - base
    return fmt(score), fmt(base), fmt(delta, signed=True)


def fmt(value, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def scope_label(scope: tuple[str, ...]) -> str:
    return " + ".join(CORPUS_LABELS.get(c, c) for c in scope)


def main() -> None:
    rows, bases, unevaluated = collect()
    if not rows:
        sys.exit("no runs/*/eval/tasks.json yet — wait for the re-eval queue")
    rows.sort(key=lambda r: (r["scope"], r["model"], r["run"]))
    scope_complete = sum(set(r["scope"]).issubset(r["recall"]) for r in rows)
    damage_complete = sum(bool(r["damage"].get("n_common")) for r in rows)
    checkpoint_total = len(rows) + len(unevaluated)
    out = ["# Checkpoint re-evaluation: recall and model damage", "",
           f"Recall artifacts exist for {len(rows)}/{checkpoint_total} checkpoints; "
           f"{scope_complete}/{checkpoint_total} cover every corpus in the declared "
           f"training scope. Checkpoint-versus-epoch-zero standard-capability "
           f"results exist for "
           f"{damage_complete}/{checkpoint_total} checkpoints. There are {len(bases)} "
           "corpus-specific recall base references.", "",
           "Recall word accuracy is the fraction of reference words recovered "
           "in order, averaged over next, previous, and cloze prompts. Each Δ "
           "uses the epoch-zero model on the same corpus. A dash means that "
           "corpus/reference has not been evaluated; it is never imputed from "
           "the other author.", ""]

    if unevaluated:
        out += ["## Missing checkpoint evaluations", "",
                "These checkpoints are included in every coverage denominator "
                "but excluded from score rankings until `eval/tasks.json` "
                "exists:", ""]
        out.extend(f"- {name}" for name in unevaluated)
        out.append("")

    def _rung_key(corpus: str) -> tuple:
        m = re.search(r"(\d+)$", corpus)
        return (corpus.split("_")[0], int(m.group(1)) if m else 0)

    scopes = sorted({r["scope"] for r in rows},
                    key=lambda s: (len(s), tuple(_rung_key(c) for c in s)))
    for scope in scopes:
        subset = [r for r in rows if r["scope"] == scope]
        header = "".join(f" {CORPUS_LABELS.get(c, c)} | epoch 0 | Δ |"
                         for c in scope)
        out += [f"## Trained for {scope_label(scope)}", "",
                "| run | model |" + header,
                "|---|---|" + "---:|" * (3 * len(scope))]
        for r in subset:
            cells = []
            for corpus in scope:
                cells += result_cell(r, corpus, bases)
            out.append(f"| {r['run']} | {r['model'].split('/')[-1]} | "
                       + " | ".join(cells) + " |")
        out.append("")

    out += ["## Model damage: fixed standard benchmark subsets", "",
            "This is the capability check. Accuracy and Δ use the same standard "
            "tasks in both checkpoint and epoch-zero "
            "artifacts — the primary suite is ARC-Easy, ARC-Challenge, and "
            "HellaSwag (n=100 fixed subsets); legacy destruction.json "
            "fallbacks may add HellaSwag/MMLU/ARC-Challenge/WinoGrande/"
            "MMLU-Pro at n=200. Negative Δ means lost general "
            "knowledge/skill. The old custom prose-loss task is no longer "
            "part of checkpoint re-evaluation.", "",
            "| run | model | common/epoch-zero tasks | accuracy | epoch 0 | Δ | worst loss |",
            "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        d = r["damage"]
        worst = ("—" if d.get("worst_delta") is None else
                 f"{d['worst_name']} {d['worst_delta']:+.2f}")
        out.append(f"| {r['run']} | {r['model'].split('/')[-1]} | "
                   f"{d.get('n_common', 0)}/{d.get('base_n_tasks', 0)} | "
                   f"{fmt(d.get('accuracy'))} | "
                   f"{fmt(d.get('base_accuracy'))} | "
                   f"{fmt(d.get('delta'), signed=True)} | {worst} |")

    if bases:
        out += ["", "## Corpus-specific base references", ""]
        for (model, corpus), result in sorted(bases.items()):
            out.append(f"- {model} — {CORPUS_LABELS.get(corpus, corpus)}: "
                       f"{result['overall_word_acc']:.2f}")
    Path("runs/tasks_report.md").write_text("\n".join(out) + "\n")
    print(f"wrote runs/tasks_report.md ({len(rows)} rows)")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ranked = []
        for row in rows:
            for corpus, result in row["recall"].items():
                ranked.append((result.get("overall_word_acc") or 0, corpus, row))
        top = sorted(ranked, reverse=True, key=lambda x: x[0])[:25]
        fig, ax = plt.subplots(figsize=(9, 0.34 * len(top) + 1.5))
        names = [f"{r['run'][:31]} "
                 f"[{'M' if c == 'machado' else c.replace('quijote_ch', 'Q')}]"
                 for _, c, r in top][::-1]
        vals = [score for score, _, _ in top][::-1]
        colors = ["#3b7" if corpus == "machado" else "#b73"
                  for _, corpus, _ in top][::-1]
        ax.scatter(vals, range(len(top)), c=colors, alpha=0.8)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("word accuracy")
        ax.set_title("Top corpus-specific recall measurements")
        fig.tight_layout()
        fig.savefig("runs/tasks_report.png", dpi=130)
        print("wrote runs/tasks_report.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
