"""Report over the three-task recall battery (runs/*/eval/tasks.json).

Builds runs/tasks_report.md + runs/tasks_report.png from every run that has
a tasks.json, with the base-model references (runs/base-tasks-*) as the
zero line. Metrics are the plain accuracies of the 2026-07-10 eval refocus:
exact-match and word_acc for next / prev / cloze (cloze also by deletion
count). Run after the re-eval queue drains:

    python scripts/tasks_report.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml


def collect() -> tuple[list[dict], dict]:
    rows, bases = [], {}
    for d in sorted(Path("runs").iterdir()):
        f = d / "eval" / "tasks.json" if not d.name.startswith("base-tasks-") \
            else d / "tasks.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        entry = {"run": d.name, "model": r.get("model", "?"),
                 "general_ce": (r.get("general") or {}).get("mean_ce")}
        for t in ("next", "prev", "cloze"):
            v = (r.get("tasks") or {}).get(t) or {}
            entry[f"{t}_word"] = v.get("word_acc")
            entry[f"{t}_exact"] = v.get("exact")
        entry["by_deletions"] = ((r.get("tasks") or {}).get("cloze")
                                 or {}).get("by_deletions") or {}
        entry["overall"] = r.get("overall_word_acc")
        if d.name.startswith("base-tasks-"):
            bases[entry["model"]] = entry
        else:
            cfg = d / "config.yaml"
            if cfg.exists():
                c = yaml.safe_load(cfg.read_text())
                entry["schedule"] = (c.get("train") or {}).get("schedule", "?")
            rows.append(entry)
    return rows, bases


def main() -> None:
    rows, bases = collect()
    if not rows:
        sys.exit("no runs/*/eval/tasks.json yet — wait for the re-eval queue")
    rows.sort(key=lambda r: (r["model"], -(r["overall"] or 0)))
    out = ["# Recall-task report (three-task battery)", "",
           f"{len(rows)} checkpoints, {len(bases)} base references. "
           "word = fraction of reference words recovered in order; "
           "exact = normalized exact match. Δ columns are versus the base "
           "model (what training added).", ""]
    header = ("| run | model | next word | prev word | cloze word | overall "
              "| Δoverall | exact (n/p/c) | gen-CE |")
    out += [header, "|" + "---|" * 9]
    for r in rows:
        base = bases.get(r["model"], {})
        delta = (None if r["overall"] is None or base.get("overall") is None
                 else r["overall"] - base["overall"])
        fmt = lambda v: "—" if v is None else f"{v:.2f}"
        out.append(
            f"| {r['run'][:40]} | {r['model'].split('/')[-1]} "
            f"| {fmt(r['next_word'])} | {fmt(r['prev_word'])} "
            f"| {fmt(r['cloze_word'])} | {fmt(r['overall'])} "
            f"| {fmt(delta)} "
            f"| {fmt(r['next_exact'])}/{fmt(r['prev_exact'])}/{fmt(r['cloze_exact'])} "
            f"| {fmt(r['general_ce'])} |")
    if bases:
        out += ["", "## Base references", ""]
        for m, b in sorted(bases.items()):
            out.append(f"- {m}: overall {b['overall']:.2f} "
                       f"(next {b['next_word']:.2f} / prev {b['prev_word']:.2f} "
                       f"/ cloze {b['cloze_word']:.2f})")
    Path("runs/tasks_report.md").write_text("\n".join(out) + "\n")
    print(f"wrote runs/tasks_report.md ({len(rows)} rows)")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = sorted(rows, key=lambda r: -(r["overall"] or 0))[:25]
        fig, ax = plt.subplots(figsize=(9, 0.34 * len(top) + 1.5))
        names = [r["run"][:36] for r in top][::-1]
        for t, color in (("next", "#3b7"), ("prev", "#37b"), ("cloze", "#b73")):
            vals = [r[f"{t}_word"] or 0 for r in top][::-1]
            ax.plot(vals, range(len(top)), "o", label=t, color=color, alpha=0.8)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("word accuracy")
        ax.set_title("Top checkpoints — three-task recall battery")
        ax.legend()
        fig.tight_layout()
        fig.savefig("runs/tasks_report.png", dpi=130)
        print("wrote runs/tasks_report.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
