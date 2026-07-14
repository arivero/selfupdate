"""Generate the explicit strict-local Qwen3.5-4B one-day screen.

The generated YAML files are ordinary audited experiment configs.  This
script is only a deterministic authoring aid; the committed files remain the
campaign source of truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs" / "experiments" / "pareto_v2" / "screen_4b"
QUEUE = ROOT / "scripts" / "queue_pareto_v2_4b_screen_20260714.tsv"
BASE = "configs/experiments/pareto_v2/base_qwen35_4b.yaml"

LOSSES = ("huber", "cosine", "delta_cosine", "lens_kl")
CENSORSHIPS = ("remove", "pad_random")
GEOMETRIES = {
    "b8_all": (8, 0, "token_mean", "bucketed", 100, 1800),
    "b8_k64": (8, 64, "token_mean", "bucketed", 90, 2100),
    "b4_k128": (4, 128, "token_mean", "bucketed", 80, 2400),
    "b16_k32": (16, 32, "token_mean", "bucketed", 70, 2400),
    "b1_all": (1, 0, "answer_mean", "padded", 60, 7200),
}


def render() -> tuple[dict[Path, str], str]:
    files = {}
    rows = [
        "# done_file\tneed_mb\tafter\tcommand\tn_gpus\tpriority\texpected_seconds\tcache_group",
        "# Strict-local dataset-v5 / pipeline-v2 screen. One tile is one optimizer update.",
    ]
    for geometry, (b, k, reduction, batching, priority, expected) in GEOMETRIES.items():
        for loss in LOSSES:
            for censorship in CENSORSHIPS:
                stem = f"qwen35_4b_{loss}_{censorship}_{geometry}"
                run = f"pareto_v2_screen_{stem}"
                path = OUT / f"{stem}.yaml"
                data = {
                    "run_name": run,
                    "mask": {"compaction": censorship},
                    "train": {
                        "pipeline_version": 2,
                        "hidden_loss": loss,
                        "update_granularity": "grid",
                        "answers_per_update": b,
                        "tokens_per_answer_update": k,
                        "update_reduction": reduction,
                        "batching": batching,
                        "micro_batch": b,
                        "grad_accum": 1,
                        "conn_window": 1,
                        "conn_stride": 1,
                    },
                }
                files[path] = yaml.safe_dump(data, sort_keys=False)
                experiment = path.relative_to(ROOT)
                command = (
                    f"scripts/l40s_exec.sh scripts/train_and_report.py "
                    f"--config {BASE} --experiment {experiment}"
                )
                rows.append("\t".join((
                    f"runs/{run}/report_manifest.json", "16000", "-", command,
                    "1", str(priority), str(expected), "qwen35_4b",
                )))
    return files, "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if committed outputs differ; do not write")
    args = ap.parse_args()
    files, queue = render()
    stale = []
    for path, text in files.items():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            stale.append(path)
    if not QUEUE.is_file() or QUEUE.read_text(encoding="utf-8") != queue:
        stale.append(QUEUE)
    if args.check:
        if stale:
            raise SystemExit("stale Pareto-v2 screen outputs: "
                             + ", ".join(str(p.relative_to(ROOT)) for p in stale))
        print(f"Pareto-v2 screen is current: {len(files)} configs")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
    QUEUE.write_text(queue, encoding="utf-8")
    print(f"wrote {len(files)} configs and {QUEUE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
