"""Generate the explicit strict-local Qwen3.5-4B finite-tile screen.

The generated YAML files are ordinary audited experiment configs.  This
script is only a deterministic authoring aid; the committed files remain the
campaign source of truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs" / "experiments" / "pareto_v2" / "small_tiles_4b"
QUEUE = ROOT / "scripts" / "queue_pareto_v2_4b_screen_20260714.tsv"
BASE = "configs/experiments/pareto_v2/base_qwen35_4b.yaml"

LOSSES = ("huber", "cosine", "delta_cosine", "lens_kl")
CENSORSHIPS = ("remove", "pad_random")
# Every queued successor has a finite K.  The 128-cell diagonal is the
# controlled fourfold-smaller comparison to the 512-cell broad geometries.
# The follow-on 16/32-cell diagonal tests the many-small-updates regime.
# Both preserve answerwise/tokenwise shape as an explicit variable.
GEOMETRIES = {
    "b1_k128": (1, 128, "token_mean", "padded", 120, 7200),
    "b2_k64": (2, 64, "token_mean", "padded", 115, 5400),
    "b4_k32": (4, 32, "token_mean", "bucketed", 110, 4500),
    "b8_k16": (8, 16, "token_mean", "bucketed", 105, 4200),
    "b16_k8": (16, 8, "token_mean", "bucketed", 100, 4200),
    "b1_k16": (1, 16, "token_mean", "padded", 90, 10800),
    "b1_k32": (1, 32, "token_mean", "padded", 85, 7200),
    "b2_k16": (2, 16, "token_mean", "padded", 80, 6000),
    "b4_k8": (4, 8, "token_mean", "bucketed", 75, 6000),
    "b8_k4": (8, 4, "token_mean", "bucketed", 70, 6000),
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
                run = f"pareto_v2_micro_{stem}"
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
