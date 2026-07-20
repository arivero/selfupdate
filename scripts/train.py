"""Train and evaluate a pipeline-v4.5 student.

Usage:
    python scripts/train.py --experiment configs/experiments/lw_summed_0p6b_rag.yaml

Configs are ``configs/base.yaml`` plus a small experiment overlay; every
knob is validated against the chosen schedule at dispatch (see
``selfupdate/train/validate.py``). Run outputs land in ``runs/<run_name>/``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.utils.env import cap_cpu_threads

cap_cpu_threads()

from selfupdate.config import load_config
from selfupdate.train.layerwise import train_layerwise


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Layerwise forward-distillation trainer")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument(
        "--v4-stage", type=int, default=None,
        help="pipeline-v4.5 layer-shard stage this process runs (placement "
             "only: selects the owned block range from train.v4_stage_splits "
             "and pins model.device to that stage's physical card)")
    # Private reconstructed-evaluator worker. The live trainer owns the
    # publication/offload/ack protocol and self-invokes this mode only when
    # the native evaluator rejects an architecture.
    ap.add_argument("--v4-battery-worker", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--v4-battery-run-dir", help=argparse.SUPPRESS)
    ap.add_argument("--v4-battery-epoch", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--v4-battery-stages", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.v4_battery_worker:
        required = {
            "--experiment": args.experiment,
            "--v4-battery-run-dir": args.v4_battery_run_dir,
            "--v4-battery-epoch": args.v4_battery_epoch,
            "--v4-battery-stages": args.v4_battery_stages,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            ap.error("battery worker missing " + ", ".join(missing))
        from selfupdate.eval.reconstructed_battery import main as battery_main

        battery_main([
            "--config", args.config,
            "--experiment", args.experiment,
            "--run-dir", args.v4_battery_run_dir,
            "--epoch", str(args.v4_battery_epoch),
            "--stages", str(args.v4_battery_stages),
        ])
        return
    cfg = load_config(args.config, args.experiment)
    # The internal reconstructed fallback re-loads the same config pair in a
    # child process; the file paths are only known here.
    import os

    os.environ["SELFUPDATE_V4_CONFIG"] = (
        f"{args.config}::{args.experiment or ''}")

    if cfg.train.method != "layerwise":
        sys.exit(f"unsupported train.method {cfg.train.method!r}; use 'layerwise'")
    if cfg.train.pipeline_version != 4:
        sys.exit(
            "this checkout is pipeline-v4.5 only; set "
            "train.pipeline_version=4")
    if args.v4_stage is not None:
        cfg.train.v4_stage = args.v4_stage
        stages = len(cfg.train.v4_stage_splits or []) + 1
        devices = list(cfg.train.v4_stage_devices or range(stages))
        if not 0 <= args.v4_stage < stages:
            sys.exit(f"--v4-stage {args.v4_stage} outside 0..{stages - 1}")
        # Physical id, never renumbered; each stage is one full-model process
        # on one card writing its own runs/<name>/stage<k>/ directory.
        cfg.model.device = f"cuda:{devices[args.v4_stage]}"
        cfg.run_name = f"{cfg.run_name}/stage{args.v4_stage}"

    run_dir = train_layerwise(cfg)
    print(f"run complete: {run_dir}")


if __name__ == "__main__":
    main()
