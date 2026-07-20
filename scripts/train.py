"""Train and evaluate a pipeline-v4.6 student.

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
        help="pipeline-v4.6 layer-shard stage this process runs (placement "
             "only: selects the owned block range from train.v4_stage_splits "
             "and pins model.device to that stage's physical card)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    if cfg.train.method != "layerwise":
        sys.exit(f"unsupported train.method {cfg.train.method!r}; use 'layerwise'")
    if cfg.train.pipeline_version != 4:
        sys.exit(
            "this checkout is pipeline-v4.6 only; set "
            "train.pipeline_version=4")
    selected_stage = args.v4_stage
    if (selected_stage is None and cfg.train.v4_stage_scoped
            and not cfg.train.v4_stage_splits and cfg.train.v4_stage < 0):
        # Rotary PPP1 is a real one-rank pipeline. Direct invocations normalize
        # to stage 0 so evaluation uses the same v4.6 collective protocol as
        # launch_v4_stages.sh.
        selected_stage = 0
    if selected_stage is not None:
        cfg.train.v4_stage = selected_stage
        stages = len(cfg.train.v4_stage_splits or []) + 1
        devices = list(cfg.train.v4_stage_devices or range(stages))
        if not 0 <= selected_stage < stages:
            sys.exit(f"--v4-stage {selected_stage} outside 0..{stages - 1}")
        # Physical id, never renumbered; each stage is one full-model process
        # on one card writing its own runs/<name>/stage<k>/ directory.
        cfg.model.device = f"cuda:{devices[selected_stage]}"
        cfg.run_name = f"{cfg.run_name}/stage{selected_stage}"

    run_dir = train_layerwise(cfg)
    print(f"run complete: {run_dir}")


if __name__ == "__main__":
    main()
