"""Run one v4 training arm, retrying CUDA-OOM failures with smaller cohorts.

V4 has no gradient accumulation: each block/cohort write is an atomic local
update.  A retry therefore changes ``micro_batch`` openly and records that
scientific protocol change.  Non-OOM failures are never retried.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


OOM_MARKERS = ("torch.OutOfMemoryError", "CUDA out of memory")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--max-retries", type=int, default=2)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from selfupdate.config import load_config

    cfg = load_config(args.config, args.experiment)
    run_name = cfg.run_name
    original_micro = cfg.train.micro_batch
    original_overlay = yaml.safe_load(Path(args.experiment).read_text()) or {}
    state_dir = root / "runs" / run_name / ".oom_backoff"
    state_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(args.max_retries + 1):
        micro = max(1, original_micro // (2 ** attempt))
        experiment = Path(args.experiment)
        if attempt:
            overlay = dict(original_overlay)
            train = dict(overlay.get("train") or {})
            train.update({"micro_batch": micro})
            overlay["train"] = train
            experiment = state_dir / f"attempt_{attempt}_mb{micro}.yaml"
            experiment.write_text(yaml.safe_dump(overlay, sort_keys=False))
        command = [sys.executable, "scripts/train.py", "--config", args.config,
                   "--experiment", str(experiment)]
        print(f"oom-backoff: attempt={attempt} micro_batch={micro}", flush=True)
        proc = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
        saw_oom = False
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            saw_oom = saw_oom or any(marker in line for marker in OOM_MARKERS)
        rc = proc.wait()
        if rc == 0:
            return 0
        if not saw_oom or micro == 1 or attempt == args.max_retries:
            return rc
        state = {"failed_attempt": attempt, "failed_micro_batch": micro,
                 "next_micro_batch": max(1, micro // 2),
                 "reason": "cuda_oom"}
        (state_dir / "last_oom.json").write_text(json.dumps(state, indent=2) + "\n")
        print("oom-backoff: CUDA OOM detected; retrying with a halved micro-batch", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
