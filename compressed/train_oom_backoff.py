"""Run one training arm, retrying CUDA-OOM failures at smaller micro-batches.

The effective optimizer batch stays constant: each halving of micro_batch
doubles grad_accum.  The retry overlay is written below the run directory so
the exact recovery decision is auditable.  Non-OOM failures are never retried.
"""


# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP



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
    from selfupdate.config import load_config

    cfg = load_config(args.config, args.experiment)
    run_name = cfg.run_name
    original_micro = cfg.train.micro_batch
    original_accum = cfg.train.grad_accum
    original_overlay = yaml.safe_load(Path(args.experiment).read_text()) or {}
    state_dir = root / "runs" / run_name / ".oom_backoff"
    state_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(args.max_retries + 1):
        micro = max(1, original_micro // (2 ** attempt))
        accum = original_accum * (original_micro // micro)
        experiment = Path(args.experiment)
        if attempt:
            overlay = dict(original_overlay)
            train = dict(overlay.get("train") or {})
            train.update({"micro_batch": micro, "grad_accum": accum})
            overlay["train"] = train
            experiment = state_dir / f"attempt_{attempt}_mb{micro}.yaml"
            experiment.write_text(yaml.safe_dump(overlay, sort_keys=False))
        command = [sys.executable, "compressed/train.py", "--config", args.config,
                   "--experiment", str(experiment)]
        print(f"oom-backoff: attempt={attempt} micro_batch={micro} "
              f"grad_accum={accum}", flush=True)
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
                 "next_grad_accum": accum * 2, "reason": "cuda_oom"}
        (state_dir / "last_oom.json").write_text(json.dumps(state, indent=2) + "\n")
        print("oom-backoff: CUDA OOM detected; retrying with a halved micro-batch", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
