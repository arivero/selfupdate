"""Prefetch Hugging Face model snapshots for upcoming queued runs.

Usage:
    python scripts/prefetch_models.py model/name another/model
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from huggingface_hub import snapshot_download


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out-dir", default="runs/prefetch")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        safe = model.replace("/", "__")
        print(f"[{stamp()}] prefetch {model}", flush=True)
        try:
            path = snapshot_download(model)
        except Exception:
            msg = traceback.format_exc()
            (out_dir / f"{safe}.failed").write_text(msg)
            print(msg, flush=True)
            continue
        (out_dir / f"{safe}.done").write_text(path + "\n")
        print(f"[{stamp()}] done {model}: {path}", flush=True)


if __name__ == "__main__":
    main()
