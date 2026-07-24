"""Merge pipeline-v4 per-stage LoRA adapter shards into one checkpoint.

Each v4 stage trains a DISJOINT set of blocks and saves a full adapter file
in which every non-owned block's LoRA B matrix is still zero (PEFT zero-init,
untouched by that stage).  The merge therefore takes, for every block, the
adapter tensors from the ONE stage that owns it — no averaging, no conflict.
The result is bit-identical to what a single-process run would have saved,
because v4 layers are independent.

Usage:
    python scripts/merge_v4_adapters.py runs/<run_name> [--out runs/<run_name>/checkpoint]

Reads runs/<run>/stage*/checkpoint/, writes a merged checkpoint directory
with the stage-0 tokenizer/config files and the combined adapter weights.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safetensors.torch import load_file, save_file  # noqa: E402

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _owned_blocks(manifest: dict) -> range:
    lo, hi = manifest["owned_blocks"]
    return range(lo, hi + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    stage_dirs = sorted(
        args.run_dir.glob("stage*/checkpoint"),
        key=lambda p: int(p.parent.name.removeprefix("stage")))
    if not stage_dirs:
        raise SystemExit(f"no stage*/checkpoint under {args.run_dir}")

    ownership: dict[int, int] = {}
    tensors = {}
    for stage_index, ckpt in enumerate(stage_dirs):
        v4_manifest = json.loads(
            (ckpt / "v4_stage_manifest.json").read_text())
        owned = _owned_blocks(v4_manifest)
        shard = load_file(str(ckpt / "adapter_model.safetensors"))
        for key, value in shard.items():
            match = _LAYER_RE.search(key)
            if match is None:
                # embed/norm/head never carry adapters on this branch; any
                # non-layer adapter tensor would be a frozen-vocab violation.
                raise SystemExit(f"non-block adapter tensor in {ckpt}: {key}")
            block = int(match.group(1)) + 1  # HF 0-based -> repo 1-based
            if block not in owned:
                continue
            if block in ownership and ownership[block] != stage_index:
                raise SystemExit(
                    f"block {block} claimed by stages {ownership[block]} "
                    f"and {stage_index}")
            ownership[block] = stage_index
            tensors[key] = value
        # Non-owned zero-init tensors: keep one copy so the adapter file is
        # complete for PEFT loading (they are all-zero everywhere).
        for key, value in shard.items():
            if key not in tensors:
                tensors[key] = value

    out = args.out or (args.run_dir / "checkpoint")
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    staging = out.parent / (out.name + ".incomplete")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(stage_dirs[0], staging)
    (staging / "v4_stage_manifest.json").unlink(missing_ok=True)
    save_file(tensors, str(staging / "adapter_model.safetensors"))
    (staging / "v4_merge_manifest.json").write_text(json.dumps({
        "stages": [str(p) for p in stage_dirs],
        "block_ownership": {str(k): v for k, v in sorted(ownership.items())},
    }, indent=2) + "\n")
    staging.rename(out)
    print(f"merged {len(stage_dirs)} stages, {len(ownership)} owned blocks "
          f"-> {out}")


if __name__ == "__main__":
    main()
