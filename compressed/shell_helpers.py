#!/usr/bin/env python3
"""Standalone replacements for shell heredocs which imported selfupdate."""

from __future__ import annotations


# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP


import argparse
import hashlib


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("v4-launch-info", "node-cache-index"):
        command = commands.add_parser(name)
        command.add_argument("base")
        command.add_argument("experiment")
    commands.add_parser("venv-import-check")
    args = parser.parse_args()

    if args.command == "venv-import-check":
        import selfupdate
        from selfupdate.eval.standard import STANDARD_TASKS
        digest = hashlib.sha256()
        for entry in sorted(_SU_ARCHIVE.namelist()):
            digest.update(entry.encode("utf-8") + b"\0")
            digest.update(_SU_ARCHIVE.read(entry))
        print(f"eval.standard ok ({', '.join(list(STANDARD_TASKS)[:3])})")
        print(f"selfupdate bundle {selfupdate.__file__}")
        print(f"bundle sha256 {digest.hexdigest()}")
        return 0

    from selfupdate.config import load_config
    cfg = load_config(args.base, args.experiment)
    if args.command == "v4-launch-info":
        if cfg.train.pipeline_version != 4:
            raise SystemExit(
                "launch_v4_stages.sh requires pipeline_version=4")
        print(len(cfg.train.v4_stage_splits or []) + 1, cfg.run_name)
        return 0

    from selfupdate.teacher.cache import resolve_cache_dir
    from selfupdate.teacher.node_epoch0 import ready_manifest, runtime_identity
    root, cache_hash = resolve_cache_dir(cfg)
    if ready_manifest(root, cache_hash, compatibility=runtime_identity()) is None:
        raise SystemExit(
            f"node-epoch0 index was not atomically published: {root}")
    print(root / "index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
