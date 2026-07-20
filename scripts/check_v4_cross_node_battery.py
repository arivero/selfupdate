#!/usr/bin/env python3
"""CPU-only contract check for cross-node v4 battery publications."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfupdate.train.online_v4 import _RelayFiles  # noqa: E402
from selfupdate.train.relay_nccl import NcclBoundaryRelay  # noqa: E402


class _ScriptedDist:
    """Stage-0 view of broadcasts from two synthetic remote ranks."""

    def __init__(self, remote: dict[int, bytes]):
        self.remote = remote
        self.calls: dict[int, int] = {}
        self.groups = 0

    def new_group(self, **kwargs):
        assert kwargs["backend"] == "nccl"
        self.groups += 1
        return "battery-group"

    def broadcast(self, tensor, src: int, group):
        assert group == "battery-group"
        call = self.calls.get(src, 0)
        self.calls[src] = call + 1
        if src and call == 0:                 # publication byte length
            tensor.fill_(len(self.remote[src]))
        elif src and call == 1:               # publication payload
            incoming = torch.frombuffer(
                bytearray(self.remote[src]), dtype=torch.uint8)
            tensor.copy_(incoming)
        # Third stage-0 broadcast is the common materialization status.


class _Store:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = str(value).encode()

    def check(self, keys):
        return all(key in self.values for key in keys)

    def get(self, key):
        return self.values[key]


def _publication(path: Path, launch: str, stage: int, epoch: int) -> bytes:
    save_file(
        {f"L{stage + 1:03d}.adapter": torch.tensor([stage], dtype=torch.int64)},
        str(path), metadata={
            "launch_id": launch, "from_host": f"host{stage}",
            "from_stage": str(stage), "to_stage": "broadcast",
            "epoch": str(epoch),
        })
    return path.read_bytes()


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        os.environ["SELFUPDATE_V4_LAUNCH_ID"] = "battery-selfcheck"
        os.environ.pop("SELFUPDATE_V4_RELAY_ROOT", None)
        rf = _RelayFiles(root / "run")
        epoch = 7
        local = rf.path(epoch, "adapters_stage0.st")
        local.parent.mkdir(parents=True)
        local.write_bytes(_publication(root / "p0.st", rf.launch_id, 0, epoch))
        remote = {
            stage: _publication(root / f"p{stage}.st", rf.launch_id,
                                stage, epoch)
            for stage in (1, 2)
        }

        relay = NcclBoundaryRelay.__new__(NcclBoundaryRelay)
        relay.dist = _ScriptedDist(remote)
        relay.stage = 0
        relay.stages = 3
        relay.device = torch.device("cpu")
        relay._battery_group = None
        relay._group_timeout = None
        relay._ready_store = _Store()
        relay._battery_status_timeout_s = 1.0
        relay.collect_battery_adapters(epoch, rf)

        for stage in (1, 2):
            path = rf.path(epoch, f"adapters_stage{stage}.st")
            assert path.read_bytes() == remote[stage]
            loaded = rf.read(path, expect_epoch=epoch, as_stage=0)
            assert int(loaded[f"L{stage + 1:03d}.adapter"].item()) == stage
        assert relay.dist.groups == 1
        assert relay.exchange_battery_status(epoch, 17) == 17
        assert relay._ready_store.get("battery_status_e0007") == b"17"
        assert relay.dist.groups == 1
    print("cross-node battery publication contract: PASS")


if __name__ == "__main__":
    main()
