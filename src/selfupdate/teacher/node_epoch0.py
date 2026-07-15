"""Coordination for node-local epoch-zero teacher caches.

``/dev/shm`` is local to one host.  Multiple training arms on that host may
request the same cache concurrently, so publication follows a small lease
protocol:

1. an atomic ``mkdir`` elects one builder;
2. the builder writes a private partial directory;
3. a ready manifest is written only after ``index.json`` is complete;
4. the complete directory is atomically renamed to its canonical name.

No PID from another host is interpreted.  That matters for Lustre leases, but
the node cache and its lock are deliberately on the same local tmpfs.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import time
from pathlib import Path


READY_NAME = ".selfupdate-node-epoch0-ready.json"
OWNER_NAME = "owner.json"
MATERIALIZER_SCHEMA = 1


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def runtime_identity() -> dict:
    """Numerical runtime fields that govern safe node-cache reuse."""
    import torch

    return {
        "materializer_schema": MATERIALIZER_SCHEMA,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }


def ready_manifest(root: str | Path, expected_hash: str,
                   compatibility: dict | None = None) -> dict | None:
    """Return a validated ready manifest, or ``None`` for an incomplete cache."""
    root = Path(root)
    ready = _read_json(root / READY_NAME)
    index = _read_json(root / "index.json")
    if not ready or not index:
        return None
    if ready.get("cache_hash") != expected_hash:
        return None
    if index.get("config_hash") != expected_hash:
        return None
    if ready.get("examples") != len(index.get("examples", {})):
        return None
    if compatibility and any(
            ready.get(key) != value for key, value in compatibility.items()):
        return None
    return ready


class NodeEpoch0Lease:
    """Elect and supervise one epoch-zero cache builder on this host."""

    def __init__(self, final_root: str | Path, expected_hash: str,
                 *, compatibility: dict | None = None,
                 wait_seconds: float = 7200.0, poll_seconds: float = 2.0):
        self.final_root = Path(final_root)
        self.expected_hash = expected_hash
        self.wait_seconds = wait_seconds
        self.poll_seconds = poll_seconds
        self.compatibility = dict(compatibility or {})
        self.host = socket.gethostname().split(".")[0]
        self.pid = os.getpid()
        self.lock = self.final_root.parent / (
            f".{self.final_root.name}.epoch0-build.lock")
        self.partial = self.final_root.parent / (
            f".{self.final_root.name}.partial-{self.host}-{self.pid}")
        self._owned = False
        atexit.register(self.abort)

    def _owner_alive(self, owner: dict | None) -> bool:
        if not owner or owner.get("host") != self.host:
            return True
        try:
            pid = int(owner["pid"])
        except (KeyError, TypeError, ValueError):
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _reclaim_dead_owner(self) -> bool:
        owner = _read_json(self.lock / OWNER_NAME)
        if owner is None:
            # Give the winner time to create owner.json after the atomic
            # mkdir. A process killed in that tiny interval must not leave a
            # two-hour unreapable lock.
            try:
                if time.time() - self.lock.stat().st_mtime < 30:
                    return False
            except FileNotFoundError:
                return True
        elif self._owner_alive(owner):
            return False
        stale = self.lock.with_name(
            f"{self.lock.name}.stale-{int(time.time())}-{self.pid}")
        try:
            self.lock.rename(stale)
        except FileNotFoundError:
            return True
        partial = Path(owner.get("partial_root", "")) if owner else None
        if partial and partial.parent == self.final_root.parent:
            shutil.rmtree(partial, ignore_errors=True)
        shutil.rmtree(stale, ignore_errors=True)
        return True

    def acquire(self) -> tuple[Path | None, dict | None]:
        """Return ``(partial_root, None)`` to the winner.

        A waiter returns ``(None, manifest)`` after the winner publishes.  It
        therefore never needs to load model weights merely to discover that
        another process is already generating epoch zero.
        """
        self.final_root.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        announced = False
        while True:
            ready = ready_manifest(
                self.final_root, self.expected_hash, self.compatibility)
            if ready is not None:
                return None, ready
            try:
                self.lock.mkdir()
            except FileExistsError:
                if self._reclaim_dead_owner():
                    continue
                elapsed = time.monotonic() - started
                if elapsed >= self.wait_seconds:
                    owner = _read_json(self.lock / OWNER_NAME)
                    raise TimeoutError(
                        f"timed out after {elapsed:.0f}s waiting for node "
                        f"epoch-zero cache {self.final_root}; owner={owner}")
                if not announced:
                    owner = _read_json(self.lock / OWNER_NAME)
                    print(f"epoch0 cache wait: root={self.final_root} owner={owner}")
                    announced = True
                time.sleep(self.poll_seconds)
                continue

            self._owned = True
            (self.lock / OWNER_NAME).write_text(json.dumps({
                "host": self.host,
                "pid": self.pid,
                "started_unix": time.time(),
                "cache_hash": self.expected_hash,
                "final_root": str(self.final_root),
                "partial_root": str(self.partial),
            }, indent=2) + "\n", encoding="utf-8")
            if ready_manifest(
                    self.final_root, self.expected_hash,
                    self.compatibility) is not None:
                self._release_lock()
                return None, ready_manifest(
                    self.final_root, self.expected_hash, self.compatibility)
            if self.final_root.exists():
                stale = self.final_root.with_name(
                    f".{self.final_root.name}.superseded-{int(time.time())}-{self.pid}")
                self.final_root.rename(stale)
                shutil.rmtree(stale)
            if self.partial.exists():
                shutil.rmtree(self.partial)
            self.partial.mkdir()
            return self.partial, None

    def publish(self, build_root: str | Path, metadata: dict) -> dict:
        """Validate and atomically publish the winner's private directory."""
        if not self._owned:
            raise RuntimeError("cannot publish a node cache without its lease")
        build_root = Path(build_root)
        index = _read_json(build_root / "index.json")
        if not index or index.get("config_hash") != self.expected_hash:
            raise RuntimeError(
                f"refusing incomplete epoch-zero cache publication: {build_root}")
        manifest = {
            "schema": 1,
            "kind": "node_epoch0_teacher_cache",
            "host": self.host,
            "builder_pid": self.pid,
            "cache_hash": self.expected_hash,
            "examples": len(index.get("examples", {})),
            "completed_unix": time.time(),
            **self.compatibility,
            **metadata,
        }
        (build_root / READY_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        build_root.rename(self.final_root)
        self._release_lock()
        return manifest

    def _release_lock(self) -> None:
        if not self._owned:
            return
        shutil.rmtree(self.lock, ignore_errors=True)
        self._owned = False

    def abort(self) -> None:
        """Release our own lease and reclaim incomplete tmpfs payloads."""
        if self._owned:
            shutil.rmtree(self.partial, ignore_errors=True)
        self._release_lock()
