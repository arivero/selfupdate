"""Block rotation for pipeline-v4 (plan B4, 2026-07-17; staged pipeline
2026-07-18 — owner: "rotations should also be optimisable for max GPU
usage").

A stage may own more layers than fit its card. Because v4 blocks are
gradient-independent and the walk is ``layer_major``, only ONE owned block
needs GPU residency at a time: page block L's frozen base weights host->GPU,
train every cohort while resident, evict (drop — frozen weights are never
written back), prefetch the next owned block meanwhile.

Transport is a two-hop staged pipeline so the GPU never waits for the
truck:

  mmap master --(background thread memcpy)--> pinned host buffer
              --(async H2D on side stream)--> device buffer

Both hops run while the CURRENT block computes: the thread copy releases
the GIL, and the H2D is enqueued from that same thread onto the side
stream, so a prefetched block is already device-resident when
``activate`` swaps it in. Buffers are pooled by block signature
(name/shape/dtype set) — heterogeneous stacks (hybrid linear/full
attention, MoE vs dense blocks) each get their own ping-pong pair.
``stall`` telemetry records the honest per-activate wait (thread join +
transfer completion), so rotation's utilization cost is a measured number
in the epoch rows, not a guess.

The rotation UNIT is {frozen base weights (one-way) + Adam moments (BOTH
ways — trivial bytes under LoRA, the exact machinery base-weight
fine-tuning at 397B needs)}. LoRA adapters stay GPU-resident always.
"""

from __future__ import annotations

import threading
import time

import torch

__all__ = ["BlockRotator", "decide_rotate"]


def decide_rotate(owned_bytes: int, device, *, headroom: float = 1.25,
                  reserve_bytes: int = 20 << 30) -> bool:
    """auto policy: rotate when resident owned weights would not leave the
    activation/optimizer reserve free."""
    free, _total = torch.cuda.mem_get_info(device)
    return owned_bytes * headroom + reserve_bytes > free


class _BufferPool:
    """Pinned-host + device buffer sets, pooled by block signature."""

    def __init__(self, device):
        self.device = device
        self.free: dict[tuple, list[dict]] = {}

    @staticmethod
    def signature(tensors: dict[str, torch.Tensor]) -> tuple:
        return tuple(sorted((n, tuple(t.shape), str(t.dtype))
                            for n, t in tensors.items()))

    def take(self, tensors: dict[str, torch.Tensor]) -> dict:
        sig = self.signature(tensors)
        pool = self.free.get(sig)
        if pool:
            return pool.pop()
        host = {n: torch.empty_like(t, device="cpu", pin_memory=True)
                for n, t in tensors.items()}
        dev = {n: torch.empty_like(t, device=self.device)
               for n, t in tensors.items()}
        return {"host": host, "dev": dev, "sig": sig}

    def give(self, buf: dict) -> None:
        self.free.setdefault(buf["sig"], []).append(buf)


class BlockRotator:
    """Pages one owned block's rotation unit at a time.

    Contract: ``activate(L)`` blocks until L's weights are on the device
    and swaps them live; ``evict(L)`` restores the CPU masters and pages
    L's optimizer moments out; ``prefetch(L)`` starts L's two-hop
    transfer in the background. All swaps are ``p.data`` replacement —
    module structure, autograd graphs, and optimizer param identity are
    untouched.
    """

    def __init__(self, stack, owned, device, optimizers=None):
        self.stack = stack
        self.device = torch.device(device)
        self.optimizers = optimizers or {}
        self.stream = torch.cuda.Stream(self.device)
        self.pool = _BufferPool(self.device)
        self.masters: dict[int, dict[str, torch.Tensor]] = {}
        self._inflight: dict[int, tuple] = {}   # L -> (thread, buf, event)
        self._staged: dict[int, dict] = {}      # L -> buf (device-resident)
        # telemetry (take_counters drains)
        self.stall_seconds = 0.0
        self.h2d_bytes = 0
        self.pages = 0
        for L in owned:
            block = stack.blocks[L - 1]
            cpu = {}
            for name, p in block.named_parameters():
                if not p.requires_grad and p.device.type == "cpu":
                    cpu[name] = p.data
            for name, b in block.named_buffers():
                if b.device.type == "cpu":
                    cpu["buffer:" + name] = b.data
            if cpu:
                self.masters[L] = cpu

    @property
    def active_bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for buf in self._staged.values()
                   for t in buf["dev"].values())

    def prefetch(self, L: int) -> None:
        if (L not in self.masters or L in self._staged
                or L in self._inflight):
            return
        masters = self.masters[L]
        buf = self.pool.take(masters)
        event = torch.cuda.Event()
        side = self.stream

        def _pipeline():
            # Threads get their own current-device defaulting to cuda:0
            # (the stray-context defect, owner 2026-07-18): pin it FIRST
            # or the stream/copy calls below build a foreign context.
            torch.cuda.set_device(self.device)
            # Hop 1: mmap -> pinned (memcpy releases the GIL, overlaps
            # GPU compute). Hop 2: pinned -> device, async on the side
            # stream, enqueued from this thread (same CUDA context).
            for name, cpu_t in masters.items():
                buf["host"][name].copy_(cpu_t)
            with torch.cuda.stream(side):
                for name in masters:
                    buf["dev"][name].copy_(buf["host"][name],
                                           non_blocking=True)
                event.record(side)

        thread = threading.Thread(target=_pipeline, daemon=True)
        thread.start()
        self._inflight[L] = (thread, buf, event)

    def activate(self, L: int) -> None:
        if L not in self.masters:
            return  # fully resident (or foreign) block: nothing to do
        started = time.perf_counter()
        if L not in self._staged:
            self.prefetch(L)
            thread, buf, event = self._inflight.pop(L)
            thread.join()
            event.synchronize()
            self._staged[L] = buf
        self.stall_seconds += time.perf_counter() - started
        self.pages += 1
        self.h2d_bytes += sum(t.numel() * t.element_size()
                              for t in self._staged[L]["dev"].values())
        self._swap(L, self._staged[L]["dev"])
        self._moments_to(L, self.device)

    def evict(self, L: int) -> None:
        if L not in self.masters:
            return
        self._swap(L, self.masters[L])
        buf = self._staged.pop(L, None)
        if buf is not None:
            self.pool.give(buf)
        self._moments_to(L, torch.device("cpu"))

    def take_counters(self) -> dict:
        out = {"rotation_stall_seconds": round(self.stall_seconds, 3),
               "rotation_h2d_gb": round(self.h2d_bytes / 2**30, 2),
               "rotation_pages": self.pages}
        self.stall_seconds = 0.0
        self.h2d_bytes = 0
        self.pages = 0
        return out

    def _swap(self, L: int, tensors: dict[str, torch.Tensor]) -> None:
        block = self.stack.blocks[L - 1]
        params = dict(block.named_parameters())
        buffers = dict(block.named_buffers())
        for name, t in tensors.items():
            if name.startswith("buffer:"):
                buffers[name[len("buffer:"):]].data = t
            else:
                params[name].data = t

    def _moments_to(self, L: int, device) -> None:
        """Adam moments ride their block (state dicts are keyed by param
        object — identity survives the .data swaps)."""
        opt = self.optimizers.get(L)
        if opt is None:
            return
        for group in opt.param_groups:
            for p in group["params"]:
                state = opt.state.get(p)
                if not state:
                    continue
                for key, val in state.items():
                    if torch.is_tensor(val) and val.device != device:
                        state[key] = val.to(device, non_blocking=True)
