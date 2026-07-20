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


def _dict_bytes(tensors: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors.values())


class _BufferPool:
    """Pinned-host + device buffer sets, pooled by block signature.

    TWO pools with OPPOSITE scarcity models, split because one shared count cap
    cannot serve both:

    * HOST (pinned) buffers pool WITHOUT a size cap. ``cudaHostAlloc`` of a
      multi-GB pinned buffer is expensive (~1-2 s at 15 GB) and host RAM is
      abundant (2 TB), so a re-pin per rotation — 2500+ activations over a
      DeepSeek store-fill — would dominate wall time. Reuse them freely.
    * DEVICE buffers pool UNDER A BYTE CAP (``max_pooled_dev_bytes``). Device
      memory is the scarce resource. A homogeneous model (Qwen: one ~2-13 GB
      signature) reuses ~2 cheaply. A HETEROGENEOUS model (deepseek_v4:
      sliding / HCA / CSA layers, each a distinct ~15 GB signature) would
      otherwise pool one giant device buffer PER layer type. Measured: with a
      count-only cap of 2, stage 0 still held staged(1)+inflight(1)+pooled(2)
      = 4 x ~15 GB = 60 GB of frozen weights, leaving no room for DeepSeek's
      EAGER content-sparse attention spike (the O(B*heads*T^2) ``combined_logits``
      at T~5000) -> OOM at the ``-.max()`` copy. Above the byte cap ``give``
      FREES the device tensor (PyTorch's caching allocator recycles that freed
      block on the next same-size ``take``, so homogeneous reuse is preserved)
      while still returning the pinned HOST buffer to its pool. The default cap
      (8 GiB) admits ~2 Qwen/122B layers but zero DeepSeek/397B (~13-15 GB)
      layers, which is exactly right: those big-layer stacks rely on the
      caching allocator, whose recycling is as cheap as an explicit pool.
    """

    def __init__(self, device, max_pooled_dev_bytes: int = 8 << 30):
        self.device = device
        self.host_free: dict[tuple, list[dict]] = {}
        self.dev_free: dict[tuple, list[dict]] = {}
        self.max_pooled_dev_bytes = max_pooled_dev_bytes
        self._dev_pooled_bytes = 0

    @staticmethod
    def signature(tensors: dict[str, torch.Tensor]) -> tuple:
        return tuple(sorted((n, tuple(t.shape), str(t.dtype))
                            for n, t in tensors.items()))

    def take(self, tensors: dict[str, torch.Tensor]) -> dict:
        sig = self.signature(tensors)
        host_pool = self.host_free.get(sig)
        if host_pool:
            host = host_pool.pop()
        else:
            host = {n: torch.empty_like(t, device="cpu", pin_memory=True)
                    for n, t in tensors.items()}
        dev_pool = self.dev_free.get(sig)
        if dev_pool:
            dev = dev_pool.pop()
            self._dev_pooled_bytes -= _dict_bytes(dev)
        else:
            dev = {n: torch.empty_like(t, device=self.device)
                   for n, t in tensors.items()}
        return {"host": host, "dev": dev, "sig": sig}

    def give(self, buf: dict) -> None:
        # Host pinned buffer: always reuse (host RAM abundant, pinning slow).
        self.host_free.setdefault(buf["sig"], []).append(buf["host"])
        # Device buffer: pool only under the byte cap, else drop (freed) so the
        # caching allocator recycles it instead of the pool hoarding it.
        dev = buf["dev"]
        b = _dict_bytes(dev)
        if self._dev_pooled_bytes + b <= self.max_pooled_dev_bytes:
            self.dev_free.setdefault(buf["sig"], []).append(dev)
            self._dev_pooled_bytes += b


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

    def quiesce(self) -> None:
        """Finish and evict every staged/in-flight evaluation page.

        Epoch-boundary evaluation enters with an empty rotator. This cleanup is
        the failure-path complement: even when a block forward raises after the
        next prefetch was started, no staged CUDA weight survives into teardown.
        """
        for layer in list(self._inflight):
            self.activate(layer)
        for layer in list(self._staged):
            self.evict(layer)

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
