"""Block rotation for pipeline-v4 (plan B4, 2026-07-17).

A stage may own more layers than fit its card. Because v4 blocks are
gradient-independent and the walk is ``layer_major``, only ONE owned block
needs GPU residency at a time: page block L's frozen base weights host->GPU,
train every cohort while resident, evict (drop — frozen weights are never
written back), prefetch the next owned block on a side stream meanwhile.

The rotation UNIT is {frozen base weights (one-way H2D, CPU masters stay
mmap-backed on the /dev/shm snapshot) + Adam moments (paged BOTH ways with
their block — the owner's demonstration target: trivial bytes under LoRA,
but the exact machinery base-weight fine-tuning at 397B will need)}. LoRA
adapter params are GPU-resident always — they are the trainable state and
are tiny.

Cost at 397B: ~200 GB/stage/epoch of H2D, hidden behind minutes of
compute; ``item_major`` + rotate would page every block every cohort
(~4 TB/epoch) and is rejected by validate.
"""

from __future__ import annotations

import torch

__all__ = ["BlockRotator", "decide_rotate"]


def decide_rotate(owned_bytes: int, device, *, headroom: float = 1.25,
                  reserve_bytes: int = 20 << 30) -> bool:
    """auto policy: rotate when resident owned weights would not leave the
    activation/optimizer reserve free. ``reserve_bytes`` covers activations,
    teacher tensors, and eval headroom — deliberately generous; rotation's
    cost is near-zero under layer_major, so auto errs toward rotating."""
    free, _total = torch.cuda.mem_get_info(device)
    return owned_bytes * headroom + reserve_bytes > free


class BlockRotator:
    """Pages one owned block's rotation unit at a time.

    Contract: ``activate(L)`` blocks until L's weights are on the device
    and swaps them live; ``evict(L)`` restores the CPU masters (dropping
    the GPU copies) and pages L's optimizer moments out; ``prefetch(L)``
    starts L's H2D on the side stream without swapping. All swaps are
    ``p.data`` replacement — module structure, autograd graphs, and
    optimizer param identity are untouched.
    """

    def __init__(self, stack, owned, device, optimizers=None):
        self.stack = stack
        self.device = torch.device(device)
        self.optimizers = optimizers or {}
        self.stream = torch.cuda.Stream(self.device)
        # CPU masters: the frozen base tensors of owned blocks that the
        # stage-scoped load left on host (mmap-backed). Trainable params
        # (LoRA) and anything already on the device are NOT rotated.
        self.masters: dict[int, dict[str, torch.Tensor]] = {}
        self._staged: dict[int, dict[str, torch.Tensor]] = {}
        self._events: dict[int, torch.cuda.Event] = {}
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
                   for entry in self._staged.values()
                   for t in entry.values())

    def prefetch(self, L: int) -> None:
        if L not in self.masters or L in self._staged:
            return
        entry = {}
        with torch.cuda.stream(self.stream):
            for name, cpu_t in self.masters[L].items():
                # mmap-backed tensors are not pinned; non_blocking still
                # overlaps read-side page-in with compute on the default
                # stream. A pinned double-buffer is the measured follow-up
                # if H2D ever shows in prep_seconds.
                entry[name] = cpu_t.to(self.device, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(self.stream)
        self._staged[L] = entry
        self._events[L] = ev

    def _swap(self, L: int, tensors: dict[str, torch.Tensor]) -> None:
        block = self.stack.blocks[L - 1]
        params = dict(block.named_parameters())
        buffers = dict(block.named_buffers())
        for name, t in tensors.items():
            if name.startswith("buffer:"):
                buffers[name[len("buffer:"):]].data = t
            else:
                params[name].data = t

    def activate(self, L: int) -> None:
        if L not in self.masters:
            return  # fully resident block (or foreign): nothing to do
        self.prefetch(L)
        torch.cuda.current_stream(self.device).wait_event(self._events[L])
        self._swap(L, self._staged[L])
        self._moments_to(L, self.device)

    def evict(self, L: int) -> None:
        if L not in self.masters:
            return
        self._swap(L, self.masters[L])
        self._staged.pop(L, None)
        self._events.pop(L, None)
        self._moments_to(L, torch.device("cpu"))

    def _moments_to(self, L: int, device) -> None:
        """Adam moments ride their block. LoRA moments are tiny; the point
        is the mechanism (base-FT moments at 397B are 3.2 TB and MUST
        page). State dicts are keyed by param object — identity survives
        the .data swaps, so this is a plain tensor move."""
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
