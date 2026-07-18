"""Native-InfiniBand relay transport for pipeline-v4 boundary tensors.

Owner decision 2026-07-18: after measuring the fabric (dual HDR-200 at
195.9/192.2 Gb/s node-to-node RDMA vs 0.4-1.2 GB/s single-stream Lustre)
and a history of Lustre metadata/page-fault stalls, the BULK relay
payloads move over NCCL — which speaks IB verbs natively (the broken
node-to-node IPoIB layer is irrelevant) and moves tensors GPU-to-GPU
(NVLink within a node). Only the boundary tensors ride NCCL; the small
control-plane envelopes (adapter publications, battery acks, capture
store) stay on the file relay, whose on-disk audit trail the reaper and
post-mortems depend on.

Design facts this module leans on:

* Shapes are DERIVABLE by the receiver: stage k+1 holds the same cohort
  objects as stage k, so a boundary for cohort idx has the locally known
  shape [B, T(, hc_mult), H] — no per-message header, no pickling.
* NCCL send/recv pairs match by ORDER per (src, dst) rank pair (the
  backend has no tags): one epoch relays at a time per pair and cohorts
  go in index order, so ordering is the addressing.
* Launch identity is asserted ONCE: a gloo all_gather of the launch-id
  hash at group formation replaces the per-file envelope check.
* The file relay's asynchrony (producers running ahead of consumers) is
  preserved with isend/irecv: requests are posted immediately and
  completed lazily; the sender parks the tensors until completion.

Bootstrap: MASTER_ADDR/MASTER_PORT over the Ethernet management network
(IPoIB cannot carry the TCP rendezvous here); NCCL_IB_HCA/
NCCL_SOCKET_IFNAME are exported by scripts/launch_v4_stages.sh.
"""

from __future__ import annotations

import datetime
import hashlib
import os

import torch


class NcclBoundaryRelay:
    """Order-addressed boundary exchange between adjacent stage ranks."""

    def __init__(self, cfg, cohorts, device, launch_id: str):
        import torch.distributed as dist

        self.dist = dist
        self.cfg = cfg
        self.cohorts = cohorts
        self.device = device
        self.stage = cfg.train.v4_stage
        self.stages = len(cfg.train.v4_stage_splits or []) + 1
        timeout = datetime.timedelta(
            seconds=int(cfg.train.v4_nccl_timeout_s))
        if not dist.is_initialized():
            # MASTER_ADDR/MASTER_PORT come from the launcher (stage-0
            # host's Ethernet address).
            dist.init_process_group(
                "nccl", rank=self.stage, world_size=self.stages,
                timeout=timeout,
                device_id=torch.device(device))
        self.gloo = dist.new_group(backend="gloo", timeout=timeout)
        # One-time identity assertion — replaces the per-file envelope.
        digest = int(hashlib.sha256(launch_id.encode()).hexdigest()[:15],
                     16)
        mine = torch.tensor([digest], dtype=torch.long)
        gathered = [torch.zeros_like(mine) for _ in range(self.stages)]
        dist.all_gather(gathered, mine, group=self.gloo)
        if any(int(g) != digest for g in gathered):
            raise RuntimeError(
                "nccl relay group spans different launch identities — a "
                "stale stage process joined the rendezvous")
        self._out: list[tuple[object, torch.Tensor]] = []
        self._in: dict[int, list[tuple[object, torch.Tensor]]] = {}

    def _boundary_shape(self, cohort) -> tuple:
        hidden = self._hidden_dims
        return (len(cohort.indices), cohort.T, *hidden)

    @property
    def _hidden_dims(self) -> tuple:
        # Set once by the servicer from a real boundary (mHC models carry
        # [hc_mult, H] trailing dims, others [H]).
        return self._dims

    def set_hidden_dims(self, dims: tuple, dtype) -> None:
        self._dims = tuple(dims)
        self._dtype = dtype

    # -- producer side ---------------------------------------------------

    def send_boundaries(self, epoch: int, out: dict) -> None:
        """isend every cohort boundary to stage+1 in index order; tensors
        are parked until the requests complete (reap_sent)."""
        for idx in range(len(self.cohorts)):
            t = out[idx]
            if t.device.type != "cuda":
                t = t.to(self.device, non_blocking=False)
            t = t.contiguous()
            req = self.dist.isend(t, dst=self.stage + 1)
            self._out.append((req, t))
        self.reap_sent(block=False)

    def reap_sent(self, block: bool) -> None:
        still = []
        for req, t in self._out:
            if block:
                req.wait()
            elif not req.is_completed():
                still.append((req, t))
        self._out = still if not block else []

    # -- consumer side ---------------------------------------------------

    def post_recv(self, epoch: int) -> None:
        """Post irecvs for one epoch's boundaries from stage-1 (shapes
        derived locally)."""
        reqs = []
        for cohort in self.cohorts:
            t = torch.empty(self._boundary_shape(cohort),
                            dtype=self._dtype, device=self.device)
            reqs.append((self.dist.irecv(t, src=self.stage - 1), t))
        self._in[epoch] = reqs

    def ready(self, epoch: int) -> bool:
        reqs = self._in.get(epoch)
        return reqs is not None and all(r.is_completed() for r, _ in reqs)

    def take(self, epoch: int, block: bool) -> dict | None:
        reqs = self._in.get(epoch)
        if reqs is None:
            return None
        if block:
            for r, _ in reqs:
                r.wait()
        elif not all(r.is_completed() for r, _ in reqs):
            return None
        del self._in[epoch]
        return {idx: t for idx, (_, t) in enumerate(reqs)}
