"""Native-InfiniBand relay transport for pipeline-v4 boundary tensors.

Owner decision 2026-07-18: after measuring the fabric (dual HDR-200 at
195.9/192.2 Gb/s node-to-node RDMA vs 0.4-1.2 GB/s single-stream Lustre)
and a history of Lustre metadata/page-fault stalls, the BULK relay
payloads move over NCCL — which speaks IB verbs natively (the broken
node-to-node IPoIB layer is irrelevant) and moves tensors GPU-to-GPU
(NVLink within a node). Only the boundary tensors ride NCCL; the small
control-plane envelopes (adapter publications, battery acks, store-fill
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

import contextlib
import datetime
import hashlib
import os

import torch


def resolve_relay_transport(cfg) -> str:
    """auto -> nccl whenever the stage set spans hosts (the launcher
    exports SELFUPDATE_V4_CROSS_NODE=1 from its STAGE_HOSTS map), files
    for a node-local stage set (/dev/shm exchange, RAM-speed). Cross-node
    boundary mail NEVER rides Lustre (owner decision 2026-07-18)."""
    choice = cfg.train.v4_relay_transport
    if choice != "auto":
        return choice
    return ("nccl" if os.environ.get("SELFUPDATE_V4_CROSS_NODE") == "1"
            else "files")


class NcclBoundaryRelay:
    """Order-addressed boundary exchange between adjacent stage ranks."""

    def __init__(self, cfg, cohorts, device, launch_id: str):
        import sys as _sys
        import torch.distributed as dist

        def _dbg(msg):
            # Cross-node NCCL bring-up trace (init_process_group / all_gather
            # hangs are the common failure); off unless explicitly enabled.
            if os.environ.get("SELFUPDATE_V4_RELAY_DEBUG"):
                print(f"V4DBG nccl stage{cfg.train.v4_stage}: {msg}",
                      file=_sys.stderr, flush=True)

        _dbg("enter NcclBoundaryRelay.__init__")
        self.dist = dist
        self.cfg = cfg
        self.cohorts = cohorts
        self.device = device
        self.stage = cfg.train.v4_stage
        self.stages = len(cfg.train.v4_stage_splits or []) + 1
        timeout = datetime.timedelta(
            seconds=int(cfg.train.v4_nccl_timeout_s))
        _dbg(f"pre init_process_group rank={self.stage}/{self.stages} "
             f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
             f"init'd={dist.is_initialized()}")
        if not dist.is_initialized():
            # MASTER_ADDR/MASTER_PORT come from the launcher (stage-0
            # host's Ethernet address).
            dist.init_process_group(
                "nccl", rank=self.stage, world_size=self.stages,
                timeout=timeout,
                device_id=torch.device(device))
        _dbg("post init_process_group; pre all_gather(launch-id)")
        # One-time identity assertion over the NCCL group itself (CUDA
        # tensors). A gloo subgroup here hung cross-node in a connect-retry
        # loop even with GLOO_SOCKET_IFNAME pinned (measured 2026-07-19): NCCL
        # init completed, then dist.new_group(backend="gloo") never rendezvoused.
        # NCCL/IB is already up and proven, so reuse it instead of standing up a
        # second (gloo/TCP) transport that needs its own working interface.
        digest = int(hashlib.sha256(launch_id.encode()).hexdigest()[:15],
                     16)
        mine = torch.tensor([digest], dtype=torch.long, device=device)
        gathered = [torch.zeros_like(mine) for _ in range(self.stages)]
        dist.all_gather(gathered, mine)
        if any(int(g.item()) != digest for g in gathered):
            raise RuntimeError(
                "nccl relay group spans different launch identities — a "
                "stale stage process joined the rendezvous")
        _dbg("post all_gather; NcclBoundaryRelay ready")
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

    # -- per-cohort (store-fill) and loop-back (eval) point-to-point --------
    # The store fill sends boundaries one cohort at a time in index order; the
    # eval decode loops the sampled token back from the last stage to stage 0.
    # NCCL matches by ORDER per (src,dst) pair, so the (stage,stage+1) forward
    # order (fill cohorts, then each epoch's cohorts) and the (N-1,0) loop-back
    # order (token 0,1,2,...) are each self-consistent and never collide.

    def isend_one(self, t: torch.Tensor, dst: int) -> None:
        if t.device.type != "cuda":
            t = t.to(self.device, non_blocking=False)
        t = t.contiguous()
        self._out.append((self.dist.isend(t, dst=dst), t))

    def irecv_one(self, shape: tuple, src: int):
        t = torch.empty(shape, dtype=self._dtype, device=self.device)
        return self.dist.irecv(t, src=src), t

    def send_token(self, tok: torch.Tensor, dst: int) -> None:
        """Blocking small send of the sampled token ids (eval loop-back)."""
        self.dist.send(tok.to(self.device).contiguous(), dst=dst)

    def recv_token(self, batch: int, src: int) -> torch.Tensor:
        t = torch.empty((batch,), dtype=torch.long, device=self.device)
        self.dist.recv(t, src=src)
        return t


def parse_stage_hosts() -> list:
    """Stage -> short hostname map from the launcher (already resolved: no
    'local' entries). Empty list for a single-node launch."""
    return os.environ.get("SELFUPDATE_V4_STAGE_HOSTS", "").split()


class BoundaryTransport:
    """Co-location-routed stage-boundary exchange (owner 2026-07-19).

    A SAME-NODE consumer gets a /dev/shm file (RAM, via the file relay ``rf``);
    a CROSS-NODE consumer gets an NCCL/IB send. No comms path ever touches a
    disk filesystem (ssd/lustre/nfs). Serves the forward pipe (k -> k+1) for
    the epoch relay, store-fill, and eval decode, plus the eval token loop-back
    (last stage N-1 -> stage 0).

    The NCCL group (init_process_group) forms COLLECTIVELY across ALL stages
    when this is constructed with transport==nccl, even the stages whose
    neighbours are all same-node — the group init is a barrier, so every stage
    must reach it. Only the stage pairs that actually straddle a node boundary
    then exchange over it; the rest use shm.
    """

    def __init__(self, cfg, rf, cohorts, device, launch_id: str):
        self.cfg = cfg
        self.rf = rf
        self.device = device
        self.cohorts = cohorts
        self.stage = cfg.train.v4_stage
        self.stages = len(cfg.train.v4_stage_splits or []) + 1
        self.hosts = parse_stage_hosts()
        self.transport = resolve_relay_transport(cfg)
        self.nccl = None
        if self.transport == "nccl":
            self.nccl = NcclBoundaryRelay(cfg, cohorts, device, launch_id)

    # -- co-location ------------------------------------------------------
    def same_node(self, a: int, b: int) -> bool:
        if not self.hosts or not (0 <= a < len(self.hosts)) \
                or not (0 <= b < len(self.hosts)):
            return True                      # no map => single node
        return self.hosts[a] == self.hosts[b]

    def fwd_remote(self) -> bool:
        """Forward consumer (stage+1) on another node?"""
        return (self.nccl is not None and self.stage + 1 < self.stages
                and not self.same_node(self.stage, self.stage + 1))

    def up_remote(self) -> bool:
        """Forward producer (stage-1) on another node?"""
        return (self.nccl is not None and self.stage - 1 >= 0
                and not self.same_node(self.stage - 1, self.stage))

    def set_hidden_dims(self, dims: tuple, dtype) -> None:
        if self.nccl is not None:
            self.nccl.set_hidden_dims(dims, dtype)

    def barrier(self) -> None:
        """Synchronize ALL ranks before run teardown. Without this a fast
        stage (few owned layers, no eval tail) finishes its epochs and exits,
        destroying its NCCL rank while a slow sibling (e.g. the last stage's
        eval tail) is still mid-relay — the surviving ranks then time out on a
        collective and the whole set dumps + dies (DeepSeek PPP8 finalize crash,
        task #24, 2026-07-19). The barrier holds every rank until the slowest
        finishes, so the group is torn down symmetrically. No-op single-node."""
        if self.nccl is not None:
            self.nccl.reap_sent(block=True)   # flush any parked isends first
            self.nccl.dist.barrier()

    # -- forward pipe: epoch relay (all cohorts per epoch) ---------------
    def send_forward(self, epoch: int, out: dict) -> None:
        """out: {idx: boundary tensor on device}. Cross-node -> NCCL, else
        a /dev/shm file addressed to stage+1."""
        if self.fwd_remote():
            self.nccl.send_boundaries(epoch, out)
        else:
            self.rf.write(
                self.rf.path(epoch, f"stage{self.stage}.st"),
                {f"c{idx}": t.detach().cpu() for idx, t in out.items()},
                stage=self.stage, epoch=epoch, to_stage=self.stage + 1)

    def recv_forward(self, epoch: int, block: bool) -> dict | None:
        """-> {idx: tensor on device}, or None when non-blocking and the
        boundary has not arrived yet."""
        if self.up_remote():
            if epoch not in self.nccl._in:
                self.nccl.post_recv(epoch)
            return self.nccl.take(epoch, block=block)
        path = self.rf.path(epoch, f"stage{self.stage - 1}.st")
        if not path.exists():
            if not block:
                return None
            self.rf.wait(path)
        loaded = self.rf.read(path, expect_epoch=epoch, as_stage=self.stage)
        path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            path.parent.rmdir()
        return {int(k[1:]): v for k, v in loaded.items()}

    def reap_forward(self, block: bool) -> None:
        if self.nccl is not None:
            self.nccl.reap_sent(block=block)

    def throttle_sends(self, max_inflight: int) -> None:
        """Store-fill backpressure for the NCCL path: the parked isend tensors
        are boundary hiddens held on-device until the peer irecvs, so bound
        their count the way the file relay bounded in-flight files. Reap the
        completed, then block on the oldest until under the cap."""
        if self.nccl is None:
            return
        self.nccl.reap_sent(block=False)
        while len(self.nccl._out) > max_inflight:
            req, _t = self.nccl._out.pop(0)
            req.wait()
        self.nccl.reap_sent(block=False)

    # -- forward pipe: store-fill (one cohort at a time, index order) ----
    def send_fill_one(self, idx: int, t: torch.Tensor, epoch: int = 0) -> None:
        if self.fwd_remote():
            self.nccl.isend_one(t, dst=self.stage + 1)
        else:
            self.rf.write(
                self.rf.path(epoch, f"capture_c{idx:04d}_stage{self.stage}.st"),
                {"h": t.detach().cpu()}, stage=self.stage, epoch=epoch,
                to_stage=self.stage + 1)

    def recv_fill_one(self, idx: int, cohort, epoch: int = 0) -> torch.Tensor:
        if self.up_remote():
            req, t = self.nccl.irecv_one(
                self.nccl._boundary_shape(cohort), src=self.stage - 1)
            req.wait()
            return t
        path = self.rf.wait(self.rf.path(
            epoch, f"capture_c{idx:04d}_stage{self.stage - 1}.st"))
        loaded = self.rf.read(path, expect_epoch=epoch, as_stage=self.stage)
        path.unlink(missing_ok=True)
        return loaded["h"]

    # -- eval token loop-back (last stage N-1 -> stage 0) ----------------
    def send_token_home(self, tok: torch.Tensor) -> None:
        """Last stage: hand the sampled token to stage 0 (next input)."""
        if self.nccl is not None and not self.same_node(self.stages - 1, 0):
            self.nccl.send_token(tok, dst=0)
        else:
            self.rf.write(self.rf.path(0, "_loopback.st"), {"tok": tok.cpu()},
                          stage=self.stage, epoch=0, to_stage=0)

    def recv_token_home(self, batch: int) -> torch.Tensor:
        """Stage 0: receive the looped-back token from the last stage."""
        if self.nccl is not None and not self.same_node(self.stages - 1, 0):
            return self.nccl.recv_token(batch, src=self.stages - 1)
        path = self.rf.wait(self.rf.path(0, "_loopback.st"))
        tok = self.rf.read(path, expect_epoch=0, as_stage=0)["tok"]
        path.unlink(missing_ok=True)
        return tok.to(self.device)
