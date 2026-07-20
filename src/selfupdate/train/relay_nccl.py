"""Native-InfiniBand relay transport for pipeline-v4 boundary tensors.

Owner decision 2026-07-18: after measuring the fabric (dual HDR-200 at
195.9/192.2 Gb/s node-to-node RDMA vs 0.4-1.2 GB/s single-stream Lustre)
and a history of Lustre metadata/page-fault stalls, the BULK relay
payloads move over NCCL — which speaks IB verbs natively (the broken
node-to-node IPoIB layer is irrelevant) and moves tensors GPU-to-GPU
(NVLink within a node). Boundary tensors and explicit shared-KV side channels
ride the main relay communicator. The synchronous live-owner battery uses its
own all-rank NCCL group. Same-node control and the store-fill audit trail
remain on the file relay.

Design facts this module leans on:

* Shapes are DERIVABLE by the receiver: stage k+1 holds the same cohort
  objects as stage k, so a boundary for cohort idx has the locally known
  hidden shape [B, T(, hc_mult), H]. The loaded block topology also derives
  which shared-KV types exist at each cut and their [B, n_kv, T, head_dim]
  shapes — no per-message header, no pickling.
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
import time

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
        self._in: dict[int, list[dict]] = {}
        self._shared_in_types: tuple[str, ...] = ()
        self._shared_out_types: tuple[str, ...] = ()
        self._shared_heads = 0
        self._shared_head_dim = 0

        # Out-of-band readiness gate (fix, 2026-07-20; issues.md "OPEN --
        # DeepSeek-V4-Flash PPP8 cross-node NCCL hang", repro in
        # scripts/relay_nccl_hang_repro.py). Store-fill and epoch-relay
        # traffic share this ONE ordered, untagged channel per adjacent
        # pair, and NCCL matches send/recv strictly by call order. The
        # original diagnosis read the symptom as "irecv blocks waiting for
        # the peer's isend"; the repro shows the primitive does NOT block --
        # every post_recv/is_completed poll returns in milliseconds
        # regardless of the peer's state. The REAL hazard is a silent
        # ORDER-BASED MISMATCH: if a successor posts its epoch-relay irecv
        # burst (post_recv, below) before its predecessor has issued every
        # one of its own store-fill sends, those irecvs get satisfied by
        # LEFTOVER store-fill tensors, not the real boundary -- confirmed by
        # the repro's controlled A/B (interleaved store-fill traffic during
        # the predecessor's stall reliably produces a data mismatch;
        # removing it does not). When a predecessor is stuck for a genuinely
        # long time -- the real DeepSeek PPP8 crash -- the resulting
        # backlog of never-matched irecvs is what eventually trips the NCCL
        # watchdog timeout, which is what reads as "hung" from the outside.
        # Fix: a TINY second TCPStore (never the file relay, which is
        # node-local /dev/shm and invisible cross-node; never the main NCCL
        # group, which is exactly the channel this needs to stay out of)
        # carrying one boolean per stage: "every store-fill send I will ever
        # issue toward stage+1 has been issued." A successor checks (or, for
        # a blocking drain, waits on) its predecessor's flag before ever
        # calling post_recv. is_master mirrors the main rendezvous: stage
        # 0's host is already the leader (MASTER_ADDR).
        # `or` (not `.get(key, default)`): the launcher forwards several of
        # these as explicitly-empty-string env vars when unset
        # (`VAR=${VAR:-}`), which `.get` would NOT treat as absent.
        master_port = int(os.environ.get("MASTER_PORT") or "29517")
        ready_port = int(os.environ.get("SELFUPDATE_V4_READY_PORT")
                         or str(master_port + 1))
        _dbg(f"pre ready-gate TCPStore on port {ready_port}")
        self._ready_store = dist.TCPStore(
            os.environ.get("MASTER_ADDR") or "127.0.0.1", ready_port,
            world_size=self.stages, is_master=(self.stage == 0),
            timeout=timeout, wait_for_workers=True)
        self._relay_ready_marked = False
        _dbg("post ready-gate TCPStore; NcclBoundaryRelay fully ready")

    def mark_relay_ready(self) -> None:
        """Announce: every store-fill send this rank will ever issue toward
        stage+1 has been issued (or, if store-fill never ran, that there was
        never any such traffic) -- safe for the successor to post its
        epoch-relay irecv burst without risking the order-based mismatch
        this gate exists to prevent. Idempotent."""
        if not self._relay_ready_marked:
            self._ready_store.set(f"relay_ready_{self.stage}", "1")
            self._relay_ready_marked = True

    def predecessor_relay_ready(self, pred_stage: int) -> bool:
        """Non-blocking (returns immediately either way): has stage
        ``pred_stage`` marked itself ready? Safe to call before every new
        epoch's first post_recv -- once true it stays true (store-fill runs
        at most once), so later epochs pay one cheap TCPStore round-trip
        and nothing else."""
        return self._ready_store.check([f"relay_ready_{pred_stage}"])

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

    def set_shared_kv_contract(self, *, incoming_types, outgoing_types,
                               num_key_value_heads: int,
                               head_dim: int) -> None:
        self._shared_in_types = tuple(sorted(incoming_types))
        self._shared_out_types = tuple(sorted(outgoing_types))
        self._shared_heads = int(num_key_value_heads)
        self._shared_head_dim = int(head_dim)

    def _shared_shape(self, cohort) -> tuple:
        return (len(cohort.indices), self._shared_heads, cohort.T,
                self._shared_head_dim)

    @staticmethod
    def _split_payload(payload):
        if torch.is_tensor(payload):
            return payload, {}
        return payload["h"], payload.get("shared_kv", {})

    # -- producer side ---------------------------------------------------

    def send_boundaries(self, epoch: int, out: dict) -> None:
        """isend every cohort boundary to stage+1 in index order; tensors
        are parked until the requests complete (reap_sent)."""
        for idx in range(len(self.cohorts)):
            hidden, shared = self._split_payload(out[idx])
            self.isend_one(hidden, dst=self.stage + 1)
            for layer_type in self._shared_out_types:
                if layer_type not in shared:
                    raise RuntimeError(
                        f"missing outgoing shared KV {layer_type!r} at "
                        f"stage {self.stage} cohort {idx}")
                key, value = shared[layer_type]
                self.isend_one(key, dst=self.stage + 1)
                self.isend_one(value, dst=self.stage + 1)
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
        payloads = []
        for cohort in self.cohorts:
            request, hidden = self.irecv_one(
                self._boundary_shape(cohort), src=self.stage - 1)
            shared = {}
            requests = [(request, hidden)]
            for layer_type in self._shared_in_types:
                key_request, key = self.irecv_one(
                    self._shared_shape(cohort), src=self.stage - 1)
                value_request, value = self.irecv_one(
                    self._shared_shape(cohort), src=self.stage - 1)
                requests.extend(((key_request, key),
                                 (value_request, value)))
                shared[layer_type] = (key, value)
            payloads.append({"requests": requests, "h": hidden,
                             "shared_kv": shared})
        self._in[epoch] = payloads

    def ready(self, epoch: int) -> bool:
        payloads = self._in.get(epoch)
        return (payloads is not None
                and all(request.is_completed()
                        for payload in payloads
                        for request, _ in payload["requests"]))

    def take(self, epoch: int, block: bool) -> dict | None:
        payloads = self._in.get(epoch)
        if payloads is None:
            return None
        if block:
            for payload in payloads:
                for request, _ in payload["requests"]:
                    request.wait()
        elif not all(request.is_completed()
                     for payload in payloads
                     for request, _ in payload["requests"]):
            return None
        del self._in[epoch]
        return {
            idx: {"h": payload["h"],
                  "shared_kv": payload["shared_kv"]}
            for idx, payload in enumerate(payloads)
        }

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
        self._shared_in_types: tuple[str, ...] = ()
        self._shared_out_types: tuple[str, ...] = ()
        # Shared with the readiness-gate poll below (recv_forward): the same
        # budget the main NCCL group already uses, so one knob still governs
        # "how long do we tolerate a stalled peer" everywhere.
        self._ready_timeout_s = int(cfg.train.v4_nccl_timeout_s)
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

    def set_shared_kv_contract(self, *, incoming_types, outgoing_types,
                               num_key_value_heads: int,
                               head_dim: int) -> None:
        self._shared_in_types = tuple(sorted(incoming_types))
        self._shared_out_types = tuple(sorted(outgoing_types))
        if self.nccl is not None:
            self.nccl.set_shared_kv_contract(
                incoming_types=incoming_types,
                outgoing_types=outgoing_types,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim)

    @staticmethod
    def _split_payload(payload):
        if torch.is_tensor(payload):
            return payload, {}
        return payload["h"], payload.get("shared_kv", {})

    @classmethod
    def _flatten_payload(cls, prefix: str, payload) -> dict:
        hidden, shared = cls._split_payload(payload)
        flat = {f"{prefix}h": hidden.detach().cpu()}
        for layer_type, (key, value) in sorted(shared.items()):
            flat[f"{prefix}kv.{layer_type}.k"] = key.detach().cpu()
            flat[f"{prefix}kv.{layer_type}.v"] = value.detach().cpu()
        return flat

    @staticmethod
    def _unflatten_payload(loaded: dict, prefix: str) -> dict:
        shared = {}
        marker = f"{prefix}kv."
        for key in loaded:
            if key.startswith(marker) and key.endswith(".k"):
                layer_type = key[len(marker):-2]
                shared[layer_type] = (
                    loaded[key], loaded[f"{marker}{layer_type}.v"])
        return {"h": loaded[f"{prefix}h"], "shared_kv": shared}

    def mark_relay_ready(self) -> None:
        """Announce that every store-fill send this stage will ever issue
        toward stage+1 (or, if store-fill never ran, that there was never
        any such traffic) is done -- safe for stage+1 to post its
        epoch-relay irecv burst. Call once, right after store-fill returns
        (or, when there is no store-fill, at the same point the epoch loop
        would otherwise begin). No-op for the same-node file transport: its
        consumer polls file existence, which carries no order-based
        mismatch risk. See NcclBoundaryRelay.mark_relay_ready."""
        if self.nccl is not None:
            self.nccl.mark_relay_ready()

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
            tensors = {}
            for idx, payload in out.items():
                tensors.update(self._flatten_payload(
                    f"c{idx}.", payload))
            self.rf.write(
                self.rf.path(epoch, f"stage{self.stage}.st"),
                tensors,
                stage=self.stage, epoch=epoch, to_stage=self.stage + 1)

    def recv_forward(self, epoch: int, block: bool) -> dict | None:
        """-> {idx: tensor on device}, or None when non-blocking and the
        boundary has not arrived yet."""
        if self.up_remote():
            if epoch not in self.nccl._in:
                # Readiness gate (2026-07-20 fix): never post_recv (the
                # irecv burst) until the predecessor has confirmed every one
                # of its store-fill sends toward us is already issued --
                # otherwise these irecvs can be satisfied by LEFTOVER
                # store-fill data instead of the real boundary (order-based
                # mismatch on the one shared, untagged channel; see
                # NcclBoundaryRelay.mark_relay_ready and the repro in
                # scripts/relay_nccl_hang_repro.py). Non-blocking callers
                # just see "not ready yet" and retry later, exactly like an
                # unarrived boundary; a blocking drain really waits for the
                # predecessor instead of posting early.
                if not self.nccl.predecessor_relay_ready(self.stage - 1):
                    if not block:
                        return None
                    self._wait_predecessor_relay_ready(self.stage - 1)
                self.nccl.post_recv(epoch)
            return self.nccl.take(epoch, block=block)
        path = self.rf.path(epoch, f"stage{self.stage - 1}.st")
        if not path.exists():
            if not block:
                return None
            self.rf.wait(path)
        loaded = self.rf.read(path, expect_epoch=epoch, as_stage=self.stage,
                              expect_stage=self.stage - 1)
        path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            path.parent.rmdir()
        indices = sorted({
            int(key.split(".", 1)[0][1:]) for key in loaded
            if key.startswith("c") and "." in key
        })
        return {idx: self._unflatten_payload(loaded, f"c{idx}.")
                for idx in indices}

    def _wait_predecessor_relay_ready(self, pred_stage: int,
                                      poll_s: float = 2.0) -> None:
        """Blocking counterpart to the readiness gate, for a drain(): really
        wait for the predecessor rather than posting our irecvs early. Polls
        the (non-blocking) check in a loop instead of a raw
        ``TCPStore.wait()`` -- a raw wait would block the FULL
        v4_nccl_timeout_s (up to 1800s) with no chance to notice a
        sibling's cooperative stop, exactly the failure class drain()
        exists to survive (the 2026-07-18 e500 finale: a blocked wait that
        outlived a stopped sibling cost three stages their checkpoints).
        Mirrors _RelayFiles.wait()'s own stop-aware polling so both halves
        of the boundary transport honor a stop the same way."""
        from .online_v4 import _RelayStopped
        from .stop import stop_requested
        deadline = time.monotonic() + self._ready_timeout_s
        while not self.nccl.predecessor_relay_ready(pred_stage):
            if stop_requested() or (self.rf is not None
                                    and self.rf.stop_seen()):
                raise _RelayStopped(
                    f"waiting for stage {pred_stage} relay-ready")
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"relay-ready timeout after {self._ready_timeout_s:.0f}s"
                    f" waiting for stage {pred_stage} to mark its "
                    "store-fill sends done; it may have crashed or "
                    "stalled -- inspect its log")
            time.sleep(poll_s)

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
    def send_fill_one(self, idx: int, t, epoch: int = 0) -> None:
        if self.fwd_remote():
            hidden, shared = self._split_payload(t)
            self.nccl.isend_one(hidden, dst=self.stage + 1)
            for layer_type in self._shared_out_types:
                if layer_type not in shared:
                    raise RuntimeError(
                        f"missing outgoing shared KV {layer_type!r} at "
                        f"stage {self.stage} cohort {idx}")
                key, value = shared[layer_type]
                self.nccl.isend_one(key, dst=self.stage + 1)
                self.nccl.isend_one(value, dst=self.stage + 1)
        else:
            self.rf.write(
                self.rf.path(epoch, f"capture_c{idx:04d}_stage{self.stage}.st"),
                self._flatten_payload("", t),
                stage=self.stage, epoch=epoch,
                to_stage=self.stage + 1)

    def recv_fill_one(self, idx: int, cohort, epoch: int = 0):
        if self.up_remote():
            req, hidden = self.nccl.irecv_one(
                self.nccl._boundary_shape(cohort), src=self.stage - 1)
            req.wait()
            shared = {}
            for layer_type in self._shared_in_types:
                key_req, key = self.nccl.irecv_one(
                    self.nccl._shared_shape(cohort), src=self.stage - 1)
                value_req, value = self.nccl.irecv_one(
                    self.nccl._shared_shape(cohort), src=self.stage - 1)
                key_req.wait()
                value_req.wait()
                shared[layer_type] = (key, value)
            return {"h": hidden, "shared_kv": shared}
        path = self.rf.wait(self.rf.path(
            epoch, f"capture_c{idx:04d}_stage{self.stage - 1}.st"))
        loaded = self.rf.read(path, expect_epoch=epoch, as_stage=self.stage,
                              expect_stage=self.stage - 1)
        path.unlink(missing_ok=True)
        return self._unflatten_payload(loaded, "")

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
        tok = self.rf.read(path, expect_epoch=0, as_stage=0,
                           expect_stage=self.stages - 1)["tok"]
        path.unlink(missing_ok=True)
        return tok.to(self.device)
