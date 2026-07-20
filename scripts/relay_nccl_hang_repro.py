#!/usr/bin/env python3
"""Minimal 2-rank cross-node repro for the DeepSeek-V4-Flash PPP8 NCCL hang
(issues.md, "OPEN — DeepSeek-V4-Flash PPP8 cross-node NCCL hang", 2026-07-19).

Exercises the REAL `BoundaryTransport`/`NcclBoundaryRelay` from
`selfupdate.train.relay_nccl` directly — no model, no dataset, no DeepSeek —
with synthetic tensors, across two real nodes, over the SAME NCCL process
group shape the trainer uses (store-fill traffic and epoch-relay traffic on
one shared communicator between an adjacent stage pair).

Mechanism under test: stage k+1 ("fast", done with its own work) calls
``recv_forward(epoch, block=False)`` — designed to return ``None``
immediately when the boundary has not arrived — while stage k ("slow", still
mid store-fill and has not reached its epoch-relay send at all) has issued
no matching send. Does the *call itself* return promptly (correct,
non-blocking contract honored), or does it block for as long as the slow
peer takes (the bug)?

Usage (run once per node, same launch id, opposite --rank):

    # on agpuh01 (stage 0, plays the SLOW predecessor, e.g. real stage3):
    scripts/relay_nccl_hang_repro.py --rank 0 \
        --peer-host agpuh02 --sender-delay-s 25 --warm-n 8

    # on agpuh02 (stage 1, plays the FAST successor, e.g. real stage4):
    scripts/relay_nccl_hang_repro.py --rank 1 \
        --peer-host agpuh02 --sender-delay-s 25 --warm-n 8

``scripts/relay_nccl_hang_repro_launch.sh`` drives both sides over ssh from
one shell and prints a verdict.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class _TrainNS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _CfgNS:
    def __init__(self, train):
        self.train = train


class _Cohort:
    """Just enough surface for NcclBoundaryRelay._boundary_shape."""
    def __init__(self, b: int, t: int):
        self.indices = list(range(b))
        self.T = t


def log(rank: int, msg: str) -> None:
    print(f"[repro rank{rank} {socket.gethostname()} "
          f"t={time.perf_counter():.3f}] {msg}", file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True, choices=[0, 1])
    ap.add_argument("--peer-host", required=True,
                     help="short hostname of the OTHER rank (for the "
                          "SELFUPDATE_V4_STAGE_HOSTS co-location map)")
    ap.add_argument("--this-host", default=socket.gethostname().split(".")[0])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--warm-n", type=int, default=8,
                     help="matched store-fill-style sends/recvs BEFORE the "
                          "test, to prove the channel is already warm "
                          "(rules out a cold-connect artifact)")
    ap.add_argument("--sender-delay-s", type=float, default=25.0,
                     help="rank0 (slow predecessor) sleeps this long, still "
                          "issuing MORE store-fill-style sends, before it "
                          "ever calls send_forward (the epoch-relay send)")
    ap.add_argument("--nccl-timeout-s", type=int, default=90)
    ap.add_argument("--poll-interval-s", type=float, default=1.0)
    ap.add_argument("--extra-fill-during-delay", type=int, default=1,
                     help="control knob: 1 = predecessor keeps issuing "
                          "MORE store-fill-style sends during its stall "
                          "(the real DeepSeek shape); 0 = predecessor just "
                          "sleeps, issuing nothing, before its true send "
                          "(isolates whether corruption needs interleaved "
                          "same-phase traffic)")
    ap.add_argument("--post-ready-delay-s", type=float, default=0.0,
                     help="predecessor: extra sleep AFTER mark_relay_ready() "
                          "but BEFORE the true send_forward (simulates slow "
                          "local epoch training after store-fill)")
    ap.add_argument("--launch-id", default=os.environ.get(
        "SELFUPDATE_V4_LAUNCH_ID", "relay-repro"))
    ap.add_argument("--hidden", type=int, default=8)
    args = ap.parse_args()

    import torch
    from selfupdate.train.relay_nccl import BoundaryTransport

    os.environ["SELFUPDATE_V4_LAUNCH_ID"] = args.launch_id
    hosts = ([args.this_host, args.peer_host] if args.rank == 0
             else [args.peer_host, args.this_host])
    os.environ["SELFUPDATE_V4_STAGE_HOSTS"] = " ".join(hosts)
    os.environ.setdefault("SELFUPDATE_V4_CROSS_NODE", "1")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    train = _TrainNS(v4_stage=args.rank, v4_stage_splits=[1],
                     v4_nccl_timeout_s=args.nccl_timeout_s,
                     v4_relay_transport="auto")
    cfg = _CfgNS(train)

    cohorts = [_Cohort(2, 4) for _ in range(64)]
    log(args.rank, f"constructing BoundaryTransport (stages=2, "
        f"hosts={hosts}, timeout={args.nccl_timeout_s}s) ...")
    t0 = time.perf_counter()
    bt = BoundaryTransport(cfg, rf=None, cohorts=cohorts, device=device,
                          launch_id=args.launch_id)
    bt.set_hidden_dims((args.hidden,), torch.bfloat16)
    log(args.rank, f"BoundaryTransport ready in {time.perf_counter()-t0:.3f}s"
        f" (this proves group init + launch-id all_gather both crossed "
        f"nodes fine)")

    # -- Phase 1: warm the channel exactly like real store-fill: paced,
    #    one-at-a-time, immediate .wait(). Both ranks participate the same
    #    way store-fill does (send_fill_one / recv_fill_one).
    if args.rank == 0:
        for idx in range(args.warm_n):
            t = torch.full((2, 4, args.hidden), float(idx),
                          dtype=torch.bfloat16, device=device)
            bt.send_fill_one(idx, t)
        bt.reap_forward(block=True)
        log(args.rank, f"warm-up: sent {args.warm_n} store-fill-style "
            f"tensors, all reaped")
    else:
        for idx in range(args.warm_n):
            got = bt.recv_fill_one(idx, cohorts[idx])
            assert float(got[0, 0, 0]) == float(idx), "warm-up data mismatch"
        log(args.rank, f"warm-up: received+verified {args.warm_n} "
            f"store-fill-style tensors — channel is proven warm")

    bt.nccl.dist.barrier()
    log(args.rank, "post warm-up barrier cleared; entering test phase")

    # -- Phase 2: the actual test.
    if args.rank == 0:
        # SLOW predecessor (plays real stage3): keeps doing MORE
        # store-fill-style traffic for --sender-delay-s BEFORE it EVER
        # calls send_forward. This is the "has not reached its epoch loop
        # at all" state from the diagnosis.
        log(args.rank, f"SLOW predecessor: simulating "
            f"{args.sender_delay_s:.1f}s of continued store-fill (NOT yet "
            f"at the epoch-relay send) ...")
        extra = 0
        deadline = time.perf_counter() + args.sender_delay_s
        while time.perf_counter() < deadline:
            if args.extra_fill_during_delay:
                t = torch.zeros((2, 4, args.hidden), dtype=torch.bfloat16,
                                device=device)
                bt.send_fill_one(args.warm_n + extra, t)
                bt.throttle_sends(2)
                extra += 1
            time.sleep(0.5)
        bt.reap_forward(block=True)
        log(args.rank, f"predecessor done stalling ({extra} more fill sends "
            f"issued). Marking relay-ready NOW (mirrors the real "
            f"capture_relay_store()-returns call site) ...")
        bt.mark_relay_ready()
        if args.post_ready_delay_s > 0:
            log(args.rank, f"(still not sending yet -- simulating "
                f"{args.post_ready_delay_s:.1f}s of local epoch training "
                f"AFTER store-fill/ready but BEFORE the real relay send)")
            time.sleep(args.post_ready_delay_s)
        log(args.rank, "NOW calling send_forward(epoch=0) — the first real "
            "epoch-relay send.")
        out = {i: torch.full((2, 4, args.hidden), 99.0, dtype=torch.bfloat16,
                             device=device) for i in range(len(cohorts))}
        bt.send_forward(0, out)
        bt.reap_forward(block=True)
        log(args.rank, "predecessor: send_forward(epoch=0) issued and "
            "reaped")
    else:
        # FAST successor (plays real stage4): immediately (right after the
        # warm-up barrier) tries the exact call the real bug hangs in —
        # recv_forward(epoch, block=False) via submit()/service(block=False)
        # — while the predecessor is STILL stalling. Every poll is timed
        # individually: if any single call takes anywhere near
        # --sender-delay-s, the "block=False is defeated" claim is
        # CONFIRMED. If every poll returns in milliseconds regardless of
        # the peer's state, the primitive is NOT the culprit.
        log(args.rank, "FAST successor: predecessor is still stalling. "
            "Calling recv_forward(epoch=0, block=False) NOW (this is "
            "exactly _RelayServicer.submit()->service(block=False)) ...")
        worst = 0.0
        polls = 0
        result = None
        overall_start = time.perf_counter()
        while result is None:
            call_t0 = time.perf_counter()
            result = bt.recv_forward(0, block=False)
            call_dt = time.perf_counter() - call_t0
            worst = max(worst, call_dt)
            polls += 1
            posted = 0 in bt.nccl._in
            log(args.rank, f"  poll #{polls}: recv_forward(block=False) "
                f"took {call_dt:.3f}s, returned "
                f"{'DATA' if result is not None else 'None (not ready)'} "
                f"[post_recv posted={posted}, "
                f"predecessor_ready={bt.nccl.predecessor_relay_ready(0)}]")
            if result is None:
                time.sleep(args.poll_interval_s)
        total = time.perf_counter() - overall_start
        log(args.rank, f"VERDICT: worst single recv_forward(block=False) "
            f"call = {worst:.3f}s over {polls} polls "
            f"(total wall time to get data: {total:.3f}s, predecessor "
            f"delay was {args.sender_delay_s:.1f}s)")
        if worst > 2.0:
            log(args.rank, "CONFIRMED: a single 'non-blocking' "
                "recv_forward(block=False) call itself blocked for "
                f"{worst:.3f}s -- the block=False contract is defeated "
                "exactly as issues.md diagnosed.")
        else:
            log(args.rank, "NOT CONFIRMED at this primitive: every "
                "recv_forward(block=False) call returned promptly "
                "(<2s) even though the peer had not sent yet. The "
                "hang, if reproducible, must live elsewhere (drain()'s "
                "blocking service(block=True), or the teardown barrier).")
        assert float(result[0][0, 0, 0]) == 99.0, "relay data mismatch"

    bt.barrier()
    log(args.rank, "final barrier cleared, exiting cleanly")


if __name__ == "__main__":
    main()
