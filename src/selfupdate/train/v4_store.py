"""Fill-once teacher store (store-fill) for pipeline-v4 (plan B5, 2026-07-17).

``v4_teacher_source: store`` — teacher hidden states are EPOCH-INVARIANT
(frozen teacher, frozen epoch-0 target semantics), so the per-epoch
per-epoch teacher recompute that dominated the measured 27B epoch (~136 s of ~198 s; see
docs/training_pipeline_v4.md timing table) is paid ONCE, before the epoch
loop, as a stage relay:

- stage 0 embeds each cohort's teacher ids and walks its owned layers
  adapters-off, filling the SAME per-(layer, cohort) store entries the
  training loop consumes (`_TeacherTensors.put`/`put_linear`: frozen KV +
  query-row inputs + targets for attention layers; full inputs + targets
  for linear-attention layers);
- it ships the boundary hidden to stage 1 through the existing postal
  exchange (launch-id envelopes, atomic rename, consumer deletes);
- stages fill pipelined: stage k+1 starts as soon as cohort 0's boundary
  arrives, while stage k is already capturing cohort 1. Backpressure keeps
  at most ``v4_capture_inflight`` boundary files of a producer alive.

Every epoch then runs with ZERO teacher forwards. MoE routing interventions
record teacher top-k in the same pass (the controller's teacher_phase).
Under weight rotation the staged store-fill pages blocks per (cohort, layer) — a
one-time cost of minutes at 397B, accepted to preserve the pipeline.
"""

from __future__ import annotations

import contextlib
import time

import torch

__all__ = ["capture_relay_store"]


def capture_relay_store(cfg, stack, ds, cohorts, tensors, adapters_off,
                        device, run_dir, log, *, owned, n_layers,
                        moe_ctrl=None, moe_routing=None,
                        teacher_eval_rows=None, rotator=None) -> None:
    from .online_v4 import _FrozenKV, _RelayFiles, _bk_layer_type

    stage = cfg.train.v4_stage
    stages = len(cfg.train.v4_stage_splits or []) + 1
    first = stage <= 0
    last = stage < 0 or stage == stages - 1
    rf = _RelayFiles(run_dir.parent) if run_dir is not None else None
    if rf is None and stage >= 0 and stages > 1:
        raise ValueError("staged store-fill relay needs run_dir for the "
                         "boundary exchange")
    inflight = max(1, int(getattr(cfg.train, "v4_capture_inflight", 2)))
    written: list = []
    n = n_layers
    started = time.perf_counter()
    entries = 0
    owned_list = list(owned)

    if stage < 0 and rotator is not None:
        # Rotary PPP1 (task #18): cohort-outer store-fill pages EVERY owned
        # block once per cohort — 130 cohorts x 30 blocks ≈ 6 TB of H2D
        # and 617 s of measured stall at 26B. A single process has no
        # pipelining reason to be cohort-outer: walk LAYER-outer instead
        # (one page-in per block, amortized over all cohorts; boundary
        # hiddens for every cohort stay resident, ~7-12 GB at 31B/122B).
        _capture_layer_outer(cfg, stack, cohorts, tensors, adapters_off,
                             device, log, owned=owned_list, n_layers=n,
                             moe_ctrl=moe_ctrl, moe_routing=moe_routing,
                             teacher_eval_rows=teacher_eval_rows,
                             rotator=rotator, started=started)
        return

    for idx, cohort in enumerate(cohorts):
        B, T = len(cohort.indices), cohort.T
        pos = torch.arange(T, device=device)[None].expand(B, -1)
        if first:
            h = stack.embed(cohort.teacher_ids.to(device))
        else:
            path = rf.wait(
                rf.path(0, f"capture_c{idx:04d}_stage{stage - 1}.st"))
            loaded = rf.read(path, expect_epoch=0, as_stage=stage)
            h = loaded["h"].to(device)
            path.unlink(missing_ok=True)
        ctx_a = (adapters_off() if adapters_off is not None
                 else contextlib.nullcontext())
        ctx_m = (moe_ctrl.teacher_phase() if moe_ctrl is not None
                 else contextlib.nullcontext())
        with torch.no_grad(), ctx_a, ctx_m:
            pe = stack.rope(h, pos)
            for layer in owned_list:
                if rotator is not None:
                    rotator.activate(layer)
                if _bk_layer_type(stack, layer) == "linear_attention":
                    h_in = h
                    h = stack.run_block(layer, h, pe, position_ids=pos)
                    view = h if layer < n else stack.final_norm(h)
                    tensors.put_linear(
                        layer, idx, h_in.clone(),
                        cohort.gather_query_inputs(view).clone())
                else:
                    kv = _FrozenKV()
                    inputs_q = cohort.gather_query_inputs(h).clone()
                    # One pass both produces h[L] (causal mask built from
                    # causal_length; sliding windows applied per layer
                    # type) and records the frozen teacher KV.
                    h = stack.run_block(
                        layer, h, pe, position_ids=pos,
                        past_key_values=kv, use_cache=True,
                        causal_length=T)
                    kv.recording = False
                    view = h if layer < n else stack.final_norm(h)
                    tensors.put(layer, idx, kv, inputs_q,
                                cohort.gather_query_inputs(view).clone())
                entries += 1
                if layer == n and teacher_eval_rows is not None:
                    rows = []
                    for b in range(B):
                        r = cohort.eval_rows[b].to(device)
                        positions = cohort.qpos_dev[b].index_select(0, r)
                        rows.append(
                            view[b].index_select(0, positions).detach())
                    teacher_eval_rows[idx] = rows
                if rotator is not None:
                    rotator.evict(layer)
        if moe_ctrl is not None and moe_routing is not None:
            moe_routing[idx] = {
                "idx": {L: moe_ctrl.t_idx[L] for L in moe_ctrl.t_idx
                        if L in owned},
                "logp": {L: moe_ctrl.t_logp[L] for L in moe_ctrl.t_logp
                         if L in owned},
            }
        if not last:
            out_path = rf.path(0, f"capture_c{idx:04d}_stage{stage}.st")
            rf.write(out_path, {"h": h.detach().cpu()},
                     stage=stage, epoch=0, to_stage=stage + 1)
            written.append(out_path)
            while sum(1 for p in written if p.exists()) >= inflight:
                time.sleep(0.5)
        del h
    log.log(kind="v4_store_capture",
            seconds=round(time.perf_counter() - started, 3),
            cohorts=len(cohorts), layer_entries=entries,
            stage=stage, pipelined=stages > 1,
            teacher_forwards_per_later_epoch=0)


def _capture_layer_outer(cfg, stack, cohorts, tensors, adapters_off,
                         device, log, *, owned, n_layers, moe_ctrl,
                         moe_routing, teacher_eval_rows, rotator,
                         started) -> None:
    """Layer-outer store-fill for rotary PPP1: activate each block ONCE and
    run every cohort through it before evicting. Boundary hiddens and the
    per-cohort rope bundles (which carry gemma's shared-KV side channel
    across the whole sweep) stay resident for the duration."""
    from .online_v4 import _FrozenKV, _bk_layer_type

    n = n_layers
    entries = 0
    hs, poss, pes = {}, {}, {}
    with torch.no_grad():
        for idx, cohort in enumerate(cohorts):
            B, T = len(cohort.indices), cohort.T
            poss[idx] = torch.arange(T, device=device)[None].expand(B, -1)
            hs[idx] = stack.embed(cohort.teacher_ids.to(device))
            pes[idx] = stack.rope(hs[idx], poss[idx])
        for pos_i, layer in enumerate(owned):
            rotator.activate(layer)
            if pos_i + 1 < len(owned):
                rotator.prefetch(owned[pos_i + 1])
            ltype = _bk_layer_type(stack, layer)
            for idx, cohort in enumerate(cohorts):
                h = hs[idx]
                ctx_a = (adapters_off() if adapters_off is not None
                         else contextlib.nullcontext())
                ctx_m = (moe_ctrl.teacher_phase() if moe_ctrl is not None
                         else contextlib.nullcontext())
                with ctx_a, ctx_m:
                    if ltype == "linear_attention":
                        h_new = stack.run_block(layer, h, pes[idx],
                                                position_ids=poss[idx])
                        view = (h_new if layer < n
                                else stack.loss_view(n, h_new))
                        tensors.put_linear(
                            layer, idx, h.clone(),
                            cohort.gather_query_inputs(view).clone())
                    else:
                        kv = _FrozenKV()
                        inputs_q = cohort.gather_query_inputs(h).clone()
                        h_new = stack.run_block(
                            layer, h, pes[idx], position_ids=poss[idx],
                            past_key_values=kv, use_cache=True,
                            causal_length=cohort.T)
                        kv.recording = False
                        view = (h_new if layer < n
                                else stack.loss_view(n, h_new))
                        tensors.put(layer, idx, kv, inputs_q,
                                    cohort.gather_query_inputs(view).clone())
                entries += 1
                if moe_ctrl is not None and moe_routing is not None:
                    r = moe_routing.setdefault(idx, {"idx": {}, "logp": {}})
                    if layer in moe_ctrl.t_idx:
                        r["idx"][layer] = moe_ctrl.t_idx[layer]
                    if layer in moe_ctrl.t_logp:
                        r["logp"][layer] = moe_ctrl.t_logp[layer]
                if layer == n and teacher_eval_rows is not None:
                    rows = []
                    for b in range(len(cohort.indices)):
                        sel = cohort.eval_rows[b].to(device)
                        positions = cohort.qpos_dev[b].index_select(0, sel)
                        rows.append(
                            view[b].index_select(0, positions).detach())
                    teacher_eval_rows[idx] = rows
                hs[idx] = h_new
            rotator.evict(layer)
    log.log(kind="v4_store_capture",
            seconds=round(time.perf_counter() - started, 3),
            cohorts=len(cohorts), layer_entries=entries,
            stage=-1, pipelined=False, layer_outer=True,
            teacher_forwards_per_later_epoch=0)
