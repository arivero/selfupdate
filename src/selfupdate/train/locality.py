"""Model-resident certification of the strict block-local gradient contract."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from ..data.dataset import DistillDataset
from .losses import HiddenLoss


def _norm2(params) -> float:
    return sum(float((p.grad.float() ** 2).sum())
               for p in params if p.grad is not None)


def certify_locality_resident(cfg, stack, tok, cache, run_dir: Path,
                              items: int = 16, teacher=None) -> dict:
    """Reproduce sampled layer losses without unloading/reloading the model.

    Each backward must touch exactly the current block.  Any foreign-block or
    frozen-vocabulary gradient fails the run before checkpoint publication.
    """
    if cfg.train.conn_window > 1:
        raise ValueError("strict-local certification requires conn_window <= 1")
    n = stack.n_layers
    loss_fn = HiddenLoss.from_config(cfg.train, stack)
    teacher_hidden = (cfg.train.pipeline_version == 3
                      and cfg.train.trajectory_source == "teacher_hidden")
    cached_teacher_hidden = (
        teacher_hidden and cfg.train.teacher_hidden_source in (
            "cpu_cache", "gpu_cache"))
    ds = DistillDataset(
        cfg.data.examples_path, cache, tok, list(range(1, n + 1)),
        with_teacher_ids=teacher_hidden and not cached_teacher_hidden,
        with_teacher_inputs=cached_teacher_hidden,
        cache_source_compaction=cfg.cache.source_compaction,
        student_compaction=cfg.mask.compaction,
        pad_random=(cfg.mask.compaction == "pad_random"),
        rebase_gap=(cfg.mask.compaction in ("stub_gap", "remove_gap")),
    )
    stride = max(1, len(ds) // items)
    sampled = [ds[i] for i in range(0, len(ds), stride)][:items]
    local2 = {L: 0.0 for L in range(1, n + 1)}
    foreign2 = {L: 0.0 for L in range(1, n + 1)}
    frozen2 = {L: 0.0 for L in range(1, n + 1)}
    was_training = stack.model.training
    stack.model.eval()

    for it in sampled:
        ids = it.student_ids.to(cfg.model.device)[None]
        pos = it.position_ids.to(cfg.model.device)[None]
        teacher_states = (
            it.teacher_inputs if cached_teacher_hidden else
            teacher.full_states_cpu(it, cfg.model.device)
            if teacher_hidden else None)
        h = stack.embed(ids) if not teacher_hidden else None
        pos_emb = stack.rope(h, pos) if h is not None else None
        flow_keep = None
        if cfg.mask.compaction == "flow_mask" and not teacher_hidden:
            flow_keep = torch.ones(
                (1, len(it.student_ids)), dtype=torch.bool,
                device=cfg.model.device)
            for start, stop in it.t_priv or []:
                flow_keep[:, start:stop] = False
        for L in range(1, n + 1):
            if teacher_hidden:
                block_device = next(stack.blocks[L - 1].parameters()).device
                source = (teacher_states[L] if cached_teacher_hidden
                          else teacher_states[L - 1])
                if cached_teacher_hidden:
                    source = source.unsqueeze(0)
                h_in = source.to(block_device).detach()
                layer_pos = torch.arange(
                    h_in.shape[1], device=block_device)[None]
                pos_emb = stack.rope(h_in, layer_pos)
            else:
                h_in = h.detach()
                layer_pos = pos.to(h_in.device)
            with torch.autocast(h_in.device.type, dtype=torch.bfloat16):
                h_out = stack.run_block(
                    L, h_in, pos_emb, position_ids=layer_pos,
                    flow_keep=(flow_keep.to(h_in.device)
                               if flow_keep is not None else None))
                aligned_start = it.t0 if teacher_hidden else it.s0
                if loss_fn.is_delta and 1 < L < n:
                    target = it.hidden[L].to(h_out.device)
                    loss = loss_fn.delta(
                        h_out[0, aligned_start:aligned_start + it.A],
                        h_in.to(h_out.device)[
                            0, aligned_start:aligned_start + it.A],
                        target, it.hidden[L - 1].to(h_out.device))
                else:
                    view = stack.loss_view(L, h_out)[
                        0, aligned_start:aligned_start + it.A]
                    loss = loss_fn(
                        view, it.hidden[L].to(view.device),
                        normed=(L == n), layer=L)
            stack.model.zero_grad(set_to_none=True)
            loss.backward()
            local2[L] += _norm2(stack.block_params(L))
            foreign2[L] += max(
                (_norm2(stack.block_params(other))
                 for other in range(1, n + 1) if other != L),
                default=0.0)
            frozen2[L] += _norm2(
                list(stack.embed_tokens.parameters())
                + list(stack.final_norm.parameters())
                + list(stack.lm_head.parameters()))
            if not teacher_hidden:
                h = h_out.detach()

    stack.model.zero_grad(set_to_none=True)
    stack.model.train(was_training)
    local = sum(local2.values()) ** 0.5
    foreign = sum(foreign2.values()) ** 0.5
    frozen = sum(frozen2.values()) ** 0.5
    local_signal_present = all(value > 0.0 and math.isfinite(value)
                               for value in local2.values())
    passed = (local_signal_present and math.isfinite(local)
              and foreign == 0.0 and frozen == 0.0)
    payload = {
        "schema_version": 2,
        "run": cfg.run_name,
        "items": len(sampled),
        "teacher_target_source": (
            ("pipeline_v3_full_prefix_teacher_input_cache_plus_disk_target"
             if cached_teacher_hidden else
             "pipeline_v3_uncensored_teacher_prefix_plus_disk_target")
            if teacher_hidden else
            f"pipeline_v{cfg.train.pipeline_version}_disk_cache"),
        "teacher_cache_hash": cache._index["config_hash"],
        "run_class": cfg.train.run_class,
        "hidden_loss": cfg.train.hidden_loss,
        "trajectory_source": cfg.train.trajectory_source,
        "history_policy": cfg.train.history_policy,
        "gradient_contract": "strict_block_local_hidden_state",
        "final_logit_training": False,
        "local_grad_norm": local,
        "cross_block_leak_grad_norm": foreign,
        "frozen_vocab_grad_norm": frozen,
        "local_signal_present_in_every_block": local_signal_present,
        "passed": passed,
        "per_block": {
            str(L): {
                "local_grad_norm": local2[L] ** 0.5,
                "max_foreign_grad_norm": foreign2[L] ** 0.5,
                "frozen_vocab_grad_norm": frozen2[L] ** 0.5,
            }
            for L in range(1, n + 1)
        },
    }
    out = run_dir / "eval" / "signal_attribution.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(".signal_attribution.json.tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out)
    if not passed:
        raise RuntimeError(
            f"strict-local gradient certification failed: local={local}, "
            f"all_blocks_have_signal={local_signal_present}, foreign={foreign}, "
            f"frozen_vocab={frozen}")
    return payload
