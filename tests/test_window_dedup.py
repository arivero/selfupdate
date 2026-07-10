"""Forward-deduplicated sliding windows (train.window_dedup) must produce
the same gradients and reported losses as the per-endpoint window_step
replay: exact in fp32, within autocast rounding otherwise."""

import copy

import pytest
import torch

from selfupdate.config import ExperimentConfig
from selfupdate.data.dataset import collate_padded_items
from selfupdate.train.layerwise import (
    _sliding_windows_dedup,
    _summed_batch,
    _validate_knob_schedule,
    window_step,
)
from selfupdate.train.losses import HiddenLoss

from .test_padded_batching import (
    TinyStack, _assert_same_grads, _block_grads, _items, _summed_b1,
)


def _traj(stack, ids):
    with torch.no_grad():
        h = stack.embed(ids)
        traj = {0: h.detach()}
        t = h
        for L in range(1, stack.n_layers + 1):
            t = stack.run_block(L, t, None)
            traj[L] = t.detach()
    return traj


@pytest.mark.parametrize("kind", ("nmse", "delta_nmse"))
def test_dedup_matches_window_replay_exact_fp32(kind):
    base = TinyStack()
    base.freeze_non_blocks()
    it = _items(base)[0]
    ids = it.student_ids[None]
    targets = {L: it.hidden[L] for L in range(1, base.n_layers + 1)}
    W, n = 3, base.n_layers

    replay = copy.deepcopy(base)
    loss_fn_r = HiddenLoss(kind)
    traj_r = _traj(replay, ids)
    replay_losses = []
    for L1 in range(1, n + 1):
        L0 = max(1, L1 - W + 1)
        wl, _ = window_step(replay, L0, traj_r[L0 - 1], None,
                            {L1: targets[L1]}, it.s0, it.A, it.ans0 - it.s0,
                            loss_fn_r, readout_w=0.0, L1=L1, autocast=False,
                            all_targets=targets)
        replay_losses.extend(wl)

    dedup = copy.deepcopy(base)
    loss_fn_d = HiddenLoss(kind)
    traj_d = _traj(dedup, ids)

    def _endpoint(L1, x, y):
        if loss_fn_d.is_delta and 1 < L1 < n:
            loss = loss_fn_d.delta(
                y[0, it.s0: it.s0 + it.A], x[0, it.s0: it.s0 + it.A],
                targets[L1], targets[L1 - 1],
            )
        else:
            loss = loss_fn_d(dedup.loss_view(L1, y)[0, it.s0: it.s0 + it.A],
                             targets[L1], normed=(L1 == n))
        return loss, loss.detach()

    dedup_losses = _sliding_windows_dedup(dedup, 1, n, W, traj_d, None,
                                          _endpoint, autocast=False)

    for a, b in zip(replay_losses, dedup_losses):
        assert torch.allclose(a, b, atol=1e-7), "endpoint loss drift"
    _assert_same_grads(_block_grads(replay), _block_grads(dedup),
                       atol=1e-6, rtol=1e-5)


def _sliding_cfg(dedup: bool) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.device = "cpu"
    cfg.train.hidden_loss = "nmse"
    cfg.train.conn_window = 2
    cfg.train.conn_stride = 1
    cfg.train.readout_window_blocks = 2
    cfg.train.readout_weight = 0.25
    cfg.train.readout_source = "teacher_kl"
    cfg.train.window_dedup = dedup
    return cfg


def test_summed_item_dedup_matches_replay_path():
    base = TinyStack()
    base.freeze_non_blocks()
    items = _items(base)

    stacks, grads = {}, {}
    for dedup in (False, True):
        stack = copy.deepcopy(base)
        loss_fn = HiddenLoss("nmse", stack.final_norm, stack.lm_head)
        cfg = _sliding_cfg(dedup)
        for it in items:
            _summed_b1(cfg, stack, loss_fn, it)
        stacks[dedup], grads[dedup] = stack, _block_grads(stack)

    _assert_same_grads(grads[False], grads[True])


def test_summed_batch_dedup_matches_item_dedup():
    base = TinyStack()
    base.freeze_non_blocks()
    items = _items(base)
    cfg = _sliding_cfg(True)

    item_stack = copy.deepcopy(base)
    batch_stack = copy.deepcopy(base)
    loss_fn_item = HiddenLoss("nmse", item_stack.final_norm, item_stack.lm_head)
    loss_fn_batch = HiddenLoss("nmse", batch_stack.final_norm, batch_stack.lm_head)

    for it in items:
        _summed_b1(cfg, item_stack, loss_fn_item, it)
    batch = collate_padded_items(items)
    targets = {L: batch.hidden[L] for L in range(1, batch_stack.n_layers + 1)}
    _summed_batch(cfg, batch_stack, loss_fn_batch, batch, targets, "cpu")

    _assert_same_grads(_block_grads(item_stack), _block_grads(batch_stack))


def test_window_dedup_knob_requires_sliding_windows():
    cfg = ExperimentConfig()
    cfg.train.window_dedup = True
    with pytest.raises(ValueError, match="window_dedup"):
        _validate_knob_schedule(cfg)
    cfg.train.conn_window = 4
    cfg.train.conn_stride = 0
    with pytest.raises(ValueError, match="window_dedup"):
        _validate_knob_schedule(cfg)
    cfg.train.conn_stride = 1
    _validate_knob_schedule(cfg)


def test_window_dedup_rejects_router_aligned_graph_capture():
    cfg = _sliding_cfg(True)
    cfg.train.lora.enabled = True
    cfg.train.online_teacher = True
    cfg.train.moe_mode = "router_aligned"
    cfg.train.moe_router_weight = 0.1
    with pytest.raises(ValueError, match="window_dedup with router_aligned"):
        _validate_knob_schedule(cfg)


def test_epoch_standard_probe_config_validation():
    cfg = _sliding_cfg(False)
    cfg.eval.standard_damage_every_epochs = 1
    cfg.eval.standard_damage_limit = 0
    with pytest.raises(ValueError, match="standard_damage_limit"):
        _validate_knob_schedule(cfg)
    cfg.eval.standard_damage_limit = 16
    cfg.eval.recall_corpora = ["not_a_corpus"]
    with pytest.raises(ValueError, match="unknown eval.recall_corpora"):
        _validate_knob_schedule(cfg)
