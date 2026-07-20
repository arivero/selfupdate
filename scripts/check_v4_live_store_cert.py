#!/usr/bin/env python3
"""CPU self-check for the live-store v4 locality publication gate.

This is deliberately a focused executable check, not a stored certification
fixture.  It exercises every-layer local signal, exact cross-block/frozen
zeros, byte-stable adapters and Adam state, gradient cleanup, and the
missing-store hard failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selfupdate.train.online_v4 import _certify_live_store_locality


class _Stack:
    def __init__(self) -> None:
        self.blocks = torch.nn.ModuleList([
            torch.nn.Linear(4, 4, bias=False),
            torch.nn.Linear(4, 4, bias=False),
        ])
        self.n_layers = len(self.blocks)
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.final_norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.hc_head = None
        for module in (self.embed_tokens, self.final_norm, self.lm_head):
            module.requires_grad_(False)

    def block_params(self, layer: int):
        return self.blocks[layer - 1].parameters()


class _Store:
    def __init__(self, layers, n_cohorts: int) -> None:
        self.entries = {(layer, cohort): object()
                        for layer in layers for cohort in range(n_cohorts)}

    def get(self, layer: int, cohort_idx: int):
        return self.entries.get((layer, cohort_idx))


class _Loss:
    translators = None


def main() -> None:
    torch.manual_seed(7)
    stack = _Stack()
    owned = range(1, 3)
    n_cohorts = 2
    store = _Store(owned, n_cohorts)
    x = {layer: torch.randn(3, 4) for layer in owned}
    target = {layer: torch.randn(3, 4) for layer in owned}

    def local_forward_loss(layer: int, cohort_idx: int):
        assert store.get(layer, cohort_idx) is not None
        out = stack.blocks[layer - 1](x[layer].detach())
        wanted = (out.detach() if layer == 2 and cohort_idx == 0
                  else target[layer].detach())
        summed = torch.nn.functional.mse_loss(
            out, wanted, reduction="sum")
        return {"summed": summed}

    optimizers = {}
    for layer in owned:
        params = list(stack.block_params(layer))
        opt = torch.optim.AdamW(params, lr=1e-3)
        for param in params:
            opt.state[param] = {
                "step": torch.tensor(3.0),
                "exp_avg": torch.randn_like(param),
                "exp_avg_sq": torch.rand_like(param),
            }
        optimizers[layer] = opt

    cert = _certify_live_store_locality(
        None, stack, store, [object()] * n_cohorts, owned, local_forward_loss,
        _Loss(), optimizers, None)
    assert cert["passed"], cert
    assert cert["local_signal_present_in_every_block"], cert
    assert cert["cross_block_leak_grad_norm"] == 0.0, cert
    assert cert["frozen_vocab_grad_norm"] == 0.0, cert
    assert cert["per_layer"]["1"]["probes_used"] == 1, cert
    assert cert["per_layer"]["2"]["probes_used"] == 2, cert
    assert all(param.grad is None for block in stack.blocks
               for param in block.parameters())

    del store.entries[(2, 0)]
    try:
        _certify_live_store_locality(
            None, stack, store, [object()] * n_cohorts, owned,
            local_forward_loss,
            _Loss(), optimizers, None)
    except RuntimeError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing live-store entry did not hard-fail")
    print("v4 live-store locality CPU self-check: PASS")


if __name__ == "__main__":
    main()
