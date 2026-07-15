"""Probe-only fixed-shape CUDA graph for pipeline-v3 teacher-hidden K=1.

This module is intentionally not a campaign schedule. It isolates the claim
that a fixed-address KV cache can turn the complete local token step into one
CUDA-graph replay. Promotion requires exact trainable-delta comparison against
the eager ``causal_frozen_history`` path plus a memory/throughput measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from transformers import StaticCache

from .online_v3 import _clear_block_grads, _immediate_sgd


@dataclass
class StaticGraphProbeResult:
    losses: list[torch.Tensor]
    grad_norms: list[torch.Tensor]
    cache: StaticCache
    warmup_s: float
    capture_s: float
    replay_s: float
    prefill_s: float
    replay_token_events: int
    loss_min: float
    loss_max: float
    grad_norm_min: float
    grad_norm_max: float


def _single_cuda_device(stack) -> torch.device:
    devices = {
        p.device
        for layer in range(1, stack.n_layers + 1)
        for p in stack.block_params(layer)
    }
    if len(devices) != 1:
        raise NotImplementedError(
            "causal_static_graph_probe initially requires every block on one device")
    device = next(iter(devices))
    if device.type != "cuda":
        raise NotImplementedError("causal_static_graph_probe requires CUDA")
    return device


def _detach_static_layer(cache: StaticCache, index: int) -> None:
    """Sever capture-time autograd metadata without changing cache addresses."""
    layer = cache.layers[index]
    for name in ("keys", "values"):
        value = getattr(layer, name, None)
        if torch.is_tensor(value) and value.requires_grad:
            value.detach_()


class TeacherStaticGraphProbe:
    """One-answer fixed-shape graph capture; no cross-answer claims yet."""

    def __init__(self, cfg, stack, loss_fn, it, teacher_states,
                 position_ids, targets):
        self.cfg = cfg
        self.stack = stack
        self.loss_fn = loss_fn
        self.it = it
        self.device = _single_cuda_device(stack)
        if any(kind not in (None, "full_attention")
               for kind in stack.layer_types):
            raise NotImplementedError(
                "causal_static_graph_probe initially supports full attention only")
        self.length = int(position_ids.numel())
        self.answer_length = int(it.A)
        self.cache = StaticCache(
            config=stack.text_config, max_cache_len=self.length)

        self.positions = position_ids.to(self.device).contiguous()
        self.inputs = [
            teacher_states[layer].to(self.device).detach().contiguous()
            for layer in range(stack.n_layers)
        ]
        self.targets = [
            targets[layer].to(self.device).detach().contiguous()
            for layer in range(1, stack.n_layers + 1)
        ]
        self.flow_keep = torch.ones(
            (1, self.length), dtype=torch.bool, device=self.device)
        if cfg.mask.compaction == "flow_mask":
            for start, stop in it.t_priv or []:
                self.flow_keep[:, start:min(stop, self.length)] = False
        self.key_positions = torch.arange(self.length, device=self.device)
        self.answer_index = torch.full(
            (1,), -1, dtype=torch.long, device=self.device)
        self.aligned_start = torch.tensor(
            [it.s0], dtype=torch.long, device=self.device)
        n = stack.n_layers
        self.loss_sums = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.grad_sums = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.loss_min = torch.full(
            (), float("inf"), dtype=torch.float32, device=self.device)
        self.loss_max = torch.full(
            (), float("-inf"), dtype=torch.float32, device=self.device)
        self.grad_min = torch.full(
            (), float("inf"), dtype=torch.float32, device=self.device)
        self.grad_max = torch.full(
            (), float("-inf"), dtype=torch.float32, device=self.device)
        self.graph = None
        self._fixed_graph_grads = False

    @staticmethod
    @torch.no_grad()
    def _zero_fixed_grads(params) -> None:
        groups = {}
        for param in params:
            if param.grad is None:
                raise RuntimeError("fixed graph gradient buffer is missing")
            groups.setdefault((param.device, param.grad.dtype), []).append(
                param.grad)
        for grads in groups.values():
            torch._foreach_zero_(grads)

    @staticmethod
    @torch.no_grad()
    def _write_fixed_grads(params, lr: float) -> torch.Tensor:
        groups = {}
        for param in params:
            if param.grad is None:
                raise RuntimeError("captured backward produced no fixed gradient")
            key = (param.device, param.dtype, param.grad.dtype)
            group = groups.setdefault(key, ([], []))
            group[0].append(param)
            group[1].append(param.grad)
        total = torch.zeros((), dtype=torch.float32, device=params[0].device)
        for (_, _, _), (group_params, grads) in groups.items():
            norms = torch.stack(torch._foreach_norm(grads, 2)).float()
            total.add_(norms.square().sum())
            torch._foreach_add_(group_params, grads, alpha=-lr)
        return total.sqrt()

    @torch.no_grad()
    def _reset_metrics(self) -> None:
        self.answer_index.fill_(-1)
        self.loss_sums.zero_()
        self.grad_sums.zero_()
        self.loss_min.fill_(float("inf"))
        self.loss_max.fill_(float("-inf"))
        self.grad_min.fill_(float("inf"))
        self.grad_max.fill_(float("-inf"))

    @torch.no_grad()
    def _prefill(self) -> float:
        started = time.perf_counter()
        self.cache.reset()
        stop = int(self.it.s0)
        if stop:
            pos = self.positions[:stop][None]
            q_pos = torch.arange(stop, device=self.device)[:, None]
            allowed = (self.key_positions[None, :] <= q_pos)[None]
            allowed &= self.flow_keep[:, None, :]
            dtype = self.inputs[0].dtype
            mask = torch.zeros(
                (1, 1, stop, self.length), dtype=dtype, device=self.device)
            mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
            pos_emb = self.stack.rope(self.inputs[0][:, :stop], pos)
            keep = self.flow_keep[:, :stop]
            for layer in range(1, self.stack.n_layers + 1):
                self.stack.run_block(
                    layer, self.inputs[layer - 1][:, :stop], pos_emb,
                    position_ids=pos, flow_keep=keep,
                    past_key_values=self.cache, use_cache=True,
                    causal_length=self.length,
                    prepared_attention_mask=mask,
                )
                _detach_static_layer(self.cache, layer - 1)
        torch.cuda.synchronize(self.device)
        return time.perf_counter() - started

    def _captured_token(self) -> None:
        """All tensor work for one token; Python executes only at capture."""
        self.answer_index.add_(1)
        pos_index = self.aligned_start + self.answer_index
        pos = self.positions.index_select(0, pos_index)[None]
        allowed = self.key_positions[None, :] <= pos_index[:, None]
        allowed &= self.flow_keep
        dtype = self.inputs[0].dtype
        attention_mask = torch.zeros(
            (1, 1, 1, self.length), dtype=dtype, device=self.device)
        attention_mask.masked_fill_(
            ~allowed[:, None, None], torch.finfo(dtype).min)
        first = self.inputs[0].index_select(1, pos_index)
        pos_emb = self.stack.rope(first, pos)
        query_keep = self.flow_keep.index_select(1, pos_index)
        for layer in range(1, self.stack.n_layers + 1):
            if self._fixed_graph_grads:
                params = self.stack._v3_trainable_block_params[layer - 1]
                self._zero_fixed_grads(params)
            else:
                params = _clear_block_grads(self.stack, layer)
            h_in = self.inputs[layer - 1].index_select(1, pos_index).detach()
            target = self.targets[layer - 1].index_select(
                0, self.answer_index)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h_out = self.stack.run_block(
                    layer, h_in, pos_emb, position_ids=pos,
                    flow_keep=query_keep, past_key_values=self.cache,
                    use_cache=True, causal_length=self.length,
                    prepared_attention_mask=attention_mask,
                )
                view = self.stack.loss_view(layer, h_out)[0]
                loss = self.loss_fn(
                    view, target, normed=(layer == self.stack.n_layers),
                    layer=layer)
            loss.backward()
            grad = (self._write_fixed_grads(params, self.cfg.train.lr)
                    if self._fixed_graph_grads else
                    _immediate_sgd(params, self.cfg.train.lr))
            _detach_static_layer(self.cache, layer - 1)
            detached_loss = loss.detach().float()
            detached_grad = grad.detach().float()
            index = layer - 1
            self.loss_sums[index].add_(detached_loss)
            self.grad_sums[index].add_(detached_grad)
            self.loss_min.copy_(torch.minimum(self.loss_min, detached_loss))
            self.loss_max.copy_(torch.maximum(self.loss_max, detached_loss))
            self.grad_min.copy_(torch.minimum(self.grad_min, detached_grad))
            self.grad_max.copy_(torch.maximum(self.grad_max, detached_grad))

    def run(self, token_count: int, *, capture: bool = True) -> StaticGraphProbeResult:
        if token_count <= 0 or token_count > self.answer_length:
            raise ValueError(
                f"graph probe token_count must be in 1..{self.answer_length}")
        params = [
            p for layer in range(1, self.stack.n_layers + 1)
            for p in self.stack.block_params(layer) if p.requires_grad
        ]
        self._reset_metrics()
        prefill_s = self._prefill()

        if not capture:
            replay_started = time.perf_counter()
            for _ in range(token_count):
                self._captured_token()
            torch.cuda.synchronize(self.device)
            replay_s = time.perf_counter() - replay_started
            return self._result(
                token_count, warmup_s=0.0, capture_s=0.0,
                replay_s=replay_s, prefill_s=prefill_s,
                replay_token_events=token_count)

        snapshots = [p.detach().clone() for p in params]

        warm_started = time.perf_counter()
        warm_stream = torch.cuda.Stream(device=self.device)
        warm_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warm_stream):
            self._captured_token()
        torch.cuda.current_stream(self.device).wait_stream(warm_stream)
        torch.cuda.synchronize(self.device)
        warmup_s = time.perf_counter() - warm_started

        with torch.no_grad():
            for param, snapshot in zip(params, snapshots):
                param.copy_(snapshot)
        del snapshots
        self._reset_metrics()
        prefill_s += self._prefill()
        for param in params:
            param.grad = torch.zeros_like(param)
        self._fixed_graph_grads = True
        torch.cuda.synchronize(self.device)

        capture_started = time.perf_counter()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._captured_token()
        torch.cuda.synchronize(self.device)
        capture_s = time.perf_counter() - capture_started

        replay_started = time.perf_counter()
        for _ in range(1, token_count):
            self.graph.replay()
        torch.cuda.synchronize(self.device)
        replay_s = time.perf_counter() - replay_started
        return self._result(
            token_count, warmup_s=warmup_s, capture_s=capture_s,
            replay_s=replay_s, prefill_s=prefill_s,
            replay_token_events=max(0, token_count - 1))

    def _result(self, token_count: int, *, warmup_s: float,
                capture_s: float, replay_s: float, prefill_s: float,
                replay_token_events: int) -> StaticGraphProbeResult:
        losses = [value.detach() / token_count for value in self.loss_sums]
        grads = [value.detach() / token_count for value in self.grad_sums]
        return StaticGraphProbeResult(
            losses=losses,
            grad_norms=grads,
            cache=self.cache,
            warmup_s=warmup_s,
            capture_s=capture_s,
            replay_s=replay_s,
            prefill_s=prefill_s,
            replay_token_events=replay_token_events,
            loss_min=float(self.loss_min.detach().cpu()),
            loss_max=float(self.loss_max.detach().cpu()),
            grad_norm_min=float(self.grad_min.detach().cpu()),
            grad_norm_max=float(self.grad_max.detach().cpu()),
        )


def run_teacher_static_graph_probe(cfg, stack, loss_fn, it, token_count,
                                   teacher_states, position_ids, targets,
                                   *, capture: bool = True):
    runner = TeacherStaticGraphProbe(
        cfg, stack, loss_fn, it, teacher_states, position_ids, targets)
    return runner.run(token_count, capture=capture)
