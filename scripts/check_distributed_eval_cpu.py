#!/usr/bin/env python
"""CPU-safe numerical checks for the native PP evaluation block protocol.

This does not certify NCCL or a fleet checkpoint.  It checks the architecture
adapters used on either side of stage cuts against the corresponding complete
Transformers model for tiny randomly initialized Qwen and Gemma models.  Fresh
fleet parity remains a disposable-artifact gate, never a stored fingerprint.
"""

from __future__ import annotations

import copy
import hashlib
import math
import multiprocessing as mp
import os
import sys
import tempfile
from collections import UserDict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch
import torch.nn.functional as F
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from selfupdate.train.blocks import BlockStack


def _check_boundary_payload_codec() -> None:
    from selfupdate.train.relay_nccl import BoundaryTransport

    payload = {
        "h": torch.randn(2, 3, 4),
        "shared_kv": {
            "full_attention": (
                torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 4)),
            "sliding_attention": (
                torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 4)),
        },
    }
    encoded = BoundaryTransport._flatten_payload("c7.", payload)
    decoded = BoundaryTransport._unflatten_payload(encoded, "c7.")
    torch.testing.assert_close(decoded["h"], payload["h"])
    assert sorted(decoded["shared_kv"]) == sorted(payload["shared_kv"])
    for layer_type, expected in payload["shared_kv"].items():
        torch.testing.assert_close(
            decoded["shared_kv"][layer_type][0], expected[0])
        torch.testing.assert_close(
            decoded["shared_kv"][layer_type][1], expected[1])
    print("PASS protocol: hidden/shared-KV file payload round trip")


def _check_shared_kv_local_training(model) -> None:
    """Shared consumer query rows match its full teacher-forced block."""
    stack = BlockStack(model)
    stack.freeze_non_blocks()
    ids = torch.tensor([[2, 5, 7, 9, 11, 13, 17, 19]])
    pos = torch.arange(ids.shape[1])[None]
    hidden, _ = stack.embed_and_per_layer_inputs(ids)
    shared = UserDict()
    rope = stack.rope(hidden, pos, shared_kv_states=shared)
    for layer in range(1, 7):
        hidden = stack.run_block(
            layer, hidden, rope, position_ids=pos,
            shared_kv_states=shared)
    layer = 7
    full_out = stack.run_block(
        layer, hidden, rope, position_ids=pos,
        shared_kv_states=shared)
    qpos = torch.tensor([[3, 7]])
    query = hidden.gather(
        1, qpos[..., None].expand(-1, -1, hidden.shape[-1]))
    key_pos = torch.arange(ids.shape[1])[None, None, :]
    allowed = key_pos <= qpos[:, :, None]
    if stack.layer_types[layer - 1] == "sliding_attention":
        allowed &= key_pos > (
            qpos[:, :, None] - int(stack.text_config.sliding_window))
    mask = torch.zeros((1, 1, 2, ids.shape[1]), dtype=query.dtype)
    mask.masked_fill_(~allowed[:, None], torch.finfo(query.dtype).min)
    query_out = stack.run_block(
        layer, query, stack.rope(query, qpos), position_ids=qpos,
        shared_kv_states=shared, prepared_attention_mask=mask)
    expected = full_out.gather(
        1, qpos[..., None].expand(-1, -1, full_out.shape[-1]))
    torch.testing.assert_close(query_out, expected, rtol=2e-5, atol=2e-5)
    print("PASS gemma4_shared_kv: frozen producer KV local-query parity")


def _failure_worker(rank: int, rendezvous: str, queue, injected_phase: str) -> None:
    import torch.distributed as dist
    from selfupdate.eval.distributed_pp import DistributedBattery

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    battery = DistributedBattery.__new__(DistributedBattery)
    battery.stage = rank
    battery.device = torch.device("cpu")
    battery.dist = dist
    battery.group = dist.group.WORLD
    try:
        for phase in ("rank0_decode", "rank0_postprocess", "rank0_durable_log"):
            def operation(phase=phase):
                if rank == 0 and phase == injected_phase:
                    raise ValueError(f"synthetic {phase} failure")
                return 17
            battery.guard_phase(phase, operation)
        queue.put((rank, False))
    except RuntimeError as exc:
        queue.put((rank, "failed" in str(exc)))
    finally:
        dist.destroy_process_group()


def _check_failure_propagation() -> None:
    for phase in ("rank0_decode", "rank0_postprocess", "rank0_durable_log"):
        fd, rendezvous = tempfile.mkstemp(prefix="selfupdate-eval-gloo-")
        os.close(fd)
        os.unlink(rendezvous)
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        workers = [ctx.Process(target=_failure_worker,
                               args=(rank, rendezvous, queue, phase))
                   for rank in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=20)
            assert not worker.is_alive(), (
                f"{phase} failure propagation stranded a rank")
            assert worker.exitcode == 0
        results = sorted(queue.get(timeout=2) for _ in workers)
        assert results == [(0, True), (1, True)]
        print(f"PASS protocol: {phase} failure propagated without stranded peer")


def _digest(model) -> str:
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _manual_forward(stack, ids, mask, pos, *, splits, cache=None):
    hidden, per_layer_inputs = stack.embed_and_per_layer_inputs(ids)
    shared_kv_states = UserDict()
    starts = [1] + [cut + 1 for cut in splits]
    stops = list(splits) + [stack.n_layers]
    for stage_index, (start, stop) in enumerate(zip(starts, stops)):
        stage_cache = (cache[stage_index]
                       if isinstance(cache, (list, tuple)) else cache)
        rope = stack.rope(
            hidden, pos, shared_kv_states=shared_kv_states)
        for layer in range(start, stop + 1):
            hidden = stack.run_block(
                layer, hidden, rope, position_ids=pos,
                flow_keep=mask.bool(), past_key_values=stage_cache,
                use_cache=stage_cache is not None,
                causal_length=mask.shape[1],
                shared_kv_states=shared_kv_states,
                per_layer_input=(per_layer_inputs[:, :, layer - 1]
                                 if per_layer_inputs is not None else None))
        # Emulate the distributed side-channel crossing a stage cut rather
        # than relying on one Python mapping object surviving the boundary.
        shared_kv_states = UserDict({
            key: (value[0].clone(), value[1].clone())
            for key, value in shared_kv_states.items()
        })
        if stage_cache is not None:
            from selfupdate.eval.distributed_pp import DistributedBattery
            battery = DistributedBattery.__new__(DistributedBattery)
            battery.stage = stage_index
            battery.owned = range(start, stop + 1)
            battery.stack = stack
            battery._assert_cache_ownership(stage_cache)
    view = stack.loss_view(stack.n_layers, hidden)
    return stack.lm_head(view)


def _full_cached(model, ids, mask, pos, cache):
    return model(input_ids=ids, attention_mask=mask, position_ids=pos,
                 past_key_values=cache, use_cache=True).logits


def _greedy(model, stack, ids, mask, *, steps, splits):
    pos = mask.long().cumsum(-1) - 1
    pos.masked_fill_(mask == 0, 1)
    full_cache = DynamicCache(config=stack.text_config)
    pp_cache = [DynamicCache(config=stack.text_config)
                for _ in range(len(splits) + 1)]
    full = _full_cached(model, ids, mask, pos, full_cache)
    pp = _manual_forward(stack, ids, mask, pos, splits=splits, cache=pp_cache)
    torch.testing.assert_close(pp[:, -1], full[:, -1], rtol=2e-5, atol=2e-5)
    full_tokens, pp_tokens = [], []
    for _ in range(steps):
        full_tok = full[:, -1].argmax(-1)
        pp_tok = pp[:, -1].argmax(-1)
        assert torch.equal(pp_tok, full_tok)
        full_tokens.append(full_tok)
        pp_tokens.append(pp_tok)
        mask = torch.cat((mask, torch.ones_like(full_tok[:, None])), dim=1)
        pos = mask.long().cumsum(-1)[:, -1:] - 1
        full = _full_cached(model, full_tok[:, None], mask, pos, full_cache)
        pp = _manual_forward(
            stack, pp_tok[:, None], mask, pos, splits=splits, cache=pp_cache)
    return torch.stack(full_tokens, 1), torch.stack(pp_tokens, 1)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"
    padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(ch) % 29) for ch in text]

    def __call__(self, texts, *, return_tensors, padding,
                 padding_side=None, add_special_tokens=False):
        rows = [self.encode(text, add_special_tokens=add_special_tokens)
                for text in texts]
        width = max(map(len, rows))
        side = padding_side or self.padding_side
        ids, masks = [], []
        for row in rows:
            pad = [0] * (width - len(row))
            ids.append(row + pad if side == "right" else pad + row)
            masks.append([1] * len(row) + [0] * len(pad)
                         if side == "right" else
                         [0] * len(pad) + [1] * len(row))
        return {"input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(masks)}


class LocalScoreBackend:
    def __init__(self, stack, splits):
        self.stack, self.splits = stack, splits

    def guard_phase(self, _label, fn):
        """Mirror the distributed backend's rank-local phase interface."""
        return fn()

    def score_pairs(self, tok, pairs, batch_size):
        texts = [p + c for p, c in pairs]
        enc = tok(texts, return_tensors="pt", padding=True,
                  padding_side="right", add_special_tokens=False)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        pos = torch.arange(ids.shape[1])[None].expand(ids.shape[0], -1)
        logits = _manual_forward(
            self.stack, ids, mask, pos, splits=self.splits)
        out = []
        for row, (prompt, choice) in enumerate(pairs):
            start = len(tok.encode(prompt, add_special_tokens=False))
            end = start + len(tok.encode(choice, add_special_tokens=False))
            nll = F.cross_entropy(
                logits[row, start - 1:end - 1].float(), ids[row, start:end],
                reduction="sum").item()
            out.append(-nll / (end - start))
        return out


def _models():
    qwen = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, pad_token_id=0,
        eos_token_id=1, bos_token_id=2, tie_word_embeddings=False,
        layer_types=["full_attention"] * 4))
    gemma = Gemma4ForCausalLM(Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, global_head_dim=8, max_position_embeddings=128,
        pad_token_id=0, eos_token_id=1, bos_token_id=2,
        tie_word_embeddings=False, hidden_size_per_layer_input=0,
        num_kv_shared_layers=0, sliding_window=8,
        layer_types=["sliding_attention", "sliding_attention",
                     "full_attention", "full_attention"]))
    gemma_shared = Gemma4ForCausalLM(Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, global_head_dim=8, max_position_embeddings=128,
        pad_token_id=0, eos_token_id=1, bos_token_id=2,
        tie_word_embeddings=False, hidden_size_per_layer_input=0,
        num_kv_shared_layers=2, sliding_window=8))
    gemma_ple = Gemma4ForCausalLM(Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, global_head_dim=8, max_position_embeddings=128,
        pad_token_id=0, eos_token_id=1, bos_token_id=2,
        tie_word_embeddings=False, hidden_size_per_layer_input=4,
        num_kv_shared_layers=0, sliding_window=8,
        layer_types=["sliding_attention", "sliding_attention",
                     "full_attention", "full_attention"]))
    qwen_hybrid = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, pad_token_id=0,
        eos_token_id=1, bos_token_id=2, tie_word_embeddings=False,
        linear_conv_kernel_dim=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention",
                     "linear_attention", "full_attention"]))
    return [("qwen3", qwen), ("qwen3_5_hybrid", qwen_hybrid),
            ("gemma4", gemma), ("gemma4_shared_kv", gemma_shared),
            ("gemma4_per_layer_inputs", gemma_ple)]


def main() -> None:
    torch.manual_seed(17)
    tok = TinyTokenizer()
    pairs = [("abc", " def"), ("long prompt", " x"), ("q", " yz")]
    from selfupdate.eval.standard import _score_pairs
    from selfupdate.eval.standard import evaluate_task

    models = _models()
    _check_boundary_payload_codec()
    _check_shared_kv_local_training(dict(models)["gemma4_shared_kv"])
    for family, model in models:
        splits = [6] if family == "gemma4_shared_kv" else [2]
        model.eval()
        stack = BlockStack(model)
        stack.freeze_non_blocks()
        from selfupdate.eval.distributed_pp import DistributedBattery
        invariant_probe = DistributedBattery.__new__(DistributedBattery)
        invariant_probe.stack = stack
        invariant_probe.owned = range(1, stack.n_layers + 1)
        invariant_probe.stage = 0
        invariant_probe.device = torch.device("cpu")
        assert (invariant_probe._expected_owned_adapter_keys()
                == invariant_probe._live_owned_adapter_keys())
        vocab_before = invariant_probe._frozen_vocab_digest()
        frozen = next(stack.embed_tokens.parameters())
        frozen_copy = frozen.detach().clone()
        with torch.no_grad():
            frozen.view(-1)[0].add_(1)
        assert invariant_probe._frozen_vocab_digest() != vocab_before
        with torch.no_grad():
            frozen.copy_(frozen_copy)
        assert invariant_probe._frozen_vocab_digest() == vocab_before
        assert invariant_probe._assert_own_gpu_only() is None
        before = _digest(model)
        mode_before = {m: m.training for m in model.modules()}
        ids = torch.tensor([[2, 5, 7, 9, 11, 0],
                            [0, 0, 3, 4, 8, 12]])
        right_mask = torch.tensor([[1, 1, 1, 1, 1, 0],
                                   [1, 1, 1, 1, 1, 1]])
        pos = torch.arange(ids.shape[1])[None].expand(ids.shape[0], -1)
        with torch.inference_mode():
            full = model(input_ids=ids, attention_mask=right_mask,
                         position_ids=pos, use_cache=False).logits
            pp = _manual_forward(stack, ids, right_mask, pos, splits=splits)
            valid = right_mask.bool()
            torch.testing.assert_close(pp[valid], full[valid],
                                       rtol=2e-5, atol=2e-5)

            full_scores = _score_pairs(model, tok, pairs, "cpu")
            pp_scores = _score_pairs(
                None, tok, pairs, "cpu",
                backend=LocalScoreBackend(stack, splits),
                batch_size=len(pairs))
            torch.testing.assert_close(torch.tensor(pp_scores),
                                       torch.tensor(full_scores),
                                       rtol=2e-5, atol=2e-5)
            full_task = evaluate_task(
                model, tok, "arc_easy", limit=3, batch_size=5,
                device="cpu", keep_examples=True)
            pp_task = evaluate_task(
                None, tok, "arc_easy", limit=3, batch_size=5,
                device="cpu", keep_examples=True,
                backend=LocalScoreBackend(stack, splits))
            assert pp_task == full_task

            # B=1 and a variable-left-padding batch exercise prefill plus
            # cached single-token decoding.  Token budgets are evaluated by
            # slicing the common greedy prefix, exactly as the real battery.
            for prompt_ids, prompt_mask in (
                (torch.tensor([[2, 3, 4, 5]]), torch.ones(1, 4, dtype=torch.long)),
                (torch.tensor([[0, 0, 2, 3], [2, 4, 5, 6]]),
                 torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])),
            ):
                full_ids, pp_ids = _greedy(
                    model, stack, prompt_ids, prompt_mask,
                    steps=5, splits=splits)
                assert torch.equal(pp_ids, full_ids)
                assert torch.equal(pp_ids[0, :2], full_ids[0, :2])

        assert _digest(model) == before, f"{family}: parameters mutated"
        assert all(m.training == state for m, state in mode_before.items())
        print(f"PASS {family}: full logits, normalized option NLL, "
              "prefill and 5 cached greedy tokens")
    _check_failure_propagation()


if __name__ == "__main__":
    main()
