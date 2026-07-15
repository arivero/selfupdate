"""Exercise one worst full cohort through the production v3.1 B x K path.

This is an operational fit instrument, not a scientific training arm.  It
uses the cached teacher targets and the student-hidden trajectory, walks every
K tile of one existing length-bucketed cohort, performs the same unaveraged
block-local writes as production, and emits memory/speed/locality evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selfupdate.config import load_config  # noqa: E402
from selfupdate.data.dataset import DistillDataset  # noqa: E402
from selfupdate.train.layerwise import (  # noqa: E402
    TrainingRuntime,
    _validate_knob_schedule,
    dequantize_overrides,
)
from selfupdate.train.losses import HiddenLoss  # noqa: E402
from selfupdate.train.online_v3 import (  # noqa: E402
    _bk_bucketed_cohorts,
    _bk_gather_hidden,
    _bk_layer_type,
    _bk_prepare_cohort_shards,
    _bk_prepare_shard_tile,
    _clear_block_grads,
    _detach_cache_layer,
    _immediate_sgd,
)
from selfupdate.utils.env import cap_cpu_threads  # noqa: E402
from selfupdate.utils.seeding import seed_everything  # noqa: E402


def _trainable_snapshot(stack):
    return {
        layer: [parameter.detach().float().cpu().clone()
                for parameter in stack.block_params(layer)
                if parameter.requires_grad]
        for layer in range(1, stack.n_layers + 1)
    }


def _delta_l2(stack, before):
    by_layer = []
    for layer, references in before.items():
        parameters = [parameter for parameter in stack.block_params(layer)
                      if parameter.requires_grad]
        squared = sum(float(
            (parameter.detach().float().cpu() - reference).square().sum())
            for parameter, reference in zip(parameters, references))
        by_layer.append(math.sqrt(squared))
    return by_layer


def _worst_full_cohort(ds, width: int, seed: int):
    cohorts = _bk_bucketed_cohorts(ds, width, seed)
    full = [cohort for cohort in cohorts if len(cohort) == width]
    if not full:
        raise ValueError(f"dataset has no full B={width} cohort")

    def score(cohort):
        sequence_lengths = [len(ds.pairs[index].student_ids)
                            for index in cohort]
        answer_lengths = [int(ds.pairs[index].A) for index in cohort]
        return (max(sequence_lengths), max(answer_lengths),
                sum(sequence_lengths), sum(answer_lengths))

    return max(full, key=score), score(max(full, key=score))


def main():
    cap_cpu_threads()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config, args.experiment)
    seed_everything(cfg.train.seed)
    _validate_knob_schedule(cfg)
    if cfg.train.history_policy != "causal_bk":
        raise ValueError("cohort probe requires production history_policy=causal_bk")
    if cfg.train.trajectory_source != "student_hidden":
        raise ValueError("cohort probe requires trajectory_source=student_hidden")
    if cfg.train.online_teacher or cfg.train.frozen_teacher_copy:
        raise ValueError("cohort probe must not load an online teacher")
    if not cfg.train.lora.enabled:
        raise ValueError("cohort probe currently requires LoRA")

    load_kw = dequantize_overrides(cfg.model.name, cfg.train.moe_mode)
    runtime = TrainingRuntime(cfg).load(load_kw)
    stack, tokenizer, cache = (
        runtime.stack, runtime.tokenizer, runtime.load_cache())
    if stack.block_devices is not None:
        raise NotImplementedError("causal_bk cohort probe requires one GPU")

    device = torch.device(cfg.model.device)
    n = stack.n_layers
    B = cfg.train.micro_batch
    K = cfg.train.stale_gradient_window
    shard_users = cfg.train.activation_shard_users or B
    layer_types = [_bk_layer_type(stack, layer)
                   for layer in range(1, n + 1)]
    unsupported = sorted(set(layer_types) - {
        "full_attention", "linear_attention"})
    if unsupported:
        raise NotImplementedError(
            f"causal_bk cohort probe lacks semantics for {unsupported}")

    dataset = DistillDataset(
        cfg.data.examples_path, cache, tokenizer,
        need_layers=list(range(1, n + 1)), with_teacher_ids=False,
        pad_random=(cfg.mask.compaction == "pad_random"),
        cache_source_compaction=cfg.cache.source_compaction,
        student_compaction=cfg.mask.compaction)
    indices, cohort_score = _worst_full_cohort(
        dataset, B, cfg.train.seed)
    items = [dataset[index] for index in indices]
    loss_fn = HiddenLoss.from_config(cfg.train, stack)
    before = _trainable_snapshot(stack)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.synchronize(device)
    cohort_started = time.perf_counter()
    shards = _bk_prepare_cohort_shards(
        cfg, stack, items, device, None, False, layer_types, shard_users)
    torch.cuda.synchronize(device)
    prefill_seconds = time.perf_counter() - cohort_started

    tile_started = time.perf_counter()
    max_answer = max(shard["max_answer"] for shard in shards)
    tile_cells = []
    physical_writes = 0
    loss_sums = [torch.zeros((), device=device) for _ in range(n)]
    grad_sums = [torch.zeros((), device=device) for _ in range(n)]
    total_cells = 0
    for start in range(0, max_answer, K):
        tile_states = [
            state for shard in shards
            if (state := _bk_prepare_shard_tile(
                shard, start, K, n, device, stack, False))]
        cells = sum(state["cells"] for state in tile_states)
        if not cells:
            continue
        tile_cells.append(cells)
        total_cells += cells
        for layer in range(1, n + 1):
            parameters = _clear_block_grads(stack, layer)
            layer_loss_sum = torch.zeros(
                (), dtype=torch.float32, device=device)
            for state in tile_states:
                shard = state["shard"]
                h_in = state["h"].detach()
                layer_mask = (
                    state["query_valid"]
                    if layer_types[layer - 1] == "linear_attention"
                    else state["full_mask"])
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    h_out = stack.run_block(
                        layer, h_in, state["query_rope"],
                        position_ids=state["query_positions"],
                        flow_keep=state["key_keep"],
                        past_key_values=shard["history"], use_cache=True,
                        causal_length=(
                            shard["prompt_length"] + state["stop"]),
                        prepared_attention_mask=layer_mask)
                    view = stack.loss_view(layer, h_out)[
                        state["query_valid"]]
                    target = state["window_targets"][layer - 1][
                        state["query_valid"]]
                    mean_loss = loss_fn(
                        view, target, normed=(layer == n), layer=layer)
                    summed_loss = mean_loss * state["cells"]
                summed_loss.backward()
                layer_loss_sum.add_(
                    mean_loss.detach().float() * state["cells"])
                _detach_cache_layer(shard["history"], layer - 1)
                state["h"] = h_out.detach()
                del h_in, h_out, view, target, summed_loss, mean_loss
            grad = _immediate_sgd(parameters, cfg.train.lr)
            loss_sums[layer - 1].add_(layer_loss_sum)
            grad_sums[layer - 1].add_(grad.detach())
            physical_writes += 1
        for state in tile_states:
            del state["window_targets"], state["query_rope"]
            del state["key_keep"], state["full_mask"], state["h"]
            del state["source_index"], state["query_positions"]
            del state["query_valid"]

    torch.cuda.synchronize(device)
    tile_seconds = time.perf_counter() - tile_started
    total_seconds = time.perf_counter() - cohort_started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    deltas = _delta_l2(stack, before)
    runtime.check_vocab_frozen()

    result = {
        "status": "passed",
        "instrument": "production_v31_worst_full_cohort",
        "model": cfg.model.name,
        "cache_hash": cache._index["config_hash"],
        "pipeline_revision": cfg.train.pipeline_revision,
        "trajectory_source": cfg.train.trajectory_source,
        "censorship_compaction": cfg.mask.compaction,
        "loss_kind": cfg.train.hidden_loss,
        "learning_rate": cfg.train.lr,
        "B_simultaneous_users": B,
        "K_context_tokens": K,
        "activation_shard_users": shard_users,
        "activation_shards": len(shards),
        "cohort_score": {
            "max_sequence_tokens": cohort_score[0],
            "max_answer_tokens": cohort_score[1],
            "sum_sequence_tokens": cohort_score[2],
            "sum_answer_tokens": cohort_score[3],
        },
        "prompt_length_padded_max": max(
            shard["prompt_length"] for shard in shards),
        "answer_tokens_max": max_answer,
        "tile_count": len(tile_cells),
        "tile_cells_min": min(tile_cells),
        "tile_cells_max": max(tile_cells),
        "valid_token_events": total_cells,
        "conceptual_block_local_writes": total_cells * n,
        "physical_block_local_writes": physical_writes,
        "prefill_seconds": prefill_seconds,
        "tile_seconds": tile_seconds,
        "total_seconds": total_seconds,
        "tile_token_events_per_s": total_cells / tile_seconds,
        "end_to_end_token_events_per_s": total_cells / total_seconds,
        "baseline_allocated_mib": baseline_allocated / 2**20,
        "baseline_reserved_mib": baseline_reserved / 2**20,
        "peak_allocated_mib": peak_allocated / 2**20,
        "peak_reserved_mib": peak_reserved / 2**20,
        "peak_allocated_increment_mib": (
            peak_allocated - baseline_allocated) / 2**20,
        "per_layer_mean_loss": [
            float((value / total_cells).cpu()) for value in loss_sums],
        "per_layer_mean_normalized_gradient_norm": [
            float((value / total_cells).cpu()) for value in grad_sums],
        "parameter_delta_l2_total": math.sqrt(sum(
            value * value for value in deltas)),
        "parameter_delta_l2_by_layer": deltas,
        "vocabulary_frozen": True,
        "checkpoint_published": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
