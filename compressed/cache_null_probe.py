"""Compare disk-cache targets with a same-process intact online teacher.

This is a diagnostic, not training.  For one intact-RAG batch it walks the
student blocks once and reports, at every depth, normalized discrepancies
against (a) the frozen disk cache and (b) an adapters-disabled teacher pass
through the very same resident model and batch shape.
"""

from __future__ import annotations


# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP



import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from selfupdate.utils.env import cap_cpu_threads

cap_cpu_threads()

import torch
import torch.nn.functional as F

from selfupdate.config import load_config
from selfupdate.data.dataset import collate_padded_items
from selfupdate.train.layerwise import _gather_batch_rows, _loader, _make_dataset
from selfupdate.train.runtime import TrainingRuntime
from selfupdate.train.teacher_source import OnlineTeacherSource
from selfupdate.train.validate import validate_knob_schedule


def _normalized_huber(student, target, lengths) -> float:
    values = []
    for i, count in enumerate(lengths):
        s = student[i, :count].float()
        t = target[i, :count].to(student.device).float()
        scale = t.square().mean().sqrt().clamp_min(1e-8)
        values.append(F.smooth_l1_loss(s / scale, t / scale, beta=1.0))
    return float(torch.stack(values).mean().cpu())


@torch.no_grad()
def full_cache_pass(cfg, runtime, cache) -> dict:
    """One complete no-gradient training-loader pass against disk targets."""
    stack, tok = runtime.stack, runtime.tokenizer
    ds = _make_dataset(
        cfg, cache, tok, list(range(1, stack.n_layers + 1)),
        with_teacher_ids=True,
    )
    loader = _loader(cfg, ds)
    # Per-layer GPU scalars: vector count, element count, sum vector L2,
    # relative vector L2, normalized-Huber/vector, cosine error, squared
    # component error, and max absolute component error.
    accum = None
    batches = examples = 0
    started = time.perf_counter()
    for batch in loader:
        if not torch.equal(batch.student_ids, batch.teacher_ids):
            raise RuntimeError("intact batch is not token-identical")
        ids = batch.student_ids.to(cfg.model.device)
        pos = batch.position_ids.to(cfg.model.device)
        with torch.autocast(torch.device(cfg.model.device).type,
                            dtype=torch.bfloat16):
            h = stack.embed(ids)
            pos_emb = stack.rope(h, pos)
            layer_values = []
            for layer in range(1, stack.n_layers + 1):
                h = stack.run_block(layer, h, pos_emb)
                student = _gather_batch_rows(
                    stack.loss_view(layer, h), batch.aligned_index)
                target = batch.hidden[layer].to(student.device)
                valid_student = torch.cat([
                    student[i, :int(count)].float()
                    for i, count in enumerate(batch.A)
                ])
                valid_target = torch.cat([
                    target[i, :int(count)].float()
                    for i, count in enumerate(batch.A)
                ])
                diff = valid_student - valid_target
                vector_l2 = diff.norm(dim=-1)
                target_l2 = valid_target.norm(dim=-1).clamp_min(1e-8)
                scale = valid_target.square().mean(dim=-1).sqrt().clamp_min(1e-8)
                normalized = diff / scale[:, None]
                huber = torch.where(
                    normalized.abs() < 1,
                    0.5 * normalized.square(),
                    normalized.abs() - 0.5,
                ).mean(dim=-1)
                cosine_error = 1 - F.cosine_similarity(
                    valid_student, valid_target, dim=-1, eps=1e-8)
                layer_values.append(torch.stack((
                    torch.tensor(float(vector_l2.numel()), device=student.device),
                    torch.tensor(float(diff.numel()), device=student.device),
                    vector_l2.sum(),
                    (vector_l2 / target_l2).sum(),
                    huber.sum(),
                    cosine_error.sum(),
                    diff.square().sum(),
                    diff.abs().max(),
                )))
        values = torch.stack(layer_values)
        if accum is None:
            accum = values
        else:
            accum[:, :7] += values[:, :7]
            accum[:, 7] = torch.maximum(accum[:, 7], values[:, 7])
        batches += 1
        examples += len(batch.example_ids)
    if accum is None:
        raise RuntimeError("empty intact loader")
    values = accum.cpu()
    rows = []
    for i, value in enumerate(values):
        vectors = float(value[0])
        elements = float(value[1])
        rows.append({
            "layer": i + 1,
            "vectors": int(vectors),
            "mean_vector_l2": float(value[2] / vectors),
            "mean_relative_vector_l2": float(value[3] / vectors),
            "mean_normalized_huber": float(value[4] / vectors),
            "mean_cosine_error": float(value[5] / vectors),
            "component_rmse": float((value[6] / elements).sqrt()),
            "max_abs_component": float(value[7]),
        })
    return {
        "model": cfg.model.name,
        "cache": str(cache.root),
        "examples": examples,
        "batches": batches,
        "batching": cfg.train.batching,
        "batch_size": cfg.train.micro_batch,
        "token_identity": True,
        "gradients": False,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }


@torch.no_grad()
def probe(args) -> dict:
    cfg = load_config(args.config, args.experiment)
    if cfg.mask.compaction != "intact":
        raise ValueError("cache null probe requires mask.compaction=intact")
    validate_knob_schedule(cfg)
    runtime = TrainingRuntime(cfg).load()
    cache = runtime.load_cache()
    if args.all_examples:
        return full_cache_pass(cfg, runtime, cache)
    stack, tok = runtime.stack, runtime.tokenizer
    ds = _make_dataset(
        cfg, cache, tok, list(range(1, stack.n_layers + 1)),
        with_teacher_ids=True,
    )
    items = [ds[i] for i in range(args.batch_size)]
    batch = collate_padded_items(items)
    if not torch.equal(batch.student_ids, batch.teacher_ids):
        raise RuntimeError("intact batch is not token-identical")

    online = OnlineTeacherSource(stack, peft_model=runtime.peft_model)
    online_targets = online.aligned_targets_batch(batch, cfg.model.device)
    ids = batch.student_ids.to(cfg.model.device)
    pos = batch.position_ids.to(cfg.model.device)
    lengths = batch.A.tolist()
    rows = []
    with torch.autocast(torch.device(cfg.model.device).type,
                        dtype=torch.bfloat16):
        h = stack.embed(ids)
        pos_emb = stack.rope(h, pos)
        for layer in range(1, stack.n_layers + 1):
            h = stack.run_block(layer, h, pos_emb)
            student = _gather_batch_rows(
                stack.loss_view(layer, h), batch.aligned_index)
            cached = batch.hidden[layer].to(student.device)
            live = online_targets[layer].to(student.device)
            rows.append({
                "layer": layer,
                "cache_huber": _normalized_huber(student, cached, lengths),
                "online_huber": _normalized_huber(student, live, lengths),
                "cache_max_abs": float(max(
                    (student[i, :k].float() - cached[i, :k].float()).abs().max()
                    for i, k in enumerate(lengths)).cpu()),
                "online_max_abs": float(max(
                    (student[i, :k].float() - live[i, :k].float()).abs().max()
                    for i, k in enumerate(lengths)).cpu()),
            })
    return {
        "model": cfg.model.name,
        "cache": str(cache.root),
        "batch_size": args.batch_size,
        "token_identity": True,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/pareto_v2/base_qwen35_4b.yaml")
    ap.add_argument("--experiment", default=(
        "configs/experiments/pareto_v2/"
        "control_qwen35_4b_huber_intact_b8_all.yaml"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--all-examples", action="store_true",
                    help="one no-gradient loader pass over all 2,071 examples")
    ap.add_argument("--out", default="runs/diagnostics/cache_null_probe_qwen35_4b.json")
    args = ap.parse_args()
    result = probe(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(out)
    for row in result["rows"]:
        if args.all_examples:
            print(
                f"L{row['layer']:02d} rel_vector_l2="
                f"{row['mean_relative_vector_l2']:.8g} "
                f"huber={row['mean_normalized_huber']:.8g} "
                f"cos_err={row['mean_cosine_error']:.8g} "
                f"max={row['max_abs_component']:.6g}")
        else:
            print(
                f"L{row['layer']:02d} cache_huber={row['cache_huber']:.8g} "
                f"online_huber={row['online_huber']:.8g} "
                f"cache_max={row['cache_max_abs']:.6g} "
                f"online_max={row['online_max_abs']:.6g}")


if __name__ == "__main__":
    main()
