"""Compare disk-cache targets with a same-process intact online teacher.

This is a diagnostic, not training.  For one intact-RAG batch it walks the
student blocks once and reports, at every depth, normalized discrepancies
against (a) the frozen disk cache and (b) an adapters-disabled teacher pass
through the very same resident model and batch shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selfupdate.utils.env import cap_cpu_threads

cap_cpu_threads()

import torch
import torch.nn.functional as F

from selfupdate.config import load_config
from selfupdate.data.dataset import collate_padded_items
from selfupdate.train.layerwise import _gather_batch_rows, _make_dataset
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
def probe(args) -> dict:
    cfg = load_config(args.config, args.experiment)
    if cfg.mask.compaction != "intact":
        raise ValueError("cache null probe requires mask.compaction=intact")
    validate_knob_schedule(cfg)
    runtime = TrainingRuntime(cfg).load()
    cache = runtime.load_cache()
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
        print(
            f"L{row['layer']:02d} cache_huber={row['cache_huber']:.8g} "
            f"online_huber={row['online_huber']:.8g} "
            f"cache_max={row['cache_max_abs']:.6g} "
            f"online_max={row['online_max_abs']:.6g}")


if __name__ == "__main__":
    main()
