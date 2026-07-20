#!/usr/bin/env python
"""Pause-and-eval battery subprocess for stage-scoped pipeline-v4 (plan B6).

Spawned by stage 0 at a battery epoch AFTER every stage has published its
owned adapter shard and released its VRAM (evicted rotated blocks + acked).
Loads the full model device_map=auto over every visible card from the local
snapshot cache, grafts all stages' adapters, and runs the SAME telemetry
probes as v3/v4 single-process mode — the owner's non-negotiable per-epoch
battery (recall corpora incl. epoch zero, standard damage) — appending rows
to stage 0's metrics.jsonl. Exits; stages resume training.

Never launched by hand during a run: the ack/done coordination in
online_v4._subprocess_battery owns the GPU handoff.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _load_epoch_zero_standard_baseline(run_dir: Path) -> dict:
    """Rebuild telemetry.py's baseline shape from the durable epoch-0 row.

    Each staged battery is a fresh process, so an in-memory baseline cannot be
    carried from the epoch-zero child.  The raw task accuracies are the durable
    source of truth; refusing a missing/malformed row is safer than silently
    recording every later damage delta as zero.
    """
    path = run_dir / "metrics.jsonl"
    candidates = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "standard_eval" and row.get("epoch") == 0:
                candidates.append(row)
    if not candidates:
        raise RuntimeError(
            f"missing epoch-zero standard_eval baseline in {path}; refusing "
            "to emit false zero damage deltas")
    row = candidates[-1]
    scores = row.get("standard_tasks")
    if not isinstance(scores, dict) or not scores:
        raise RuntimeError(f"malformed epoch-zero standard_tasks in {path}")
    return {
        "tasks": {
            task: {"accuracy": float(accuracy)}
            for task, accuracy in scores.items()
        },
        "macro_accuracy": float(row["standard_macro_accuracy"]),
        "limit": int(row["standard_limit"]),
        "benchmark_revisions": dict(row.get("benchmark_revisions", {})),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--run-dir", required=True,
                    help="stage 0's run directory (rows append there)")
    ap.add_argument("--epoch", type=int, required=True,
                    help="battery epoch (0 = epoch-zero baseline)")
    ap.add_argument("--stages", type=int, required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from selfupdate.config import load_config
    from selfupdate.train.blocks import BlockStack
    from selfupdate.train.lora import attach_lora
    from selfupdate.train.online_v4 import _RelayFiles
    from selfupdate.train.runtime import load_causal_lm
    from selfupdate.train.telemetry import (_epoch_end_telemetry,
                                            _epoch_zero_telemetry)
    from selfupdate.utils.runlog import RunLog

    cfg = load_config(args.config, args.experiment)
    run_dir = Path(args.run_dir)
    log = RunLog(run_dir, defaults={"battery_subprocess": True})

    # device_map="auto" BALANCES the model across every visible card even when
    # it fits one — and battery generation over that split shuttles activations
    # across cards through accelerate hooks EVERY token, which made a 26B eval
    # take ~15-20 min (measured 2026-07-19). If the model fits the emptiest card
    # (26B/31B/35B/27B all do at ~52-70GB bf16), pin the WHOLE model there so
    # generation is single-card fast; only genuinely-too-big models (122B+) fall
    # back to auto. The training stages have evicted, so the free card is real.
    # Find the emptiest card WITHOUT initializing a CUDA context on each one.
    # torch.cuda.mem_get_info(i) creates a context on card i just to query it,
    # which littered ~518 MiB stray contexts on every non-eval card (bug: the
    # single-card fix pinned the MODEL correctly but still probed all cards).
    # NVML reads free memory context-free; fall back to torch only if NVML is
    # unavailable. Physical index == torch index here (CUDA_VISIBLE_DEVICES is
    # unset for staged battery children — physical ids passed verbatim).
    n_dev = torch.cuda.device_count()
    best_card, best_free = 0, 0
    try:
        import pynvml
        pynvml.nvmlInit()
        for i in range(n_dev):
            free = pynvml.nvmlDeviceGetMemoryInfo(
                pynvml.nvmlDeviceGetHandleByIndex(i)).free
            if free > best_free:
                best_free, best_card = free, i
        pynvml.nvmlShutdown()
    except Exception:
        import subprocess as _sp
        out = _sp.run(["nvidia-smi", "--query-gpu=memory.free",
                       "--format=csv,noheader,nounits"],
                      capture_output=True, text=True).stdout
        frees = [int(x) for x in out.split() if x.strip().isdigit()]
        if frees:
            best_card = max(range(len(frees)), key=lambda i: frees[i])
            best_free = frees[best_card] * 2**20
    placement = "auto"
    model = None
    try:
        model = load_causal_lm(cfg.model.name, dtype=torch.bfloat16,
                               device_map={"": best_card})
        placement = f"single:cuda{best_card}"
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        del model
        model = None
        torch.cuda.empty_cache()
        model = load_causal_lm(cfg.model.name, dtype=torch.bfloat16,
                               device_map="auto")
    log.log(kind="v4_battery_placement", epoch=args.epoch,
            placement=placement, best_card=best_card,
            best_free_gb=round(best_free / 2**30, 1))
    peft_model = attach_lora(model, cfg.train.lora)
    model = peft_model.get_base_model()
    stack = BlockStack(model)
    stack.freeze_non_blocks()

    # Mirror the trainer: solo runs (stages == 1) use run_dir itself
    # as the exchange root; staged run_dirs are one level deeper.
    rf = _RelayFiles(run_dir.parent if args.stages > 1 else run_dir)
    grafted = 0
    with torch.no_grad():
        for k in range(args.stages):
            path = rf.wait(rf.path(args.epoch, f"adapters_stage{k}.st"))
            tensors = rf.read(path, expect_epoch=args.epoch, as_stage=0)
            path.unlink(missing_ok=True)
            for key, value in tensors.items():
                layer_tag, _, local = key.partition(".")
                layer = int(layer_tag[1:])
                param = dict(
                    stack.blocks[layer - 1].named_parameters())[local]
                param.copy_(value.to(param.device, param.dtype))
                grafted += 1
    # Telemetry probes read the device from cfg; with device_map=auto the
    # meaningful anchor is the embedding device (inputs enter there).
    cfg.model.device = str(model.get_input_embeddings().weight.device)
    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    started = time.time()
    if args.epoch == 0:
        _epoch_zero_telemetry(cfg, stack, tok, log, started)
    else:
        baseline = None
        if cfg.eval.standard_damage_every_epochs:
            baseline = _load_epoch_zero_standard_baseline(run_dir)
            log.log(kind="v4_battery_baseline_recovered", epoch=args.epoch,
                    source="stage0_metrics_epoch0_standard_eval",
                    standard_limit=baseline["limit"])
        _epoch_end_telemetry(cfg, stack, tok, log, epoch=args.epoch - 1,
                             baseline=baseline, started_at=started)
    log.log(kind="v4_battery_subprocess", epoch=args.epoch,
            grafted_tensors=grafted, stages=args.stages,
            seconds=round(time.time() - started, 3))


if __name__ == "__main__":
    main()
