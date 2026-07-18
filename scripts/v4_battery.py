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
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


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
    model = load_causal_lm(cfg.model.name, dtype=torch.bfloat16,
                           device_map="auto")
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
        _epoch_end_telemetry(cfg, stack, tok, log, epoch=args.epoch - 1,
                             baseline=None, started_at=started)
    log.log(kind="v4_battery_subprocess", epoch=args.epoch,
            grafted_tensors=grafted, stages=args.stages,
            seconds=round(time.time() - started, 3))


if __name__ == "__main__":
    main()
