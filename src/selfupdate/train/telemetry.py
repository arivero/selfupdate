"""Training telemetry: loss aggregation and epoch-boundary probes.

Everything corpus-specific lives here — which recall corpora a run reports,
the paired standard-benchmark damage probe, the epoch-0 reference — so the
schedule loops in ``layerwise.py`` stay generic: they accumulate per-layer
losses and call the epoch hooks, nothing else. Probes never change training
mode and never contribute a gradient (EVALUATION_ONLY by contract; see
scripts/train_certify.py).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from ..eval.tasks import RECALL_CORPUS_PATHS, tasks_eval


def _loss_float(v) -> float:
    if torch.is_tensor(v):
        return float(v.detach().float().cpu())
    return float(v)


def _summarize_pending_losses(pending: list[list[torch.Tensor]], n_layers: int) -> tuple[float, list[float]]:
    sums: list[torch.Tensor | None] = [None] * n_layers
    counts = [0] * n_layers
    for losses in pending:
        for i, loss in enumerate(losses[:n_layers]):
            t = loss.detach().float() if torch.is_tensor(loss) else torch.tensor(float(loss))
            sums[i] = t if sums[i] is None else sums[i] + t
            counts[i] += 1
    # single host transfer per flush: gather per-layer means onto one device
    # (async d2d under pipeline parallel) and .cpu() the stack once, instead
    # of one GPU->CPU round-trip per layer
    dev = next((s.device for s in sums if s is not None), torch.device("cpu"))
    means = [
        (sums[i] / counts[i]).to(dev, non_blocking=True) if counts[i]
        else torch.full((), float("nan"), device=dev)
        for i in range(n_layers)
    ]
    per_layer = [float(v) for v in torch.stack(means).cpu()]
    valid = [v for v in per_layer if v == v]
    mean = sum(valid) / len(valid) if valid else float("nan")
    return mean, per_layer


def _flush_train_log(log, *, epoch: int, step: int, accum: int,
                     pending: list[list[torch.Tensor]], n_layers: int, **extra) -> None:
    if not pending:
        return
    loss, per_layer = _summarize_pending_losses(pending, n_layers)
    log.log(kind="train", epoch=epoch, step=step, items_seen=accum,
            accum_items=len(pending), loss=loss, per_layer=per_layer, **extra)
    pending.clear()


def _epoch_recall_corpora(cfg) -> list[tuple[str, str]]:
    """Named corpus paths for training telemetry, never inferred from a base.

    Combined configs must pin ``eval.recall_corpora``.  A one-corpus legacy
    config continues to evaluate its declared data.poem_path.
    """
    if cfg.eval.recall_corpora:
        return [(name, RECALL_CORPUS_PATHS[name])
                for name in cfg.eval.recall_corpora]
    return [(Path(cfg.data.poem_path).stem, cfg.data.poem_path)]


def _log_epoch_recall(cfg, stack, tok, log, *, epoch: int, phase: str,
                      started_at: float) -> None:
    """Log corpus-separated fast recall without changing training mode."""
    results = {
        name: tasks_eval(stack.model, tok, path, n_per_task=8,
                         generation_batch=cfg.eval.generation_batch)
        for name, path in _epoch_recall_corpora(cfg)
    }
    summary = {
        name: {
            "next_acc": result["tasks"]["next"]["word_acc"],
            "prev_acc": result["tasks"]["prev"]["word_acc"],
            "cloze_acc": result["tasks"]["cloze"]["word_acc"],
            "overall_word_acc": result["overall_word_acc"],
        }
        for name, result in results.items()
    }
    # Retain flat fields for existing tooling; the corpus map is the source of
    # truth for new/combined arms.
    primary = next(iter(summary.values()))
    overall = sum(v["overall_word_acc"] for v in summary.values()) / len(summary)
    log.log(kind="eval", epoch=epoch, phase=phase, recall=summary,
            next_acc=primary["next_acc"], prev_acc=primary["prev_acc"],
            cloze_acc=primary["cloze_acc"], overall_word_acc=overall,
            vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
            vram_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
            minutes=round((time.time() - started_at) / 60, 1))
    print(" ".join(
        f"{name}: {value['overall_word_acc']:.2f}" for name, value in summary.items()))


def _log_standard_damage(cfg, stack, tok, log, *, epoch: int, phase: str,
                         baseline: dict | None, started_at: float) -> dict:
    """Paired fast standard-benchmark probe for epoch-gating a campaign."""
    from ..eval.standard import STANDARD_TASKS, evaluate_standard

    probe = evaluate_standard(
        stack.model, tok, tasks=STANDARD_TASKS,
        limit=cfg.eval.standard_damage_limit,
        batch_size=cfg.eval.standard_damage_batch_size,
        device=cfg.model.device, keep_examples=False,
    )
    base = baseline or probe
    deltas = {
        task: probe["tasks"][task]["accuracy"] - base["tasks"][task]["accuracy"]
        for task in STANDARD_TASKS
    }
    worst_task = min(deltas, key=deltas.get)
    mean_delta = sum(deltas.values()) / len(deltas)
    log.log(kind="standard_eval", epoch=epoch, phase=phase,
            standard_tasks={task: probe["tasks"][task]["accuracy"]
                            for task in STANDARD_TASKS},
            standard_macro_accuracy=probe["macro_accuracy"],
            standard_epoch0_delta=mean_delta,
            standard_worst_task=worst_task,
            standard_worst_delta=deltas[worst_task],
            standard_limit=probe["limit"],
            benchmark_revisions=probe["benchmark_revisions"],
            vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
            vram_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
            minutes=round((time.time() - started_at) / 60, 1))
    print(f"{phase}: standard {probe['macro_accuracy']:.3f} "
          f"(Δ {mean_delta:+.3f}; worst {worst_task} {deltas[worst_task]:+.3f})")
    return base


def _epoch_zero_telemetry(cfg, stack, tok, log, started_at: float) -> dict | None:
    """Record the paired epoch-0 reference when standard gating is enabled."""
    if not cfg.eval.standard_damage_every_epochs:
        return None
    _log_epoch_recall(cfg, stack, tok, log, epoch=0, phase="epoch0",
                      started_at=started_at)
    return _log_standard_damage(cfg, stack, tok, log, epoch=0, phase="epoch0",
                                baseline=None, started_at=started_at)


def _epoch_end_telemetry(cfg, stack, tok, log, *, epoch: int,
                         baseline: dict | None, started_at: float) -> dict | None:
    """Epoch-boundary probes shared by every epoch-driven schedule: fast
    recall on the configured cadence, plus the paired standard-damage probe
    when gating is enabled. Returns the (unchanged) standard baseline."""
    completed = epoch + 1
    last = epoch == cfg.train.epochs - 1
    if completed % cfg.eval.every_epochs == 0 or last:
        _log_epoch_recall(cfg, stack, tok, log, epoch=completed,
                          phase=f"after_epoch_{completed}",
                          started_at=started_at)
    if (cfg.eval.standard_damage_every_epochs
            and (completed % cfg.eval.standard_damage_every_epochs == 0
                 or last)):
        baseline = _log_standard_damage(
            cfg, stack, tok, log, epoch=completed,
            phase=f"after_epoch_{completed}", baseline=baseline,
            started_at=started_at)
    return baseline
