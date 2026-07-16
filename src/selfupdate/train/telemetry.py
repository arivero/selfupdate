"""Training telemetry: loss aggregation and epoch-boundary probes.

Everything corpus-specific lives here — which recall corpora a run reports,
the same-subset standard-benchmark damage probe, the epoch-0 reference — so the
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


EPOCH_RECALL_ITEMS_PER_TASK = 8


class ParameterDeltaTracker:
    """Epoch-boundary trainable-parameter deltas from the epoch-zero model.

    References live in host RAM. At a boundary they stream to each block's
    device one tensor at a time; only one scalar stack per block returns to
    the host. This avoids a second resident GPU model and never touches the
    hot block walk.
    """

    def __init__(self, stack):
        self.stack = stack
        self.refs: dict[int, list[torch.Tensor]] = {}
        self.lora_refs: dict[int, list[dict]] = {}
        self.counts: list[int] = []
        self.representation = "full_weight_delta_from_epoch0"
        for layer in range(1, stack.n_layers + 1):
            lora = []
            for module in stack.blocks[layer - 1].modules():
                adapters_a = getattr(module, "lora_A", None)
                adapters_b = getattr(module, "lora_B", None)
                scaling = getattr(module, "scaling", None)
                base_layer = getattr(module, "base_layer", None)
                if not adapters_a or not adapters_b or scaling is None:
                    continue
                for name in adapters_a.keys():
                    if name not in adapters_b:
                        continue
                    a = adapters_a[name].weight
                    b = adapters_b[name].weight
                    if not (a.requires_grad or b.requires_grad):
                        continue
                    if a.ndim != 2 or b.ndim != 2:
                        raise NotImplementedError(
                            "effective LoRA delta telemetry requires matrix adapters")
                    base_weight = getattr(base_layer, "weight", None)
                    if base_weight is None:
                        raise RuntimeError("LoRA module has no base_layer.weight")
                    lora.append({
                        "a": a,
                        "b": b,
                        "a0": a.detach().float().cpu().clone(),
                        "b0": b.detach().float().cpu().clone(),
                        "scale": float(scaling[name]),
                        "base_sq": float(
                            base_weight.detach().float().square().sum()),
                        "effective_count": base_weight.numel(),
                    })
            self.lora_refs[layer] = lora
            params = [p for p in stack.block_params(layer) if p.requires_grad]
            if lora:
                self.representation = "effective_lora_weight_delta_from_epoch0"
                self.refs[layer] = []
                self.counts.append(sum(x["effective_count"] for x in lora))
            else:
                self.refs[layer] = [p.detach().float().cpu().clone() for p in params]
                self.counts.append(sum(p.numel() for p in params))

        has_lora = any(self.lora_refs.values())
        has_full = any(self.refs.values())
        if has_lora and has_full:
            raise RuntimeError(
                "mixed LoRA/full trainable blocks need an explicit delta policy")

    @staticmethod
    def _low_rank_delta_sq(entry: dict) -> torch.Tensor:
        """||scale * (B A - B0 A0)||_F^2 without materializing out×in."""
        a, b = entry["a"].detach().float(), entry["b"].detach().float()
        a0 = entry["a0"].to(a.device)
        b0 = entry["b0"].to(b.device)
        # [B, -B0] @ [A; A0] represents BA - B0A0.  Its Frobenius
        # norm follows from rank-(2r) Gram matrices, keeping telemetry O(r²).
        left = torch.cat((b, -b0), dim=1)
        right = torch.cat((a, a0), dim=0)
        gram_left = left.T @ left
        gram_right = right @ right.T
        return (gram_left * gram_right.T).sum().clamp_min(0) * entry["scale"] ** 2

    @torch.no_grad()
    def log(self, log, *, epoch: int, phase: str, started_at: float) -> None:
        if epoch == 0:
            absolute = [0.0] * len(self.counts)
            relative = [0.0] * len(self.counts)
        else:
            absolute, relative = [], []
            for layer, refs in self.refs.items():
                lora = self.lora_refs[layer]
                if lora:
                    dev = lora[0]["a"].device
                    delta_sq = torch.zeros((), device=dev, dtype=torch.float64)
                    base_sq = 0.0
                    for entry in lora:
                        delta_sq += self._low_rank_delta_sq(entry).double()
                        base_sq += entry["base_sq"]
                    values = torch.stack((
                        delta_sq.sqrt(),
                        (delta_sq / max(base_sq, 1e-30)).sqrt(),
                    )).cpu()
                    absolute.append(float(values[0]))
                    relative.append(float(values[1]))
                    continue
                params = [p for p in self.stack.block_params(layer)
                          if p.requires_grad]
                if len(params) != len(refs):
                    raise RuntimeError(
                        f"trainable parameter set changed at layer {layer}")
                if not params:
                    absolute.append(0.0)
                    relative.append(0.0)
                    continue
                dev = params[0].device
                delta_sq = torch.zeros((), device=dev, dtype=torch.float64)
                base_sq = torch.zeros((), device=dev, dtype=torch.float64)
                for param, ref_cpu in zip(params, refs):
                    ref = ref_cpu.to(device=param.device, non_blocking=False)
                    delta = param.detach().float() - ref
                    delta_sq += delta.square().sum(dtype=torch.float64)
                    base_sq += ref.square().sum(dtype=torch.float64)
                    del delta
                    del ref
                values = torch.stack((delta_sq.sqrt(),
                                      (delta_sq / base_sq.clamp_min(1e-30)).sqrt())).cpu()
                absolute.append(float(values[0]))
                relative.append(float(values[1]))
        log.log(kind="parameter_delta", epoch=epoch, phase=phase,
                representation=self.representation,
                per_layer_absolute_l2=absolute,
                per_layer_relative_l2=relative,
                per_layer_parameter_count=self.counts,
                minutes=round((time.time() - started_at) / 60, 1))


def _loss_float(v) -> float:
    if torch.is_tensor(v):
        return float(v.detach().float().cpu())
    return float(v)


def _summarize_pending_losses(
    pending: list[list[torch.Tensor]], n_layers: int,
    token_counts: list[int] | None = None,
) -> tuple[float, list[float], float, list[float]]:
    sums: list[torch.Tensor | None] = [None] * n_layers
    token_sums: list[torch.Tensor | None] = [None] * n_layers
    counts = [0] * n_layers
    total_tokens = 0
    for row, losses in enumerate(pending):
        tokens = token_counts[row] if token_counts is not None else 1
        total_tokens += tokens
        for i, loss in enumerate(losses[:n_layers]):
            t = loss.detach().float() if torch.is_tensor(loss) else torch.tensor(float(loss))
            sums[i] = t if sums[i] is None else sums[i] + t
            weighted = t * tokens
            token_sums[i] = (weighted if token_sums[i] is None
                             else token_sums[i] + weighted)
            counts[i] += 1
    # single host transfer per flush: gather per-layer means onto one device
    # (async d2d under pipeline parallel) and .cpu() the stack once, instead
    # of one GPU->CPU round-trip per layer
    dev = next((s.device for s in sums if s is not None), torch.device("cpu"))
    answer_means = [
        (sums[i] / counts[i]).to(dev, non_blocking=True) if counts[i]
        else torch.full((), float("nan"), device=dev)
        for i in range(n_layers)
    ]
    token_means = [
        (token_sums[i] / max(total_tokens, 1)).to(dev, non_blocking=True)
        if counts[i] else torch.full((), float("nan"), device=dev)
        for i in range(n_layers)
    ]
    values = torch.stack(answer_means + token_means).cpu()
    per_layer_answer = [float(v) for v in values[:n_layers]]
    per_layer_token = [float(v) for v in values[n_layers:]]
    valid_answer = [v for v in per_layer_answer if v == v]
    valid_token = [v for v in per_layer_token if v == v]
    answer_mean = (sum(valid_answer) / len(valid_answer)
                   if valid_answer else float("nan"))
    token_mean = (sum(valid_token) / len(valid_token)
                  if valid_token else float("nan"))
    return answer_mean, per_layer_answer, token_mean, per_layer_token


def _flush_train_log(log, *, epoch: int, step: int, accum: int,
                     pending: list[list[torch.Tensor]], n_layers: int,
                     token_counts: list[int] | None = None,
                     pending_items: int | None = None, **extra) -> None:
    if not pending:
        return
    answer_loss, per_layer_answer, token_loss, per_layer_token = (
        _summarize_pending_losses(pending, n_layers, token_counts))
    aggregation = extra.get("update_granularity", "legacy_answer_sum")
    reduction = extra.get("update_reduction", aggregation)
    # B×K v3.1 updates use an unaveraged gradient sum, while the reported
    # scalar remains a valid-cell mean so runs with different B/K are
    # comparable. Reduction names describe writes; loss_measure describes
    # telemetry and must not conflate the two.
    use_token = reduction in ("token", "token_mean", "unaveraged_sum_BxK")
    log.log(kind="train", epoch=epoch, step=step, items_seen=accum,
            accum_items=(len(pending) if pending_items is None
                         else pending_items),
            loss=(token_loss if use_token else answer_loss),
            per_layer=(per_layer_token if use_token else per_layer_answer),
            answer_mean_loss=answer_loss,
            per_layer_answer_mean=per_layer_answer,
            token_mean_loss=token_loss,
            per_layer_token_mean=per_layer_token,
            loss_measure=("valid_token_mean" if use_token else "answer_mean"),
            **extra)
    pending.clear()
    if token_counts is not None:
        token_counts.clear()


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
        name: tasks_eval(stack.model, tok, path,
                         n_per_task=EPOCH_RECALL_ITEMS_PER_TASK,
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
            recall_items_per_task=EPOCH_RECALL_ITEMS_PER_TASK,
            next_acc=primary["next_acc"], prev_acc=primary["prev_acc"],
            cloze_acc=primary["cloze_acc"], overall_word_acc=overall,
            vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
            vram_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
            minutes=round((time.time() - started_at) / 60, 1))
    print(" ".join(
        f"{name}: {value['overall_word_acc']:.2f}" for name, value in summary.items()))


def _log_standard_damage(cfg, stack, tok, log, *, epoch: int, phase: str,
                         baseline: dict | None, started_at: float) -> dict:
    """Same-subset fast standard-benchmark probe for epoch-gating a campaign."""
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
    """Always record recall; gate only the optional standard-damage probe."""
    _log_epoch_recall(cfg, stack, tok, log, epoch=0, phase="epoch0",
                      started_at=started_at)
    if not cfg.eval.standard_damage_every_epochs:
        return None
    return _log_standard_damage(cfg, stack, tok, log, epoch=0, phase="epoch0",
                                baseline=None, started_at=started_at)


def _epoch_end_telemetry(cfg, stack, tok, log, *, epoch: int,
                         baseline: dict | None, started_at: float) -> dict | None:
    """Epoch-boundary probes shared by every epoch-driven schedule: fast
    recall on the configured cadence, plus the same-subset standard-damage probe
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
