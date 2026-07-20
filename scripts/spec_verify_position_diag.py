"""Per-position vLLM-reproduction divergence diagnostic (pipeline-v4).

Mirrors the 26B divergence diagnostic method (see runs/spec_verify/RESULTS.md,
"26B divergence diagnostic": commit 182f46f). That diagnostic reused the real
production v4 code path -- TrainingRuntime (model/LoRA load), DistillDataset,
_V4Cohort, _online_teacher_capture, stack.lm_head -- run B=1 per item with
v4_teacher_source: online, to surface PER-POSITION divergence detail that the
production ``teacher_output_eval_sums`` aggregate-only call (src/selfupdate/
eval/teacher_output.py) does not return.

This script is a fresh reproduction of that same method (the original 26B
script was not committed; only its RESULTS.md write-up survives in git
history). It does NOT reimplement the forward pass, tokenization, or masking:
every tensor here comes from the same ``_V4Cohort``/``_online_teacher_capture``
functions the real trainer uses, run with B=1 (one item per cohort) instead of
the production micro_batch to sidestep the stage-scoped memory budget, exactly
as the 26B diagnostic's method note describes.

Usage (training venv, one free GPU pinned via CUDA_VISIBLE_DEVICES):
  CUDA_VISIBLE_DEVICES=2 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  TRANSFORMERS_VERBOSITY=error /tmp/$USER/selfupdate-venv/bin/python \
    scripts/spec_verify_position_diag.py \
      --config configs/experiments/spec_verify/base_31b_v4_spec.yaml \
      --device cuda:0 \
      --out runs/spec_verify/31b_position_diag_full2071.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selfupdate.utils.env import cap_cpu_threads

cap_cpu_threads()

import torch

from selfupdate.config import load_config
from selfupdate.train.runtime import TrainingRuntime
from selfupdate.data.dataset import DistillDataset
from selfupdate.train.moe import dequantize_overrides
from selfupdate.train.online_v4 import (
    _V4Cohort,
    _online_teacher_capture,
    _owned_range,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="the model's base_*_v4_spec.yaml (used ALONE, no "
                         "PPPn overlay -- its own defaults are already "
                         "v4_teacher_source=online, v4_stage_scoped=false, "
                         "v4_stage=-1, i.e. single full-model residency)")
    ap.add_argument("--device", default="cuda:0",
                    help="physical device string AFTER CUDA_VISIBLE_DEVICES "
                         "masking; pair with CUDA_VISIBLE_DEVICES to pin a "
                         "specific free card")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = whole dataset (2071 items = one full epoch)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.train.v4_teacher_source != "online":
        sys.exit(f"expected v4_teacher_source=online in {args.config}, "
                  f"got {cfg.train.v4_teacher_source!r}")
    if cfg.train.v4_stage_scoped:
        sys.exit("this diagnostic needs v4_stage_scoped=false (single "
                  "full-model residency, B=1 avoids the OOM the production "
                  "micro_batch would hit)")
    cfg.model.device = args.device
    # No training steps run in this script; the batching/optimizer knobs
    # (micro_batch, v4_optimizer, epochs) are irrelevant here.

    moe_load_kw = dequantize_overrides(cfg.model.name, cfg.train.moe_mode)
    print(f"loading {cfg.model.name} onto {cfg.model.device} "
          f"(v4_teacher_source={cfg.train.v4_teacher_source})", flush=True)
    rt = TrainingRuntime(cfg).load(moe_load_kw)
    tok, stack = rt.tokenizer, rt.stack
    cache = rt.load_cache()
    print(f"loaded; n_layers={stack.n_layers} cache_root={cache.root}",
          flush=True)

    ds = DistillDataset(
        cfg.data.examples_path, cache, tok,
        need_layers=[],
        with_teacher_ids=False,
        pad_random=False,
        cache_source_compaction=cfg.cache.source_compaction,
        student_compaction=cfg.mask.compaction,
        item_cache_items=cfg.cache.item_cache_items,
    )
    n = stack.n_layers
    owned = _owned_range(cfg, n)
    if not (owned.start == 1 and owned.stop - 1 == n):
        sys.exit(f"diagnostic needs single-stage full ownership; got "
                  f"owned={owned.start}..{owned.stop - 1} of {n}")

    peft_model = rt.peft_model
    adapters_off = (peft_model.disable_adapter if peft_model is not None
                    else None)
    if adapters_off is None:
        sys.exit("expected an attached (zero-init) LoRA adapter for the "
                 "adapters-off teacher-side forward, per cfg.train.lora")

    device = torch.device(cfg.model.device)
    items = len(ds.pairs) if not args.limit else min(args.limit, len(ds.pairs))
    print(f"traversing {items} items (dataset total {len(ds.pairs)})",
          flush=True)

    total_tokens = 0
    total_match = 0
    total_first = 0
    first_total = 0
    exact_items = 0
    items_traversed = 0
    answer_lengths: list[int] = []
    divergences: list[dict] = []   # every wrong position, every divergent item

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for i in range(items):
        cohort = _V4Cohort(cfg, ds, [i], device)
        capture = _online_teacher_capture(
            cfg, stack, adapters_off, cohort, owned, device, n)
        rows = capture["eval_rows_teacher"][0]        # [L_i, H] post-norm
        ids = cohort.eval_ids[0].to(device)            # [L_i] vLLM answer ids
        L = int(rows.shape[0])
        # Zero-length answers count toward the exact-seq DENOMINATOR (parity
        # with the production ev["answers"] += len(cohort.indices) count,
        # which includes every cohort item regardless of eval length) but
        # never toward the numerator (teacher_output_eval_sums skips length-0
        # rows entirely, so they can never read as "exact") and have no
        # position-0 to score for first-token-agree.
        items_traversed += 1
        if L == 0:
            del capture, rows, ids
            continue
        with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=(device.type == "cuda")):
            logits = stack.lm_head(rows)
        logits = logits.float()
        top2 = logits.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        pred = logits.argmax(-1)
        match = pred == ids

        n_match = int(match.sum().item())
        total_tokens += L
        total_match += n_match
        first_total += 1
        total_first += int(match[0].item())
        answer_lengths.append(L)

        if bool(match.all()):
            exact_items += 1
        else:
            example_id = cohort.example_ids[0]
            for j in range(L):
                if not bool(match[j].item()):
                    divergences.append({
                        "example_id": example_id,
                        "pos": j,
                        "pos_from_end": L - 1 - j,
                        "length": L,
                        "vllm_id": int(ids[j].item()),
                        "pred_id": int(pred[j].item()),
                        "margin": round(float(margin[j].item()), 4),
                    })

        del capture, rows, ids, logits, top2, margin, pred, match, cohort
        if args.progress_every and (i + 1) % args.progress_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{i + 1}/{items}] tokens={total_tokens} "
                  f"match={total_match} divergent_tokens={len(divergences)} "
                  f"elapsed={elapsed:.1f}s ({(i + 1) / elapsed:.2f} items/s)",
                  flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0

    # Dump raw divergences BEFORE any assert can fire (advisor-flagged: a
    # crash here after a 20-40 min traversal must not destroy the evidence
    # needed to debug it).
    if args.out:
        raw_outp = Path(args.out)
        raw_outp.parent.mkdir(parents=True, exist_ok=True)
        raw_outp.write_text(json.dumps({
            "partial": True,
            "items_traversed": items_traversed,
            "exact_items": exact_items,
            "divergences": divergences,
        }, indent=2))
        print(f"wrote raw divergences (pre-assert) to {raw_outp}", flush=True)

    n_items = len(answer_lengths)          # items with at least one eval row
    divergent_by_item: dict[str, list[dict]] = {}
    for d in divergences:
        divergent_by_item.setdefault(d["example_id"], []).append(d)
    n_divergent_items = len(divergent_by_item)

    # Self-consistency asserts (advisor-flagged: an 8-item smoke almost never
    # exercises this aggregation branch, since ~97% of items are exact — so
    # gate the FULL run's own numbers, don't just trust a clean smoke).
    assert exact_items + n_divergent_items == n_items, (
        f"every non-exact item must appear in the divergence map: "
        f"{exact_items} + {n_divergent_items} != {n_items}")

    sole_pos0 = sum(1 for eid, ds_ in divergent_by_item.items()
                    if len(ds_) == 1 and ds_[0]["pos"] == 0)
    sole_last = sum(1 for eid, ds_ in divergent_by_item.items()
                    if len(ds_) == 1 and ds_[0]["pos_from_end"] == 0)
    div_pos0 = sum(1 for d in divergences if d["pos"] == 0)
    div_last = sum(1 for d in divergences if d["pos_from_end"] == 0)

    # depth-quartile histogram over the divergent tokens (same convention as
    # the 26B write-up and verify_vllm_teacher_forced.py: relative position
    # within THIS answer's length, quartile 0..3 low->high depth)
    quartiles = [0, 0, 0, 0]
    for d in divergences:
        q = min(3, (4 * d["pos"]) // max(d["length"], 1))
        quartiles[q] += 1
    assert sum(quartiles) == len(divergences), (
        f"quartile histogram {quartiles} does not sum to "
        f"{len(divergences)} divergent tokens")
    assert div_pos0 <= n_divergent_items or n_divergent_items == 0, (
        "pos0 divergent-token count exceeds divergent-answer count "
        "(a single answer can diverge at pos0 at most once)")
    assert sole_pos0 <= div_pos0, "sole-pos0 answers exceed pos0 tokens"

    terminal_vllm_ids: dict[str, int] = {}
    terminal_pred_ids: dict[str, int] = {}
    for d in divergences:
        if d["pos_from_end"] == 0:
            terminal_vllm_ids[str(d["vllm_id"])] = (
                terminal_vllm_ids.get(str(d["vllm_id"]), 0) + 1)
            terminal_pred_ids[str(d["pred_id"])] = (
                terminal_pred_ids.get(str(d["pred_id"]), 0) + 1)

    summary = {
        "model": cfg.model.name,
        "config": args.config,
        "method": "v4_online_b1_per_item_position_diag",
        "items": n_items,
        "items_traversed": items_traversed,
        "total_answer_tokens": total_tokens,
        "teacher_argmax_acceptance": total_match / max(total_tokens, 1),
        # Denominator is items_traversed (production parity: ev["answers"]
        # counts every cohort item regardless of eval length), not n_items
        # (items with >=1 eval row) -- matches the published PPP1 metric's
        # own denominator so the two are directly comparable.
        "exact_seq_rate": exact_items / max(items_traversed, 1),
        "exact_seq_match_answers": exact_items,
        "first_token_agree_rate": total_first / max(first_total, 1),
        "n_divergent_tokens": len(divergences),
        "n_divergent_answers": n_divergent_items,
        "divergent_tokens_at_pos0": div_pos0,
        "divergent_tokens_at_pos0_frac": div_pos0 / max(len(divergences), 1),
        "divergent_tokens_at_last_pos": div_last,
        "divergent_tokens_at_last_pos_frac": div_last / max(len(divergences), 1),
        "depth_quartile_histogram": quartiles,
        "divergent_answers_sole_pos0": sole_pos0,
        "divergent_answers_sole_pos0_frac": sole_pos0 / max(n_divergent_items, 1),
        "divergent_answers_sole_last": sole_last,
        "divergent_answers_sole_last_frac": sole_last / max(n_divergent_items, 1),
        "terminal_vllm_id_histogram": terminal_vllm_ids,
        "terminal_pred_id_histogram": terminal_pred_ids,
        "seconds": round(dt, 3),
        "items_per_s": round(n_items / max(dt, 1e-9), 3),
        "dataset_item_count": len(ds.pairs),
    }
    print(json.dumps(summary, indent=2), flush=True)

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({
            "summary": summary,
            "divergences": divergences,
        }, indent=2))
        print(f"\nwrote {outp}", flush=True)


if __name__ == "__main__":
    main()
