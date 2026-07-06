"""Teacher/RAG ceiling: base model recall WITH the RAG passage in context.

The copying ceiling for every student arm: prompt = shared_prefix +
privileged + shared_mid, greedy, full recite metrics. This is the unmodified
model with only the privileged RAG input added.

Thin entrypoint over ``selfupdate.eval.recite.recite_eval`` — the same
engine evaluate.py uses (batched greedy generation, OOM backoff, threaded
CER scoring). The teacher-specific parts are only: the UNCENSORED record
view (the teacher always sees the privileged content), deterministic
``--num-shards`` interleaving for multi-GPU reference runs, and the
teacher-reference JSON envelope.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/teacher_ceiling.py \
        --experiment configs/experiments/lw_r_slide8_0p6b_rag.yaml \
        --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.chatfmt import adapt_records
from selfupdate.config import load_config
from selfupdate.data.dataset import load_jsonl
from selfupdate.eval.recite import recite_eval


def teacher_view(record: dict) -> dict:
    """The teacher is never censored: the RAG block goes into its prompt
    (student_stub := privileged), and thinking_selective records expose ALL
    think runs, kept and censored alike."""
    if record.get("interleaved"):
        return {**record, "interleaved": [[t, False] for t, _ in record["interleaved"]]}
    return {**record, "student_stub": record.get("privileged", ""), "interleaved": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-extra-tokens", type=int, default=48)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--auto-map", action="store_true",
                    help="load with device_map=auto for large two-card teacher references")
    ap.add_argument("--bucket-by-length", action="store_true",
                    help="throughput mode: sort examples by reference length inside each shard")
    ap.add_argument("--score-workers", type=int, default=None,
                    help="CPU workers for CER scoring in batched eval")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="fixed random order for batched eval; results are restored by example index")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, dtype=torch.bfloat16,
        device_map="auto" if args.auto_map else None)
    if not args.auto_map:
        model.to(cfg.model.device)
    model.eval()

    # adapt once here (idempotent — recite_eval's internal call is then the
    # identity fast path), so teacher_view maps the re-rendered segments
    raw = adapt_records(load_jsonl(cfg.data.examples_path), tok)
    records = [teacher_view(r) for r in raw]
    # even subsampling across the corpus; recite_eval's own limit takes the
    # head, which would bias references toward the corpus opening
    if args.limit and args.limit < len(records):
        step = max(1, len(records) // args.limit)
        records = records[::step][: args.limit]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    if args.num_shards > 1:
        # Deterministic interleaving preserves corpus order modulo sharding:
        # shard 0 gets examples 0,4,8...; shard 1 gets 1,5,9..., etc.
        # This avoids randomizing or length-sorting the default evaluation.
        records = records[args.shard_index::args.num_shards]

    r = recite_eval(model, tok, records,
                    batch_size=args.batch_size,
                    max_extra_tokens=args.max_extra_tokens,
                    bucket_by_length=args.bucket_by_length,
                    score_workers=args.score_workers,
                    shuffle_seed=args.shuffle_seed)
    model_short = cfg.model.name.split("/")[-1]
    data_stem = Path(cfg.data.examples_path).stem
    r["teacher_reference_kind"] = "teacher_epoch0_rag_context"
    r["model"] = cfg.model.name
    r["examples_path"] = cfg.data.examples_path
    r["batch_size"] = args.batch_size
    r["bucket_by_length"] = args.bucket_by_length
    r["score_workers"] = args.score_workers
    r["shuffle_seed"] = args.shuffle_seed
    r["num_shards"] = args.num_shards
    r["shard_index"] = args.shard_index
    print(f"TEACHER RAG REFERENCE {model_short} x {data_stem}: n={r['n']} "
          f"cer {r['cer']:.4f} cer_flat {r['cer_flat']:.4f} "
          f"line_exact {r['line_exact']:.4f}")

    out = Path(args.out or f"runs/teacher_ref_rag_{model_short}_{data_stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
