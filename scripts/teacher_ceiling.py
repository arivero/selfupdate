"""Teacher/RAG ceiling: base model recall WITH the RAG passage in context.

The copying ceiling for every student arm: prompt = shared_prefix +
privileged + shared_mid, greedy, full recite metrics. This is the unmodified
model with only the privileged RAG input added.

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

import jiwer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.chatfmt import adapt_records, stop_token_id
from selfupdate.config import load_config
from selfupdate.data.dataset import load_jsonl
from selfupdate.eval.recite import normalize_verse, strip_think, student_prompt


def teacher_view(record: dict) -> dict:
    if record.get("interleaved"):
        # thinking_selective: teacher sees ALL runs (kept + censored)
        return {**record, "interleaved": [[t, False] for t, _ in record["interleaved"]]}
    return {**record, "student_stub": record.get("privileged", ""), "interleaved": None}


def score_one(record: dict, text: str) -> dict:
    ref = normalize_verse(record["answer_text"])
    text = normalize_verse(strip_think(text))
    cer = jiwer.cer(ref, text) if text else 1.0
    cer_flat = (jiwer.cer(ref.replace("\n", " "), text.replace("\n", " "))
                if text else 1.0)
    ref_lines = ref.split("\n")
    got_lines = text.split("\n")
    exact = sum(1 for g, h in zip(ref_lines, got_lines) if g == h)
    prefix = 0
    for g, h in zip(ref_lines, got_lines):
        if g != h:
            break
        prefix += 1
    return {
        "example_id": record["example_id"],
        "cer": cer,
        "cer_flat": cer_flat,
        "line_exact": exact / len(ref_lines),
        "prefix_lines": prefix,
        "n_reference_lines": len(ref_lines),
        "text": text,
    }


@torch.no_grad()
def batched_ceiling(model, tok, records: list[dict], batch_size: int,
                    max_extra_tokens: int, bucket_by_length: bool = False) -> dict:
    was_padding = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos = stop_token_id(tok)

    work = []
    for i, r in enumerate(records):
        prompt = student_prompt(r)
        ref_len = len(tok.encode(r["answer_text"], add_special_tokens=False))
        work.append((ref_len, i, r, prompt))
    if bucket_by_length:
        # Optional throughput mode: keeps max_new_tokens tighter inside each
        # batch. Default evaluation preserves corpus order.
        work.sort(key=lambda x: x[0])
    results: list[dict | None] = [None] * len(records)
    for start in range(0, len(work), batch_size):
        batch = work[start: start + batch_size]
        print(f"teacher ceiling batch {start + len(batch)}/{len(work)} "
              f"(batch_size={len(batch)})", flush=True)
        prompts = [x[3] for x in batch]
        max_new = max(x[0] for x in batch) + max_extra_tokens
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=eos,
            pad_token_id=tok.pad_token_id,
        )
        gen_start = enc["input_ids"].shape[1]
        for row, (_, idx, rec, _prompt) in enumerate(batch):
            text = tok.decode(out[row, gen_start:], skip_special_tokens=True)
            results[idx] = score_one(rec, text)
    tok.padding_side = was_padding
    final = [r for r in results if r is not None]
    mean = lambda k: sum(r[k] for r in final) / len(final)
    return {
        "cer": mean("cer"),
        "cer_flat": mean("cer_flat"),
        "line_exact": mean("line_exact"),
        "prefix_lines": mean("prefix_lines"),
        "n": len(final),
        "per_example": final,
    }


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

    raw = adapt_records(load_jsonl(cfg.data.examples_path), tok)
    records = [teacher_view(r) for r in raw]
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
    r = batched_ceiling(model, tok, records, args.batch_size, args.max_extra_tokens,
                        bucket_by_length=args.bucket_by_length)
    model_short = cfg.model.name.split("/")[-1]
    data_stem = Path(cfg.data.examples_path).stem
    r["teacher_reference_kind"] = "teacher_epoch0_rag_context"
    r["model"] = cfg.model.name
    r["examples_path"] = cfg.data.examples_path
    r["batch_size"] = args.batch_size
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
