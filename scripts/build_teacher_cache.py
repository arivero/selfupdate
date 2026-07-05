"""Precompute the frozen-teacher cache for every example in examples.jsonl.

One fp32 forward per example; stores top-k logits (+ fp32 logsumexp) at the
aligned span. Also runs the M1 premise check:
teacher answer NLL must be low WITH context and high WITHOUT (i.e., the model
needs the context — it does not already know the poem).

Usage:
    python scripts/build_teacher_cache.py [--config configs/base.yaml] [--experiment ...]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.chatfmt import adapt_records
from selfupdate.config import load_config
from selfupdate.masking import ContextMasker, SegmentedExample
from selfupdate.teacher.cache import TeacherCacheWriter, resolve_cache_dir


def load_examples(path: str, tokenizer) -> list[SegmentedExample]:
    records = [json.loads(line)
               for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return [SegmentedExample.from_record(r)
            for r in adapt_records(records, tokenizer)]


def answer_nll(logits: torch.Tensor, ids: list[int], ans: slice) -> float:
    """Mean token NLL of the answer span given the full sequence logits."""
    tgt = torch.tensor(ids[ans.start:ans.stop], device=logits.device)
    pred = logits[ans.start - 1: ans.stop - 1]
    return F.cross_entropy(pred.float(), tgt).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    root, chash = resolve_cache_dir(cfg)
    print(f"cache dir: {root}")

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    examples = load_examples(cfg.data.examples_path, tok)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.float32)
    model.to(cfg.model.device)
    model.eval()
    masker = ContextMasker(tok)
    writer = TeacherCacheWriter(root, chash, shard_size=cfg.cache.shard_size)

    nll_with, nll_without = [], []
    for ex in tqdm(examples, desc="teacher forward"):
        pair = masker.build(ex)
        t_ids = torch.tensor([pair.teacher_ids], device=model.device)
        with torch.no_grad():
            out = model(t_ids, use_cache=False)
        span = pair.t_aligned
        logits = out.logits[0, span.start:span.stop].float()
        logz = torch.logsumexp(logits, dim=-1)
        topk_v, topk_i = logits.topk(cfg.cache.topk, dim=-1)
        writer.add(
            ex.example_id, topk_v, topk_i, logz,
            span={
                "t0": pair.t_aligned.start, "s0": pair.s_aligned.start,
                "A": pair.aligned_len, "mid_len": pair.s_answer.start - pair.s_aligned.start,
                "position_gap": pair.position_gap,
                "n_teacher": len(pair.teacher_ids), "n_student": len(pair.student_ids),
            },
        )

        nll_with.append(answer_nll(out.logits[0], pair.teacher_ids, pair.t_answer))
        s_ids = torch.tensor([pair.student_ids], device=model.device)
        with torch.no_grad():
            s_out = model(s_ids, use_cache=False)
        nll_without.append(answer_nll(s_out.logits[0], pair.student_ids, pair.s_answer))

    writer.finalize()
    mean = lambda xs: sum(xs) / len(xs)
    print(f"wrote {len(examples)} examples to {root}")
    print(f"premise check — teacher answer NLL: with context {mean(nll_with):.3f}, "
          f"without context {mean(nll_without):.3f}")
    (root / "premise.json").write_text(json.dumps({
        "nll_with_context": nll_with, "nll_without_context": nll_without,
        "mean_with": mean(nll_with), "mean_without": mean(nll_without),
    }))


if __name__ == "__main__":
    main()
