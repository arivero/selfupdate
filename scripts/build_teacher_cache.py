"""Precompute the frozen-teacher hidden-state cache for every example.

One fp32 forward per example; stores per-layer hidden states (fp16) at the
aligned span. Also runs the premise check:
teacher answer-CE must be low WITH context and high WITHOUT (i.e., the model
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


def reference_ce(logits: torch.Tensor, ids: list[int], ans: slice) -> float:
    """Mean CE of the answer tokens given the full sequence logits."""
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
    n_layers = model.config.num_hidden_layers

    masker = ContextMasker(tok)
    writer = TeacherCacheWriter(root, chash, shard_size=cfg.cache.shard_size,
                                hidden_dtype=cfg.cache.hidden_dtype)

    ce_with, ce_without = [], []
    for ex in tqdm(examples, desc="teacher forward"):
        pair = masker.build(ex)
        t_ids = torch.tensor([pair.teacher_ids], device=model.device)
        with torch.no_grad():
            out = model(t_ids, output_hidden_states=True, use_cache=False)
        span = pair.t_aligned
        hidden = {
            L: out.hidden_states[L][0, span.start:span.stop]
            for L in range(1, n_layers + 1)
        }
        writer.add(
            ex.example_id, hidden,
            span={
                "t0": pair.t_aligned.start, "s0": pair.s_aligned.start,
                "A": pair.aligned_len, "mid_len": pair.s_answer.start - pair.s_aligned.start,
                "position_gap": pair.position_gap,
                "n_teacher": len(pair.teacher_ids), "n_student": len(pair.student_ids),
            },
        )

        ce_with.append(reference_ce(out.logits[0], pair.teacher_ids, pair.t_answer))
        s_ids = torch.tensor([pair.student_ids], device=model.device)
        with torch.no_grad():
            s_out = model(s_ids, use_cache=False)
        ce_without.append(reference_ce(s_out.logits[0], pair.student_ids, pair.s_answer))

    writer.finalize()
    mean = lambda xs: sum(xs) / len(xs)
    print(f"wrote {len(examples)} examples, {n_layers} layers each, to {root}")
    print(f"premise check — teacher answer CE: with context {mean(ce_with):.3f}, "
          f"without context {mean(ce_without):.3f}")
    (root / "premise.json").write_text(json.dumps({
        "ce_with_context": ce_with, "ce_without_context": ce_without,
        "mean_with": mean(ce_with), "mean_without": mean(ce_without),
    }))


if __name__ == "__main__":
    main()
