"""Precompute the frozen-teacher hidden-state cache for every example.

One fp32 forward per example; stores per-layer hidden states (fp16) at the
aligned span. Also runs the premise check:
teacher answer-CE must be low WITH context and high WITHOUT (i.e., the model
needs the context — it does not already know the poem).

v5 open-answer records (empty ``answer``; see data/questions.py) add a
GENERATION step before the forward: the teacher, holding the master-RAG
tool turn, greedily generates its answer (stop at the turn closer, hard cut
at 2x the record's expected length); the generated ids become the aligned
span for the teacher-forced forward and are stored in the cache index —
answers are cache content (per-model), never dataset content. For these
records the premise pair reads differently: CE-with-context of the model's
own greedy generation is near zero by construction, so the CONTRAST
against CE-without-context measures how much the generation actually
depended on the RAG — a low contrast flags a non-attending teacher (the
RAG-authority failure, issues.md 2026-07-12). A per-item recitation report
(generation vs the targeted corpus span) lands in generation_report.json.

Usage:
    python scripts/build_teacher_cache.py [--config configs/base.yaml] [--experiment ...]
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.utils.env import cap_cpu_threads  # noqa: E402

cap_cpu_threads()

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.chatfmt import adapt_records, stop_token_id
from selfupdate.config import load_config
from selfupdate.masking import ContextMasker, SegmentedExample
from selfupdate.teacher.cache import TeacherCacheWriter, resolve_cache_dir


def load_records(path: str, tokenizer) -> list[dict]:
    records = [json.loads(line)
               for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return adapt_records(records, tokenizer)


def reference_ce(logits: torch.Tensor, ids: list[int], ans: slice) -> float:
    """Mean CE of the answer tokens given the full sequence logits."""
    tgt = torch.tensor(ids[ans.start:ans.stop], device=logits.device)
    pred = logits[ans.start - 1: ans.stop - 1]
    return F.cross_entropy(pred.float(), tgt).item()


def _generation_budget(masker: ContextMasker, ex: SegmentedExample,
                       expected_chars: int, extra_tokens: int) -> int:
    """Token budget = 2x the expected answer length + a fixed conversational
margin. The chars-per-token ratio is measured on this record's own
    passage, so the budget adapts to the corpus without anchoring the
    dataset to a tokenizer. The margin absorbs answer FRAMING ("Sí, el
    verso que sigue es: ..."): the 0.6B smoke measured 91.7%% hard-cuts at
    +8 because preamble ate the budget before the quoted answer finished —
a cut span teaches the student truncated behavior, so short answers
must be able to terminate naturally; the explicit margin comes from
``cache.generation_extra_tokens`` and the 2x proportional control is
unchanged."""
    priv_ids = masker._encode(ex.privileged)
    ratio = (len(priv_ids) / max(len(ex.privileged), 1)) if priv_ids else 0.35
    est = max(4, math.ceil(expected_chars * ratio))
    return 2 * est + extra_tokens


@torch.no_grad()
def generate_answer(model, masker, ex, stop_id: int,
                    max_new: int) -> tuple[list[int], bool]:
    """Greedy teacher answer over the full teacher prompt (RAG included).
    Returns (answer_ids ending in the turn closer, hard_cut)."""
    prompt = masker.build(ex).teacher_ids  # empty answer: ends at mid end
    input_ids = torch.tensor([prompt], device=model.device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new,
        do_sample=False,
        eos_token_id=stop_id,
        pad_token_id=stop_id,
    )
    gen = out[0, len(prompt):].tolist()
    if stop_id in gen:
        return gen[: gen.index(stop_id) + 1], False
    return gen + [stop_id], True


def _corpus_texts(examples_path: Path) -> dict[str, list[str]]:
    """prefix -> corpus lines, resolved through the coverage manifest the
    v5 builder writes next to the jsonl (the record itself carries no
    corpus text by design)."""
    from selfupdate.data.poem import load_poem

    manifest_path = examples_path.with_name(
        examples_path.stem + "_coverage.json")
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {prefix: [v.text for v in load_poem(entry["poem_path"])]
            for prefix, entry in manifest.items() if "poem_path" in entry}


def _recitation_stats(record: dict, answer_text: str,
                      corpora: dict[str, list[str]]) -> dict:
    """Teacher-recitation telemetry (never a gate): word-LCS against the
    targeted span for next/prev; word containment in the target block for
    cloze (the deleted words are not stored anywhere — by design)."""
    from selfupdate.eval.tasks import _words, score

    texts = corpora.get(record.get("corpus", ""))
    if texts is None or "target_lines" not in record:
        return {}
    lo, hi = record["target_lines"]
    target = "\n".join(texts[lo:hi])
    if record.get("kind") == "cloze":
        block = set(_words(target))
        gen_words = _words(answer_text)
        contained = sum(1 for w in gen_words if w in block)
        return {"containment": contained / max(len(gen_words), 1)}
    return {"word_acc": score(target, answer_text)["word_acc"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    root, chash = resolve_cache_dir(cfg)
    print(f"cache dir: {root}")

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    records = load_records(cfg.data.examples_path, tok)
    examples = [SegmentedExample.from_record(r) for r in records]
    open_answer = [not ex.answer for ex in examples]
    if any(open_answer) and not all(open_answer):
        sys.exit("mixed open-answer/legacy records in one jsonl — rebuild")
    v5 = all(open_answer) and bool(examples)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.float32)
    model.to(cfg.model.device)
    model.eval()
    n_layers = model.config.num_hidden_layers

    masker = ContextMasker(tok, pad_random=(cfg.mask.compaction == "pad_random"))
    writer = TeacherCacheWriter(root, chash, shard_size=cfg.cache.shard_size,
                                hidden_dtype=cfg.cache.hidden_dtype)
    stop_id = stop_token_id(tok)
    corpora = _corpus_texts(Path(cfg.data.examples_path)) if v5 else {}

    ce_with, ce_without = [], []
    gen_report = []
    for record, ex in tqdm(list(zip(records, examples)), desc="teacher forward"):
        extra = None
        if v5:
            budget = _generation_budget(
                masker, ex, int(record.get("expected_answer_chars", 64)),
                cfg.cache.generation_extra_tokens)
            answer_ids, hard_cut = generate_answer(
                model, masker, ex, stop_id, budget)
            answer_text = tok.decode(answer_ids[:-1])
            pair = masker.build(ex, answer_ids=answer_ids)
            extra = {"answer_ids": answer_ids, "hard_cut": hard_cut}
            gen_report.append({
                "example_id": ex.example_id,
                "kind": record.get("kind"),
                "corpus": record.get("corpus"),
                "gen_tokens": len(answer_ids),
                "hard_cut": hard_cut,
                "answer_text": answer_text,
                **_recitation_stats(record, answer_text, corpora),
            })
        else:
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
            extra=extra,
        )

        ce_with.append(reference_ce(out.logits[0], pair.teacher_ids, pair.t_answer))
        s_ids = torch.tensor([pair.student_ids], device=model.device)
        with torch.no_grad():
            s_out = model(s_ids, use_cache=False)
        ce_without.append(reference_ce(s_out.logits[0], pair.student_ids, pair.s_answer))

    writer.finalize()
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"wrote {len(examples)} examples, {n_layers} layers each, to {root}")
    print(f"premise check — teacher answer CE: with context {mean(ce_with):.3f}, "
          f"without context {mean(ce_without):.3f}")
    (root / "premise.json").write_text(json.dumps({
        "ce_with_context": ce_with, "ce_without_context": ce_without,
        "mean_with": mean(ce_with), "mean_without": mean(ce_without),
    }))
    if v5:
        accs = [g["word_acc"] for g in gen_report if "word_acc" in g]
        cuts = [g for g in gen_report if g["hard_cut"]]
        summary = {
            "n": len(gen_report),
            "mean_word_acc_nextprev": mean(accs),
            "hard_cut_fraction": len(cuts) / max(len(gen_report), 1),
            "mean_gen_tokens": mean([g["gen_tokens"] for g in gen_report]),
        }
        (root / "generation_report.json").write_text(json.dumps(
            {"summary": summary, "items": gen_report}, ensure_ascii=False))
        print(f"teacher recitation — next/prev word-LCS {summary['mean_word_acc_nextprev']:.3f}, "
              f"hard-cut {summary['hard_cut_fraction']:.1%}, "
              f"mean gen len {summary['mean_gen_tokens']:.0f} tokens")


if __name__ == "__main__":
    main()
