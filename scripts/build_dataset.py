"""Build data/poem/examples.jsonl: rendered SegmentedExamples for one mask mode.

Usage:
    python scripts/build_dataset.py [--config configs/base.yaml] [--experiment ...]

RAG mode is pure text work. Thinking mode loads the model to harvest <think>
traces (greedy, frozen into the jsonl for reproducibility).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.config import load_config
from selfupdate.data.poem import STYLES, load_poem, make_specs
from selfupdate.masking import RAG_STUB, THINK_STUB, render_rag, render_rag_tool


def build_v5(cfg) -> None:
    """v5 question-only build: multi-corpus conversational questions with a
    master-RAG tool turn and NO answers (the teacher generates them at the
    teacher stage; see src/selfupdate/data/questions.py). Coverage of every
    corpus line is a build invariant; the manifest lands next to the jsonl."""
    from selfupdate.data.questions import coverage_report, make_v5_specs

    if cfg.mask.mode != "rag_tool":
        sys.exit("question_set=v5 requires mask.mode=rag_tool "
                 "(the master RAG is a retrieval-tool turn)")
    corpora = cfg.data.corpora or [{
        "poem_path": cfg.data.poem_path,
        "corpus_style": cfg.data.corpus_style,
        "prefix": "",
    }]
    records, manifest = [], {}
    for corpus in corpora:
        style_name = corpus["corpus_style"]
        style = STYLES[style_name]
        verses = load_poem(corpus["poem_path"])
        specs = make_v5_specs(
            verses,
            style=style,
            corpus_style=style_name,
            rag_scope=cfg.data.rag_scope,
            rag_window_lines=cfg.data.rag_window_lines,
            next_windows=tuple(cfg.data.v5_next_windows),
            prev_stride=cfg.data.v5_prev_stride,
            cloze_block=cfg.data.v5_cloze_block,
            cloze_deletions=tuple(cfg.data.v5_cloze_deletions),
            seed=cfg.data.v5_seed,
        )
        prefix = corpus.get("prefix", "")
        report = coverage_report(specs, len(verses))
        report["poem_path"] = corpus["poem_path"]
        manifest[prefix or style_name] = report
        if report["uncovered_lines"]:
            sys.exit(f"v5 coverage hole in {corpus['poem_path']}: lines "
                     f"{report['uncovered_lines'][:10]}...")
        for s in specs:
            ex = render_rag_tool(
                f"{prefix}-{s.task_id}" if prefix else s.task_id,
                s.question, s.passage, answer="", system=style.system,
                open_answer=True,
            )
            records.append({
                **ex.to_json(),
                "question": s.question,
                "answer_text": "",
                "kind": s.kind,
                "target_lines": list(s.target_lines),
                "expected_answer_chars": s.expected_answer_chars,
                "rag_scope": s.rag_scope,
                "corpus": prefix or style_name,
            })

    out = Path(cfg.data.examples_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    coverage_path = out.with_name(out.stem + "_coverage.json")
    coverage_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    kinds = {}
    for r in records:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"wrote {len(records)} v5 question records "
          f"({cfg.data.rag_scope}-scope RAG; {kinds}) to {out}; "
          f"coverage manifest {coverage_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    if cfg.data.question_set == "v5":
        build_v5(cfg)
        return
    if cfg.data.question_set != "legacy":
        sys.exit(f"unknown data.question_set {cfg.data.question_set!r}")

    specs = make_specs(
        load_poem(cfg.data.poem_path),
        window=cfg.data.window,
        stride=cfg.data.stride,
        include_full=cfg.data.include_full,
        full_lines=cfg.data.full_lines,
        context_pad=cfg.data.context_pad,
        include_sections=cfg.data.include_sections,
        section_max_lines=cfg.data.section_max_lines,
        long_windows=cfg.data.long_windows,
        paraphrase=cfg.data.paraphrase,
        part_chunk_lines=cfg.data.part_chunk_lines,
        catechism=cfg.data.catechism,
        maieutic=cfg.data.maieutic,
        style=STYLES[cfg.data.corpus_style],
    )

    stub = ""
    if cfg.mask.compaction in ("stub", "stub_gap"):
        stub = RAG_STUB if cfg.mask.mode == "rag" else THINK_STUB

    if cfg.mask.mode == "rag":
        examples = [
            render_rag(s.task_id, s.question, s.passage, s.answer, student_stub=stub,
                       system=STYLES[cfg.data.corpus_style].system)
            for s in specs
        ]
    elif cfg.mask.mode == "rag_tool":
        # system was silently dropped here and in the thinking harvest below
        # (fell back to DEFAULT_SYSTEM) — identity for verse datasets, wrong
        # for prose_quijote. v5 fix; v4 artifacts stay byte-guarded.
        examples = [
            render_rag_tool(s.task_id, s.question, s.passage, s.answer,
                            student_stub=stub,
                            system=STYLES[cfg.data.corpus_style].system)
            for s in specs
        ]
    elif cfg.mask.mode in ("thinking", "thinking_selective"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from selfupdate.teacher.generate import harvest_traces

        tok = AutoTokenizer.from_pretrained(cfg.model.name)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, dtype=getattr(torch, cfg.model.dtype)
        ).to(cfg.model.device)
        model.eval()
        selective = cfg.mask.mode == "thinking_selective"
        verses = None
        if selective:
            verses = [v.text for v in load_poem(cfg.data.poem_path)]
        examples = harvest_traces(
            model, tok, specs,
            max_think_tokens=cfg.mask.max_think_tokens, student_stub=stub,
            system=STYLES[cfg.data.corpus_style].system,
            # RAG-in-prompt makes the trace actually QUOTE the passage, so
            # selective censoring has verse spans to remove
            rag_in_prompt=selective, selective_verses=verses,
        )
        if selective:
            n_priv = sum(1 for ex in examples
                         for _, p in (ex.interleaved or []) if p)
            frac = [sum(len(t) for t, p in ex.interleaved or [] if p)
                    / max(sum(len(t) for t, _ in ex.interleaved or []), 1)
                    for ex in examples]
            print(f"selective censoring: {n_priv} privileged runs; "
                  f"mean censored-char fraction "
                  f"{sum(frac)/max(len(frac),1):.2f}; "
                  f"{sum(1 for f in frac if f == 0)} traces with NO quoted "
                  f"verse (fully kept)")
    else:
        sys.exit(f"unknown mask mode {cfg.mask.mode!r}")

    out = Path(cfg.data.examples_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for spec, ex in zip(specs, examples):
            f.write(json.dumps({**ex.to_json(), "answer_text": spec.answer,
                                "question": spec.question}, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples)} examples ({cfg.mask.mode}/{cfg.mask.compaction}) to {out}")


if __name__ == "__main__":
    main()
