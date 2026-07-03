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
from selfupdate.data.poem import load_poem, make_specs
from selfupdate.masking import RAG_STUB, THINK_STUB, render_rag, render_rag_tool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

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
    )

    stub = ""
    if cfg.mask.compaction in ("stub", "stub_gap"):
        stub = RAG_STUB if cfg.mask.mode == "rag" else THINK_STUB

    if cfg.mask.mode == "rag":
        examples = [
            render_rag(s.task_id, s.question, s.passage, s.answer, student_stub=stub)
            for s in specs
        ]
    elif cfg.mask.mode == "rag_tool":
        examples = [
            render_rag_tool(s.task_id, s.question, s.passage, s.answer, student_stub=stub)
            for s in specs
        ]
    elif cfg.mask.mode in ("thinking", "rag_thinking"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from selfupdate.teacher.generate import harvest_rag_thinking_traces, harvest_traces

        tok = AutoTokenizer.from_pretrained(cfg.model.name)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, dtype=getattr(torch, cfg.model.dtype)
        ).to(cfg.model.device)
        model.eval()
        harvest = harvest_rag_thinking_traces if cfg.mask.mode == "rag_thinking" else harvest_traces
        examples = harvest(
            model, tok, specs,
            max_think_tokens=cfg.mask.max_think_tokens, student_stub=stub,
        )
    else:
        sys.exit(f"unknown mask mode {cfg.mask.mode!r}")

    out = Path(cfg.data.examples_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for spec, ex in zip(specs, examples):
            answer_text = spec.answer
            if cfg.mask.mode == "rag_thinking":
                answer_text = ex.answer.removesuffix("<|im_end|>")
            f.write(json.dumps({**ex.to_json(), "answer_text": answer_text,
                                "question": spec.question}, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples)} examples ({cfg.mask.mode}/{cfg.mask.compaction}) to {out}")


if __name__ == "__main__":
    main()
