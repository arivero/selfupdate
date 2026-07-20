"""Epoch-zero RAG censorship evaluation through programmatic vLLM.

This is the pipeline-v2 replacement for the historical Transformers
``model.generate`` evaluator.  The old implementation is recoverable from
Git history.  It builds the same deterministic next/prev/cloze prompts,
censorship contexts, budgets, and JSON result surface as ``tasks_eval``, but
submits mixed per-prompt budgets to vLLM's continuous scheduler.
"""

from __future__ import annotations


# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP



import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from selfupdate.chatfmt import stop_token_id  # noqa: E402
from selfupdate.config import load_config  # noqa: E402
from selfupdate.eval.tasks import (  # noqa: E402
    QUESTIONS,
    build_tasks,
    corpus_blocks,
    retrieve_chapter,
    retrieve_window,
    score,
)
from selfupdate.masking import (  # noqa: E402
    random_fill_ids,
    render_rag_system,
    render_rag_tool,
)

CORPUS_PATHS = {
    "machado": "data/poem/raw.txt",
    "quijote_ch1": "data/quijote/raw_ch1.txt",
    "quijote_ch4": "data/quijote/raw_ch4.txt",
    "quijote_ch8": "data/quijote/raw_ch8.txt",
    "quijote_ch16": "data/quijote/raw_ch16.txt",
}


def strip_think(text: str) -> str:
    """Return the final answer after an optional reasoning envelope."""
    value = text.lstrip()
    if value.startswith("<think>"):
        end = value.find("</think>")
        return "" if end == -1 else value[end + len("</think>"):]
    if value.startswith("analysis"):
        end = value.find("assistantfinal")
        return "" if end == -1 else value[end + len("assistantfinal"):]
    return text


def _contexts(tokenizer, poem_path: str, items: list[dict], scope: str,
              *, window_lines: int, pad_random: bool, wrong: bool,
              seed: int) -> list[str] | None:
    if scope == "none":
        contexts = None
    elif scope == "full":
        contexts = [Path(poem_path).read_text(encoding="utf-8")] * len(items)
    elif scope == "window":
        lines = [line for block in corpus_blocks(poem_path) for line in block]
        contexts = [retrieve_window(lines, item["block"], pad=window_lines)
                    for item in items]
    elif scope == "chapter":
        contexts = [retrieve_chapter(poem_path, item["block"]) for item in items]
    else:
        raise ValueError(f"unknown context scope {scope!r}")
    if pad_random and wrong:
        raise ValueError("context-pad-random and context-wrong are exclusive")
    if pad_random:
        if contexts is None:
            raise ValueError("context-pad-random needs a real context scope for sizing")
        contexts = [
            tokenizer.decode(random_fill_ids(
                tokenizer, f"evalfloor-{seed}-{i}",
                len(tokenizer.encode(text, add_special_tokens=False))))
            for i, text in enumerate(contexts)
        ]
    elif wrong:
        if contexts is None:
            raise ValueError("context-wrong needs a real context scope")
        k = len(contexts) // 2
        contexts = contexts[k:] + contexts[:k]
    return contexts


def _prepare(tokenizer, poem_path: str, *, seed: int, n_per_task: int,
             max_extra_tokens: int, budget_multiplier: float,
             context_scope: str, context_window_lines: int,
             context_pad_random: bool, context_wrong: bool,
             prompt_regime: str) -> list[dict]:
    items = build_tasks(poem_path, seed=seed, n_per_task=n_per_task)
    contexts = _contexts(
        tokenizer, poem_path, items, context_scope,
        window_lines=context_window_lines, pad_random=context_pad_random,
        wrong=context_wrong, seed=seed)
    stop_id = stop_token_id(tokenizer)
    prepared = []
    for i, item in enumerate(items):
        question = QUESTIONS[item["kind"]].format(x=item["x"], n=item["n"])
        context = contexts[i] if contexts is not None else ""
        if prompt_regime == "rag_system":
            ex = render_rag_system(f"eval-{i}", question, context,
                                   answer="", open_answer=True)
            prompt = ex.shared_prefix + ex.privileged + ex.shared_mid
        elif prompt_regime == "rag_tool":
            ex = render_rag_tool(f"eval-{i}", question, context,
                                 answer="", open_answer=True)
            prompt = ex.shared_prefix + ex.privileged + ex.shared_mid
        else:
            content = (f"{question}\n\nDocumento recuperado:\n{context}"
                       if contexts is not None else question)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        basis = context if contexts is not None else item["reference"]
        budget = (math.ceil(len(tokenizer.encode(basis)) * budget_multiplier)
                  + max_extra_tokens)
        prepared.append({"item": item, "question": question,
                         "prompt_ids": tokenizer.encode(
                             prompt, add_special_tokens=False),
                         "budget": budget, "stop_id": stop_id})
    return prepared


def _score(prepared: list[dict], answers: list[str], completions: list[dict],
           *, seed: int, n_per_task: int, context_scope: str,
           context_window_lines: int, context_pad_random: bool,
           context_wrong: bool, prompt_regime: str, keep_examples: int = 6) -> dict:
    aggregate: dict[str, list[dict]] = {}
    examples = []
    for row, raw, completion in zip(prepared, answers, completions):
        item = row["item"]
        answer = strip_think(raw)
        metrics = score(item["reference"], answer)
        metrics["n_deleted"] = item["n"]
        aggregate.setdefault(item["task"], []).append(metrics)
        if len(examples) < keep_examples:
            examples.append({"kind": item["kind"], "q": row["question"],
                             "reference": item["reference"],
                             "answer": answer.strip()[:200],
                             **completion, **metrics})
    tasks = {}
    for task, rows in aggregate.items():
        tasks[task] = {
            "n": len(rows),
            "exact": sum(row["exact"] for row in rows) / len(rows),
            "word_acc": sum(row["word_acc"] for row in rows) / len(rows),
        }
    if "cloze" in aggregate:
        grouped: dict[int, list[float]] = {}
        for row in aggregate["cloze"]:
            grouped.setdefault(row["n_deleted"], []).append(row["word_acc"])
        tasks["cloze"]["by_deletions"] = {
            str(n): sum(values) / len(values)
            for n, values in sorted(grouped.items())}
    bounded = [meta for row, meta in zip(prepared, completions)
               if row["item"]["kind"] not in {"start_block", "end_block"}]
    return {
        "seed": seed,
        "n_per_task": n_per_task,
        "generation_backend": "vllm",
        "with_context": False if context_scope == "none" else context_scope,
        "context_pad_random": context_pad_random,
        "context_wrong": context_wrong,
        "prompt_regime": prompt_regime,
        **({"context_window_lines": context_window_lines}
           if context_scope == "window" else {}),
        "tasks": tasks,
        "overall_word_acc": (
            sum(row["word_acc"] for rows in aggregate.values() for row in rows)
            / max(1, sum(len(rows) for rows in aggregate.values()))),
        "generation": {
            "n": len(completions),
            "mean_generated_tokens": sum(x["generated_tokens"] for x in completions) / len(completions),
            "mean_budget_tokens": sum(x["budget_tokens"] for x in completions) / len(completions),
            "stopped_fraction": sum(x["stopped"] for x in completions) / len(completions),
            "hard_cut_fraction": sum(x["hard_cut"] for x in completions) / len(completions),
            "n_bounded": len(bounded),
            "hard_cut_fraction_bounded": sum(x["hard_cut"] for x in bounded) / max(1, len(bounded)),
        },
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", default="",
                        help="published student checkpoint to evaluate; empty uses the base model")
    parser.add_argument("--n-per-task", type=int, default=24)
    parser.add_argument("--max-extra-tokens", type=int, default=32)
    parser.add_argument("--budget-multiplier", type=float, default=1.0)
    parser.add_argument("--context-scope", choices=("none", "full", "window", "chapter"), default="full")
    parser.add_argument("--context-window-lines", type=int, default=4)
    parser.add_argument("--context-pad-random", action="store_true")
    parser.add_argument("--context-wrong", action="store_true")
    parser.add_argument("--recall-corpora", nargs="+", default=["machado", "quijote_ch1", "quijote_ch4"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--use-cudagraphs", action="store_true")
    parser.add_argument(
        "--async-scheduling", action="store_true",
        help=("opt into vLLM asynchronous scheduling; synchronous is the "
              "evaluation default because vLLM 0.25 async+pipeline parallel "
              "can mis-account heterogeneous output placeholders"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config, args.experiment)
    model_source = args.checkpoint or cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    prompt_regime = "rag_system" if cfg.mask.mode == "rag_system" else "rag_tool"
    corpus_rows = {}
    all_prepared = []
    slices = {}
    for corpus in args.recall_corpora:
        start = len(all_prepared)
        rows = _prepare(
            tokenizer, CORPUS_PATHS[corpus], seed=17,
            n_per_task=args.n_per_task, max_extra_tokens=args.max_extra_tokens,
            budget_multiplier=args.budget_multiplier,
            context_scope=args.context_scope,
            context_window_lines=args.context_window_lines,
            context_pad_random=args.context_pad_random,
            context_wrong=args.context_wrong, prompt_regime=prompt_regime)
        all_prepared.extend(rows)
        slices[corpus] = slice(start, len(all_prepared))

    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    started = time.perf_counter()
    llm = LLM(
        model=model_source, dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        enforce_eager=not args.use_cudagraphs, disable_log_stats=True,
        async_scheduling=args.async_scheduling)
    load_seconds = time.perf_counter() - started
    generated_at = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": row["prompt_ids"]} for row in all_prepared],
        [SamplingParams(temperature=0.0, top_p=1.0,
                        max_tokens=row["budget"],
                        stop_token_ids=[row["stop_id"]], ignore_eos=False)
         for row in all_prepared], use_tqdm=False)
    answers, completions = [], []
    for row, result in zip(all_prepared, outputs):
        ids = list(result.outputs[0].token_ids)
        stopped = bool(ids and ids[-1] == row["stop_id"])
        answers.append(tokenizer.decode(ids, skip_special_tokens=True))
        completions.append({"generated_tokens": len(ids),
                            "budget_tokens": row["budget"],
                            "stopped": stopped,
                            "hard_cut": len(ids) >= row["budget"] and not stopped})
    generation_seconds = time.perf_counter() - generated_at

    for corpus, span in slices.items():
        result = _score(
            all_prepared[span], answers[span], completions[span], seed=17,
            n_per_task=args.n_per_task, context_scope=args.context_scope,
            context_window_lines=args.context_window_lines,
            context_pad_random=args.context_pad_random,
            context_wrong=args.context_wrong, prompt_regime=prompt_regime)
        result["poem_path"] = CORPUS_PATHS[corpus]
        corpus_rows[corpus] = result

    kind = ("teacher_epoch0_native_no_rag" if args.context_scope == "none"
            else f"teacher_epoch0_rag_{args.context_scope}")
    if args.context_pad_random:
        kind += "_padfloor"
    artifact = {
        "schema_version": 2,
        "teacher_reference_kind": kind,
        "generation_backend": "vllm",
        "generation_backend_version": __import__("vllm").__version__,
        "source_commit": source_commit,
        "context_scope": args.context_scope,
        "context_pad_random": args.context_pad_random,
        "context_wrong": args.context_wrong,
        "max_extra_tokens": args.max_extra_tokens,
        "budget_multiplier": args.budget_multiplier,
        "model": cfg.model.name,
        "model_source": model_source,
        "checkpoint": args.checkpoint or None,
        "corpora_measured": args.recall_corpora,
        "corpus_selection": "cli_override",
        "timings": {"load_seconds": load_seconds,
                    "generation_seconds": generation_seconds,
                    "total_seconds": time.perf_counter() - started},
        "parallelism": {"tensor_parallel_size": args.tensor_parallel_size,
                        "pipeline_parallel_size": args.pipeline_parallel_size,
                        "max_num_seqs": args.max_num_seqs,
                        "use_cudagraphs": args.use_cudagraphs,
                        "async_scheduling": args.async_scheduling},
        "corpora": corpus_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {out} ({len(all_prepared)} prompts, "
          f"load {load_seconds:.1f}s, generation {generation_seconds:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
