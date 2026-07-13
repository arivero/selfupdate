"""Greedy V5 teacher-generation benchmark with vLLM.

This deliberately measures only answer generation: no hidden-state request,
no cache write, and no teacher-forced second forward.  Prompts and per-record
generation ceilings mirror ``build_teacher_cache.py`` so output quality can be
compared with the cached PyTorch teacher generations.

Run under the dedicated CUDA-12.8 vLLM environment, for example:

  source ../venvs/vllm126/bin/activate
  PYTHONPATH=../2025/vllm python scripts/benchmark_vllm_generation.py \\
      --model Qwen/Qwen3-0.6B --batch-sizes 1 4 16 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selfupdate.chatfmt import adapt_records, stop_token_id  # noqa: E402
from selfupdate.config import load_config  # noqa: E402
from selfupdate.masking import DEFAULT_SYSTEM, ContextMasker, SegmentedExample  # noqa: E402

# Reuse the source-of-truth budget and corpus scoring logic, while keeping this
# script's import guard local to this checkout.
from build_teacher_cache import _corpus_texts, _generation_budget, _recitation_stats  # noqa: E402


class GpuMonitor:
    def __init__(self, device: int):
        self.device = device
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = subprocess.check_output([
                    "nvidia-smi", "--id", str(self.device),
                    "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                ], text=True).strip()
                self.peak_mib = max(self.peak_mib, int(text))
            except Exception:
                pass
            self._stop.wait(0.1)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=2)


def cached_report(model: str, examples: Path, expected_ids: set[str]) -> dict[str, dict]:
    """Best-effort PyTorch generation baseline keyed by V5 example id."""
    short = model.split("/")[-1]
    candidates = sorted((ROOT / "caches").glob(f"{short}-*/generation_report.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        report = json.loads(path.read_text(encoding="utf-8"))
        items = {x["example_id"]: x for x in report.get("items", [])}
        # Cache directory names are hashes and do not contain the RAG scope.
        # Exact example-id coverage is the only reliable identity check.
        if expected_ids.issubset(items):
            return {key: items[key] for key in expected_ids}
    return {}


def gpt_oss_prompt(record: dict, expected_chars: int, extra_tokens: int,
                   *, guided_memory: bool = False):
    """Render one V5 question/passage with GPT-OSS's native Harmony protocol.

    V5 ``rag_system`` records put the passage *inside a Qwen system turn* and
    cannot be transplanted textually.  Harmony's developer message is the
    corresponding high-priority place for the fixed recitation instruction and
    privileged memory; the question stays a user turn.  ``guided_memory``
    preserves the same memory-framed task while making the instruction to
    continue literally from that memory unambiguous for GPT-OSS.  It never
    describes a document, tool, retrieval, or external source.  The function
    returns token IDs, a per-record answer ceiling, the Harmony decoder, and
    EOS.
    """
    from openai_harmony import (Conversation, HarmonyEncodingName, Message,
                                ReasoningEffort, Role, SystemContent,
                                load_harmony_encoding)
    # Constructing this object is cheap, but hold the encoder across records.
    if not hasattr(gpt_oss_prompt, "encoding"):
        gpt_oss_prompt.encoding = load_harmony_encoding(  # type: ignore[attr-defined]
            HarmonyEncodingName.HARMONY_GPT_OSS)
    encoding = gpt_oss_prompt.encoding  # type: ignore[attr-defined]
    system = (SystemContent.new()
              .with_reasoning_effort(ReasoningEffort.LOW)
              .with_conversation_start_date("2026-07-13"))
    passage = record.get("privileged", "")
    if guided_memory:
        developer_text = (
            "Eres un experto en poesía española. Respondes recitando con "
            "exactitud literal. Recuerdas literalmente el texto que sigue y "
            "puedes hablar de él con todo detalle:\n"
            + passage
        )
    else:
        developer_text = DEFAULT_SYSTEM + passage
    conversation = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, system),
        Message.from_role_and_content(Role.DEVELOPER, developer_text),
        Message.from_role_and_content(Role.USER, record["question"]),
    ])
    ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    passage_ids = encoding.encode(passage, disallowed_special=())
    ratio = len(passage_ids) / max(len(passage), 1) if passage_ids else 0.35
    budget = 2 * max(4, math.ceil(expected_chars * ratio)) + extra_tokens
    # Harmony returns final text only; an analysis channel is deliberately not
    # scored as the recited answer.
    def final_text(token_ids: list[int]) -> tuple[str, str]:
        raw = encoding.decode(token_ids)
        try:
            messages = encoding.parse_messages_from_completion_tokens(
                token_ids, Role.ASSISTANT, strict=False)
            finals = [m for m in messages if m.channel == "final"]
            if finals:
                text = "\n".join(
                    c.text for m in finals for c in m.content
                    if getattr(c, "text", None) is not None)
                return text, raw
        except Exception:
            pass
        return raw, raw
    return ids, budget, final_text, 200002  # <|return|>, GPT-OSS EOS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default="configs/experiments/v5rs/cache_0p6b_window_remove.yaml")
    ap.add_argument("--examples", default="data/combined/examples_v5rs_window.jsonl")
    # Large batches give the useful throughput result first; B=1 remains in
    # the sweep, but should not delay every other measurement.
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 16, 4, 1])
    ap.add_argument("--limit", type=int, default=0,
                    help="deterministic evenly-spaced prompt subsample; 0 = all")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--pipeline-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="vLLM concurrent-sequence ceiling; set explicitly for capacity sweeps")
    ap.add_argument("--max-num-batched-tokens", type=int, default=None,
                    help="vLLM scheduler token ceiling; set with --max-num-seqs for a real prefill-capacity probe")
    ap.add_argument("--prompt-format", choices=("native", "gpt_oss_harmony",
                                                   "gpt_oss_guided_memory"),
                    default="native",
                        help="native template, GPT-OSS memory framing, or guided GPT-OSS memory framing")
    ap.add_argument("--generation-extra-tokens", type=int, default=None,
                    help="override config's conversational generation margin")
    ap.add_argument("--use-cudagraphs", action="store_true",
                    help="performance mode: permit vLLM compilation/CUDA graphs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    cfg = load_config(args.config, args.experiment)
    examples_path = ROOT / args.examples
    records = [json.loads(x) for x in examples_path.read_text(encoding="utf-8").splitlines()]
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.prompt_format == "native":
        records = adapt_records(records, tok)
    if args.limit and args.limit < len(records):
        # Retain all task/corpus regions rather than using the first records,
        # which are ordered by corpus and would make a speed/quality sample
        # misleading.
        step = len(records) / args.limit
        records = [records[min(int(i * step), len(records) - 1)]
                   for i in range(args.limit)]
    extra_tokens = (cfg.cache.generation_extra_tokens if args.generation_extra_tokens is None
                    else args.generation_extra_tokens)
    examples = [SegmentedExample.from_record(x) for x in records]
    masker = ContextMasker(tok, pad_random=(cfg.mask.compaction == "pad_random"))
    stop_id = stop_token_id(tok)
    corpus = _corpus_texts(examples_path)
    prompts = []
    for record, ex in zip(records, examples):
        if args.prompt_format.startswith("gpt_oss_"):
            ids, budget, decoder, prompt_stop_id = gpt_oss_prompt(
                record, int(record.get("expected_answer_chars", 64)), extra_tokens,
                guided_memory=(args.prompt_format == "gpt_oss_guided_memory"))
            prompts.append({"example_id": ex.example_id, "record": record, "ids": ids,
                            "budget": budget, "decoder": decoder, "stop_id": prompt_stop_id})
        else:
            budget = _generation_budget(masker, ex, int(record.get("expected_answer_chars", 64)),
                                        extra_tokens)
            prompts.append({"example_id": ex.example_id, "record": record,
                            "ids": masker.build(ex).teacher_ids, "budget": budget,
                            "stop_id": stop_id})

    out_dir = ROOT / (args.out or f"runs/vllm_benchmark/{args.model.split('/')[-1]}_tp{args.tensor_parallel_size}_pp{args.pipeline_parallel_size}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # vLLM workers see their local device as cuda:0 after CUDA_VISIBLE_DEVICES
    # masking, while nvidia-smi addresses physical GPUs.  Record the physical
    # first visible id so each concurrent benchmark reports its own peak.
    device = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    baseline = cached_report(args.model, examples_path,
                             {x["example_id"] for x in prompts})
    with GpuMonitor(device) as monitor:
        t_load = time.perf_counter()
        llm_kw = dict(model=args.model, dtype="bfloat16",
                      enforce_eager=not args.use_cudagraphs,
                      tensor_parallel_size=args.tensor_parallel_size,
                      pipeline_parallel_size=args.pipeline_parallel_size,
                      gpu_memory_utilization=args.gpu_memory_utilization,
                      max_model_len=args.max_model_len, disable_log_stats=True)
        if args.max_num_seqs is not None:
            llm_kw["max_num_seqs"] = args.max_num_seqs
        if args.max_num_batched_tokens is not None:
            llm_kw["max_num_batched_tokens"] = args.max_num_batched_tokens
        llm = LLM(**llm_kw)
        load_seconds = time.perf_counter() - t_load

        rows = []
        for bs in args.batch_sizes:
            t0 = time.perf_counter()
            outputs = []
            for start in range(0, len(prompts), bs):
                chunk = prompts[start:start + bs]
                # vLLM SamplingParams has a batch-wide ceiling. Grouping by
                # actual per-record ceilings keeps the precise V5 protocol.
                by_budget: dict[int, list[dict]] = {}
                for item in chunk:
                    by_budget.setdefault(item["budget"], []).append(item)
                for budget, group in by_budget.items():
                    generated = llm.generate(
                        [{"prompt_token_ids": x["ids"]} for x in group],
                        SamplingParams(temperature=0.0, top_p=1.0, max_tokens=budget,
                                       stop_token_ids=[group[0]["stop_id"]], ignore_eos=False),
                        use_tqdm=False,
                    )
                    for item, result in zip(group, generated):
                        token_ids = result.outputs[0].token_ids
                        item_stop_id = item["stop_id"]
                        hard_cut = not token_ids or token_ids[-1] != item_stop_id
                        if hard_cut:
                            token_ids = token_ids + [item_stop_id]
                        if args.prompt_format.startswith("gpt_oss_"):
                            text, raw_text = item["decoder"](token_ids[:-1])
                        else:
                            text, raw_text = tok.decode(token_ids[:-1]), None
                        stats = _recitation_stats(item["record"], text, corpus)
                        old = baseline.get(item["example_id"], {})
                        outputs.append({"batch_size": bs, "example_id": item["example_id"],
                                        "gen_tokens": len(token_ids), "hard_cut": hard_cut,
                                        "answer_text": text, "raw_answer_text": raw_text, **stats,
                                        "pytorch_word_acc": old.get("word_acc"),
                                        "matches_cached_text": text == old.get("answer_text") if old else None})
            elapsed = time.perf_counter() - t0
            batch_outputs = [x for x in outputs if x["batch_size"] == bs]
            tokens = sum(x["gen_tokens"] for x in batch_outputs)
            word_scores = [x["word_acc"] for x in batch_outputs if "word_acc" in x]
            containment_scores = [x["containment"] for x in batch_outputs
                                  if "containment" in x]
            # next/prev use reference-word LCS; cloze deliberately has no
            # stored deleted-word reference and uses target-block containment.
            # Treating missing word_acc as zero silently scored every cloze
            # example as a failure (249/2071 in v5rs), depressing the Qwen3-14B
            # aggregate by 11.26 percentage points.
            task_scores = [x.get("word_acc", x.get("containment", 0.0))
                           for x in batch_outputs]
            rows.append({"batch_size": bs, "examples": len(prompts), "seconds": elapsed,
                         "generated_tokens": tokens, "tokens_per_second": tokens / elapsed,
                         "peak_gpu_mib": monitor.peak_mib,
                         "mean_gen_tokens": tokens / max(len(prompts), 1),
                         "hard_cut_fraction": sum(x["hard_cut"] for x in batch_outputs) / max(len(prompts), 1),
                         "mean_task_score": sum(task_scores) / max(len(task_scores), 1),
                         "mean_word_acc": sum(word_scores) / max(len(word_scores), 1),
                         "word_acc_examples": len(word_scores),
                         "mean_containment": sum(containment_scores) / max(len(containment_scores), 1),
                         "containment_examples": len(containment_scores),
                         "cached_text_match_fraction": sum(x.get("matches_cached_text") is True for x in batch_outputs) / max(sum(x.get("matches_cached_text") is not None for x in batch_outputs), 1)})
            (out_dir / f"responses_bs{bs}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in outputs))
            print(json.dumps(rows[-1]), flush=True)
    summary = {"model": args.model, "vllm": __import__("vllm").__version__, "torch": torch.__version__,
               "cuda": torch.version.cuda, "load_seconds": load_seconds, "prompt_count": len(prompts),
               "tensor_parallel_size": args.tensor_parallel_size, "pipeline_parallel_size": args.pipeline_parallel_size,
               "use_cudagraphs": args.use_cudagraphs,
               "max_num_seqs": args.max_num_seqs,
               "max_num_batched_tokens": args.max_num_batched_tokens,
               "prompt_format": args.prompt_format,
               "limit": args.limit,
               "generation_extra_tokens": extra_tokens, "results": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
