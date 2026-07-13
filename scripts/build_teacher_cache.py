"""Precompute the frozen-teacher hidden-state cache for every example.

One configured-dtype forward per example stores per-layer hidden states at the
aligned span. RAG authority is certified separately by
``scripts/rag_generation_gate.py``; cache construction never runs the teacher
on a student/censored prompt.

v5 open-answer records (empty ``answer``; see data/questions.py) add a
GENERATION step before the forward: the teacher, holding the master-RAG
tool turn, greedily generates its answer (stop at the turn closer, hard cut
at 2x the record's expected length); the generated ids become the aligned
span for the teacher-forced forward and are stored in the cache index —
answers are cache content (per-model), never dataset content. A per-item
recitation report (generation vs the targeted corpus span) lands in
generation_report.json.

Usage:
    python scripts/build_teacher_cache.py [--config configs/base.yaml] [--experiment ...]
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.utils.env import cap_cpu_threads  # noqa: E402

cap_cpu_threads()

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig
from transformers.utils import logging as transformers_logging

from selfupdate.chatfmt import adapt_records, stop_token_id
from selfupdate.config import load_config
from selfupdate.masking import ContextMasker, SegmentedExample
from selfupdate.teacher.cache import AsyncTeacherCacheWriter, resolve_cache_dir


def load_records(path: str, tokenizer) -> list[dict]:
    records = [json.loads(line)
               for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return adapt_records(records, tokenizer)


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


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError)
        and "out of memory" in str(exc).lower()
    )


@torch.inference_mode()
def generate_answers_batched(
    model,
    prompts: list[list[int]],
    budgets: list[int],
    stop_id: int,
    batch_size: int,
    max_sequence_tokens: int = 0,
    budget_bucket: int = 1,
    compile_generation: bool = False,
    cache_implementation: str = "",
    shuffle_seed: int = 0,
    compile_dynamic: bool = True,
    cache_max_tokens: int = 0,
    fixed_batch: bool = False,
    progress_callback=None,
) -> tuple[list[tuple[list[int], bool]], list[int]]:
    """Left-padded batched greedy decode with per-record answer ceilings.

    ``transformers.generate`` accepts one ``max_new_tokens`` per batch, so a
    row is truncated back to its own budget after decode.  Tokens after that
    bound cannot affect its earlier greedy choices.  A CUDA OOM retries the
    same records at half the batch size and records the effective sizes.
    """
    if batch_size < 1:
        raise ValueError("cache.generation_batch must be positive")
    if budget_bucket < 0:
        raise ValueError("cache.generation_budget_bucket must be non-negative")
    if compile_generation and not cache_implementation:
        raise ValueError(
            "compiled generation requires cache.generation_cache_implementation")
    if len(prompts) != len(budgets):
        raise ValueError("prompt/budget length mismatch")
    for i, (prompt, budget) in enumerate(zip(prompts, budgets)):
        if budget < 1:
            raise ValueError(f"record {i} has non-positive generation budget")
        if max_sequence_tokens and len(prompt) + budget > max_sequence_tokens:
            raise ValueError(
                f"record {i} needs {len(prompt)} prompt + {budget} answer = "
                f"{len(prompt) + budget} tokens, above cache.max_sequence_tokens="
                f"{max_sequence_tokens}")

    answers: list[tuple[list[int], bool] | None] = [None] * len(prompts)
    effective_batches: list[int] = []
    safe_batch_size = batch_size
    completed_rows = 0
    generated_token_count = 0
    order = list(range(len(prompts)))
    outer_batches: list[list[int]] = []
    if shuffle_seed:
        # Keep similarly sized decode jobs together so rounded allowance
        # groups actually reach the requested batch size.  Randomize both
        # members and batch order deterministically: this retains stochastic
        # scheduling without mixing a 100-token answer with an 800-token one.
        rng = random.Random(shuffle_seed)
        rng.shuffle(order)
        aligned: dict[int, list[int]] = {}
        for index in order:
            key = (budgets[index] if budget_bucket == 1 else 0
                   if budget_bucket == 0 else
                   math.ceil(budgets[index] / budget_bucket) * budget_bucket)
            aligned.setdefault(key, []).append(index)
        for group in aligned.values():
            outer_batches.extend(
                group[start:start + batch_size]
                for start in range(0, len(group), batch_size))
        rng.shuffle(outer_batches)
    else:
        outer_batches = [
            order[start:start + batch_size]
            for start in range(0, len(order), batch_size)]
    # With no scheduling seed this remains the benchmark-compatible dataset
    # order.  A positive seed opts into the aligned randomized schedule above.
    for outer_indices in outer_batches:
        by_budget: dict[int, list[int]] = {}
        outer_budgets = [budgets[index] for index in outer_indices]
        for index in outer_indices:
            budget = budgets[index]
            group_budget = (
                budget if budget_bucket == 1 else
                max(outer_budgets) if budget_bucket == 0 else
                math.ceil(budget / budget_bucket) * budget_bucket)
            by_budget.setdefault(group_budget, []).append(index)
        for group_budget, group in by_budget.items():
            offset = 0
            current = min(len(group), safe_batch_size)
            while offset < len(group):
                indices = group[offset:offset + current]
                physical_indices = list(indices)
                if fixed_batch and physical_indices:
                    physical_indices.extend(
                        [physical_indices[-1]]
                        * (safe_batch_size - len(physical_indices)))
                chunk_prompts = [prompts[index] for index in physical_indices]
                width = max(map(len, chunk_prompts))
                if (max_sequence_tokens
                        and width + group_budget > max_sequence_tokens):
                    raise ValueError(
                        f"padded generation needs {width + group_budget} tokens, "
                        f"above cache.max_sequence_tokens={max_sequence_tokens}")
                if cache_max_tokens and width + group_budget > cache_max_tokens:
                    raise ValueError(
                        f"padded generation needs {width + group_budget} tokens, "
                        f"above cache.generation_cache_max_tokens={cache_max_tokens}")
                input_ids = torch.full(
                    (len(physical_indices), width), stop_id, dtype=torch.long,
                    device=model.device)
                attention_mask = torch.zeros_like(input_ids)
                for row, prompt in enumerate(chunk_prompts):
                    input_ids[row, -len(prompt):] = torch.tensor(
                        prompt, dtype=torch.long, device=model.device)
                    attention_mask[row, -len(prompt):] = 1
                try:
                    generation_options = {}
                    if cache_implementation:
                        generation_options["cache_implementation"] = cache_implementation
                    if cache_max_tokens:
                        generation_options["cache_config"] = {
                            "max_cache_len": cache_max_tokens}
                    if compile_generation:
                        generation_options["compile_config"] = CompileConfig(
                            mode="reduce-overhead", dynamic=compile_dynamic)
                    out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=group_budget,
                        do_sample=False,
                        eos_token_id=stop_id,
                        pad_token_id=stop_id,
                        use_cache=True,
                        **generation_options,
                    )
                except RuntimeError as exc:
                    cannot_reduce = (safe_batch_size <= 1 if fixed_batch
                                     else len(indices) <= 1)
                    if not _is_cuda_oom(exc) or cannot_reduce:
                        raise
                    del input_ids, attention_mask
                    current = max(
                        1, (safe_batch_size if fixed_batch else len(indices)) // 2)
                    safe_batch_size = min(safe_batch_size, current)
                    current = min(len(group) - offset, safe_batch_size)
                    torch.cuda.empty_cache()
                    print(
                        f"generation OOM: retrying allowance {group_budget} at "
                        f"batch {current}", flush=True)
                    continue
                generated = out[:len(indices), width:].to("cpu")
                for row, index in enumerate(indices):
                    budget = budgets[index]
                    ids = generated[row, :budget].tolist()
                    if stop_id in ids:
                        ids = ids[:ids.index(stop_id) + 1]
                        hard_cut = False
                    else:
                        ids.append(stop_id)
                        hard_cut = True
                    answers[index] = (ids, hard_cut)
                effective_batches.append(len(indices))
                completed_rows += len(indices)
                generated_token_count += sum(
                    len(answers[index][0]) for index in indices)
                if progress_callback is not None:
                    progress_callback(
                        completed_rows, len(prompts), generated_token_count,
                        len(indices), group_budget)
                offset += len(indices)
                current = min(len(group) - offset, safe_batch_size)
                del input_ids, attention_mask, out, generated
    if any(answer is None for answer in answers):
        raise RuntimeError("batched generation left an unanswered record")
    return [answer for answer in answers if answer is not None], effective_batches


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
    # Weight-loading bars rewrite one terminal line hundreds of times and make
    # detached logs/transcripts enormous.  Cache progress is reported by the
    # outer teacher-forward bar and final structured timings instead.
    transformers_logging.disable_progress_bar()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--generation-batch", type=int, default=None,
                    help="override cache.generation_batch")
    ap.add_argument("--max-sequence-tokens", type=int, default=None,
                    help="override cache.max_sequence_tokens; e.g. 8192")
    ap.add_argument("--generation-budget-bucket", type=int, default=None,
                    help="1=exact allowances, N=round up to N, 0=one outer group")
    ap.add_argument("--generation-compile", action="store_true",
                    help="compile decode with PyTorch reduce-overhead/CUDA graphs")
    ap.add_argument("--generation-cache-implementation", default=None,
                    help="Transformers cache implementation, e.g. hybrid or static")
    ap.add_argument("--generation-static-shapes", action="store_true",
                    help="compile decode with dynamic=False")
    ap.add_argument("--generation-cache-max-tokens", type=int, default=None,
                    help="pin Transformers cache max length for graph reuse")
    ap.add_argument("--generation-fixed-batch", action="store_true",
                    help="duplicate padding rows to keep the physical batch fixed")
    ap.add_argument("--generation-shuffle-seed", type=int, default=None,
                    help="deterministically shuffle generation scheduling")
    ap.add_argument("--cache-root", default=None,
                    help="override cache.root (use node-local /tmp for the hot path)")
    ap.add_argument("--limit", type=int, default=None,
                    help="evenly-spaced cache subset for a performance probe")
    ap.add_argument("--generation-only", action="store_true",
                    help="benchmark/write teacher answers without hidden-state caching")
    ap.add_argument("--generation-responses", default=None,
                    help="reuse exact token_ids from a completed response JSONL")
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)
    if args.generation_batch is not None:
        cfg.cache.generation_batch = args.generation_batch
    if args.max_sequence_tokens is not None:
        cfg.cache.max_sequence_tokens = args.max_sequence_tokens
    if args.generation_budget_bucket is not None:
        cfg.cache.generation_budget_bucket = args.generation_budget_bucket
    if args.generation_compile:
        cfg.cache.generation_compile = True
    if args.generation_cache_implementation is not None:
        cfg.cache.generation_cache_implementation = args.generation_cache_implementation
    if args.generation_static_shapes:
        cfg.cache.generation_compile_dynamic = False
    if args.generation_cache_max_tokens is not None:
        cfg.cache.generation_cache_max_tokens = args.generation_cache_max_tokens
    if args.generation_fixed_batch:
        cfg.cache.generation_fixed_batch = True
    if args.generation_shuffle_seed is not None:
        cfg.cache.generation_shuffle_seed = args.generation_shuffle_seed
    if args.cache_root is not None:
        cfg.cache.root = args.cache_root
    if args.limit is not None:
        cfg.cache.limit = args.limit
    if args.generation_responses is not None:
        cfg.cache.generation_responses_path = args.generation_responses

    root, chash = resolve_cache_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    print(f"cache dir: {root}")

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    records = load_records(cfg.data.examples_path, tok)
    if cfg.cache.limit < 0:
        raise ValueError("cache.limit must be non-negative")
    if cfg.cache.limit and cfg.cache.limit < len(records):
        step = len(records) / cfg.cache.limit
        records = [records[min(int(i * step), len(records) - 1)]
                   for i in range(cfg.cache.limit)]
    examples = [SegmentedExample.from_record(r) for r in records]
    open_answer = [not ex.answer for ex in examples]
    if any(open_answer) and not all(open_answer):
        sys.exit("mixed open-answer/legacy records in one jsonl — rebuild")
    v5 = all(open_answer) and bool(examples)
    if args.generation_only and not v5:
        sys.exit("--generation-only requires an open-answer V5 dataset")
    # Import-only scoring needs the tokenizer and exact response IDs, but no
    # teacher weights.  Avoid a many-GiB model load when validating a completed
    # generation artifact before the independent hidden-state cache pass.
    import_only = bool(args.generation_only
                       and cfg.cache.generation_responses_path)
    model = None
    decoder = None
    n_layers = 0
    if not import_only:
        try:
            model_dtype = getattr(torch, cfg.model.dtype)
        except AttributeError as exc:
            raise ValueError(f"unknown model.dtype {cfg.model.dtype!r}") from exc
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, dtype=model_dtype, low_cpu_mem_usage=True)
        model.to(cfg.model.device)
        model.eval()
        # Text-only cache targets come from the decoder body.  Multimodal Gemma
        # 4 wraps that body under model.language_model; ordinary causal LMs
        # expose it directly as model.  Calling the body avoids materializing a
        # full [sequence, vocabulary] logits tensor.
        decoder = getattr(model.model, "language_model", model.model)
        n_layers = decoder.config.num_hidden_layers

    def sync() -> None:
        # Phase timings must not assign queued CUDA work to the next CPU
        # section. The cache loop is dependency-serial already (generation →
        # forward → D2H write → next forward), so these synchronization
        # points measure existing waits rather than creating a new overlap.
        if model is not None and torch.cuda.is_available():
            torch.cuda.synchronize(model.device)

    masker = ContextMasker(tok, pad_random=(cfg.mask.compaction == "pad_random"))
    stop_id = stop_token_id(tok)
    corpora = _corpus_texts(Path(cfg.data.examples_path)) if v5 else {}

    gen_report = []
    gen_summary = None
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    timings = {"generation_setup_seconds": 0.0,
               "generation_seconds": 0.0, "teacher_forward_seconds": 0.0,
               "cache_write_seconds": 0.0,
               "finalize_seconds": 0.0}
    hidden_bytes = 0
    teacher_compute_events = []
    started_at = time.perf_counter()
    generated_answers: list[tuple[list[int], bool]] = []
    effective_generation_batches: list[int] = []
    if v5:
        prompts = [masker.build(ex).teacher_ids for ex in examples]
        budgets = [
            _generation_budget(masker, ex,
                               int(record.get("expected_answer_chars", 64)),
                               cfg.cache.generation_extra_tokens)
            for record, ex in zip(records, examples)
        ]
        timings["maximum_prompt_tokens"] = max(map(len, prompts), default=0)
        timings["maximum_generation_budget"] = max(budgets, default=0)
        timings["maximum_individual_sequence_tokens"] = max(
            (len(prompt) + budget for prompt, budget in zip(prompts, budgets)),
            default=0)
        if cfg.cache.generation_responses_path:
            t_import = time.perf_counter()
            response_rows = [json.loads(line) for line in Path(
                cfg.cache.generation_responses_path).read_text(
                    encoding="utf-8").splitlines()]
            response_by_id = {row["example_id"]: row for row in response_rows}
            if len(response_by_id) != len(response_rows):
                raise ValueError("duplicate example_id in generation responses")
            missing = [ex.example_id for ex in examples
                       if ex.example_id not in response_by_id]
            if missing:
                raise ValueError(
                    f"generation responses miss {len(missing)} examples; "
                    f"first is {missing[0]}")
            for ex, budget in zip(examples, budgets):
                row = response_by_id[ex.example_id]
                ids = row.get("token_ids")
                if not isinstance(ids, list) or not ids:
                    raise ValueError(
                        f"generation response {ex.example_id} has no token_ids")
                ids = [int(token_id) for token_id in ids]
                if ids[-1] != stop_id:
                    raise ValueError(
                        f"generation response {ex.example_id} lacks stop sentinel")
                if len(ids) > budget + 1:
                    raise ValueError(
                        f"generation response {ex.example_id} exceeds its budget")
                generated_answers.append((ids, bool(row["hard_cut"])))
            timings["generation_import_seconds"] = (
                time.perf_counter() - t_import)
            timings["generation_responses_path"] = (
                cfg.cache.generation_responses_path)
        if (not generated_answers
                and cfg.cache.generation_compile and prompts):
            order = list(range(len(prompts)))
            if cfg.cache.generation_fixed_batch:
                warm_groups: dict[int, list[int]] = {}
                for index in order:
                    budget = budgets[index]
                    key = (budget if cfg.cache.generation_budget_bucket == 1
                           else 0 if cfg.cache.generation_budget_bucket == 0
                           else math.ceil(
                               budget / cfg.cache.generation_budget_bucket)
                           * cfg.cache.generation_budget_bucket)
                    warm_groups.setdefault(key, []).append(index)
                order = max(warm_groups.values(), key=len)
            elif cfg.cache.generation_shuffle_seed:
                generator = torch.Generator().manual_seed(
                    cfg.cache.generation_shuffle_seed)
                order = torch.randperm(
                    len(order), generator=generator).tolist()
            warm_indices = order[:cfg.cache.generation_batch]
            t_setup = time.perf_counter()
            generate_answers_batched(
                model,
                [prompts[index] for index in warm_indices],
                [budgets[index] for index in warm_indices],
                stop_id, cfg.cache.generation_batch,
                cfg.cache.max_sequence_tokens,
                cfg.cache.generation_budget_bucket,
                True, cfg.cache.generation_cache_implementation, 0,
                cfg.cache.generation_compile_dynamic,
                cfg.cache.generation_cache_max_tokens,
                cfg.cache.generation_fixed_batch)
            sync()
            timings["generation_setup_seconds"] = (
                time.perf_counter() - t_setup)
        if not generated_answers:
            progress_path = root / "generation_progress.jsonl"
            progress_path.write_text("")
            t_phase = time.perf_counter()

            def record_generation_progress(completed, total, tokens,
                                           effective_batch, allowance):
                elapsed = time.perf_counter() - t_phase
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "completed": completed,
                        "total": total,
                        "generated_tokens": tokens,
                        "elapsed_seconds": elapsed,
                        "tokens_per_second": tokens / elapsed if elapsed else 0.0,
                        "effective_batch": effective_batch,
                        "allowance": allowance,
                    }) + "\n")

            generated_answers, effective_generation_batches = generate_answers_batched(
                model, prompts, budgets, stop_id, cfg.cache.generation_batch,
                cfg.cache.max_sequence_tokens, cfg.cache.generation_budget_bucket,
                cfg.cache.generation_compile,
                cfg.cache.generation_cache_implementation,
                cfg.cache.generation_shuffle_seed,
                cfg.cache.generation_compile_dynamic,
                cfg.cache.generation_cache_max_tokens,
                cfg.cache.generation_fixed_batch,
                record_generation_progress)
            sync()
            timings["generation_seconds"] = time.perf_counter() - t_phase
        for record, ex, (answer_ids, hard_cut) in zip(
                records, examples, generated_answers):
            answer_text = tok.decode(answer_ids[:-1])
            gen_report.append({
                "example_id": ex.example_id,
                "kind": record.get("kind"),
                "corpus": record.get("corpus"),
                "gen_tokens": len(answer_ids),
                "hard_cut": hard_cut,
                "answer_text": answer_text,
                **_recitation_stats(record, answer_text, corpora),
            })
        accs = [g["word_acc"] for g in gen_report if "word_acc" in g]
        cuts = [g for g in gen_report if g["hard_cut"]]
        gen_summary = {
            "n": len(gen_report),
            "mean_word_acc_nextprev": mean(accs),
            "hard_cut_fraction": len(cuts) / max(len(gen_report), 1),
            "mean_gen_tokens": mean([g["gen_tokens"] for g in gen_report]),
        }
        # Generation is independently useful and expensive.  Persist it before
        # the hidden-state walk so a later D2H/storage failure remains
        # diagnosable and resumable rather than erasing the measurement.
        (root / "generation_report.json").write_text(json.dumps(
            {"summary": gen_summary, "items": gen_report},
            ensure_ascii=False))
        (root / "generation_timings.json").write_text(json.dumps({
            key: value for key, value in timings.items()
            if key.startswith("generation_") or key.startswith("maximum_")
        }, indent=2) + "\n")
        if cfg.cache.generation_compile and model is not None:
            # Transformers retains the torch.compile callable on the model.
            # Its CUDA-graph private pool is sized by the largest decode batch
            # and can otherwise leave too little room for output_hidden_states.
            # Generation is complete, so discard that specialization before
            # entering the independent teacher-forced cache phase.
            for attr in ("_compiled_call", "_last_compile_config"):
                if hasattr(model, attr):
                    delattr(model, attr)
            torch.compiler.reset()
        del prompts
        torch.cuda.empty_cache()

    timings["examples"] = len(examples)
    timings["generated_tokens"] = sum(
        len(answer_ids) for answer_ids, _ in generated_answers)
    timings["generation_tokens_per_second"] = (
        timings["generated_tokens"] / timings["generation_seconds"]
        if timings["generation_seconds"] else 0.0)
    timings["requested_generation_batch"] = cfg.cache.generation_batch
    timings["generation_budget_bucket"] = cfg.cache.generation_budget_bucket
    timings["generation_compile"] = cfg.cache.generation_compile
    timings["generation_cache_implementation"] = (
        cfg.cache.generation_cache_implementation)
    timings["generation_compile_dynamic"] = (
        cfg.cache.generation_compile_dynamic)
    timings["generation_cache_max_tokens"] = (
        cfg.cache.generation_cache_max_tokens)
    timings["generation_fixed_batch"] = cfg.cache.generation_fixed_batch
    timings["generation_shuffle_seed"] = cfg.cache.generation_shuffle_seed
    timings["minimum_effective_generation_batch"] = (
        min(effective_generation_batches) if effective_generation_batches else 0)
    timings["maximum_effective_generation_batch"] = (
        max(effective_generation_batches) if effective_generation_batches else 0)

    if args.generation_only:
        timings["total_seconds"] = time.perf_counter() - started_at
        (root / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
        print(f"generated {len(examples)} answers to {root}")
        print(f"teacher generation — {timings['generation_seconds']:.1f}s steady, "
              f"{timings['generation_setup_seconds']:.1f}s setup, "
              f"next/prev word-LCS {gen_summary['mean_word_acc_nextprev']:.3f}, "
              f"hard-cut {gen_summary['hard_cut_fraction']:.1%}")
        return

    writer = AsyncTeacherCacheWriter(
        root, chash, shard_size=cfg.cache.shard_size,
        hidden_dtype=cfg.cache.hidden_dtype)
    copy_stream = (torch.cuda.Stream(device=model.device)
                   if torch.cuda.is_available() else None)
    teacher_progress_path = root / "teacher_progress.jsonl"
    teacher_progress_path.write_text("")
    teacher_wall_started = time.perf_counter()

    for item_no, (record, ex) in enumerate(tqdm(
            zip(records, examples), total=len(examples), desc="teacher forward")):
        extra = None
        if v5:
            answer_ids, hard_cut = generated_answers[item_no]
            pair = masker.build(ex, answer_ids=answer_ids)
            extra = {"answer_ids": answer_ids, "hard_cut": hard_cut}
        else:
            pair = masker.build(ex)
        t_ids = torch.tensor([pair.teacher_ids], device=model.device)
        if copy_stream is not None:
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            compute_start.record(torch.cuda.current_stream(model.device))
        else:
            t_phase = time.perf_counter()
        with torch.no_grad():
            out = decoder(t_ids, output_hidden_states=True, use_cache=False)
        if copy_stream is not None:
            compute_end.record(torch.cuda.current_stream(model.device))
            teacher_compute_events.append((compute_start, compute_end))
        else:
            timings["teacher_forward_seconds"] += time.perf_counter() - t_phase
        span = pair.t_aligned
        # One packed, asynchronous D2H transfer replaces one implicit CUDA
        # synchronization per layer.  A dedicated copy stream and background
        # writer overlap transfer, finite checks, and /tmp shard writes with
        # the next teacher forward while preserving the per-layer file layout.
        t_phase = time.perf_counter()
        packed_gpu = torch.stack([
            out.hidden_states[L][0, span.start:span.stop]
            for L in range(1, n_layers + 1)
        ]).to(writer.hidden_dtype)
        if copy_stream is not None:
            packed_hidden = torch.empty_like(
                packed_gpu, device="cpu", pin_memory=True)
            copy_stream.wait_stream(torch.cuda.current_stream(model.device))
            with torch.cuda.stream(copy_stream):
                copy_start_event = torch.cuda.Event(enable_timing=True)
                copy_start_event.record(copy_stream)
                packed_hidden.copy_(packed_gpu, non_blocking=True)
                packed_gpu.record_stream(copy_stream)
                ready_event = torch.cuda.Event(enable_timing=True)
                ready_event.record(copy_stream)
        else:
            packed_hidden = packed_gpu.cpu()
            copy_start_event = None
            ready_event = None
        hidden_bytes += packed_hidden.numel() * packed_hidden.element_size()
        hidden = {L: packed_hidden[L - 1] for L in range(1, n_layers + 1)}
        writer.add(
            ex.example_id, hidden,
            span={
                "t0": pair.t_aligned.start, "s0": pair.s_aligned.start,
                "A": pair.aligned_len, "mid_len": pair.s_answer.start - pair.s_aligned.start,
                "position_gap": pair.position_gap,
                "n_teacher": len(pair.teacher_ids), "n_student": len(pair.student_ids),
            },
            extra=extra,
            copy_start_event=copy_start_event,
            ready_event=ready_event,
        )
        timings["cache_write_seconds"] += time.perf_counter() - t_phase
        completed = item_no + 1
        if completed % 100 == 0 or completed == len(examples):
            # This is deliberately wall/queue telemetry only: no .item(), CPU
            # tensor copy, CUDA event wait, or stream synchronization in the
            # teacher walk.  Exact compute/D2H/storage timings are finalized
            # from their events after the asynchronous writer drains.
            elapsed = time.perf_counter() - teacher_wall_started
            with teacher_progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "completed": completed,
                    "total": len(examples),
                    "elapsed_seconds": elapsed,
                    "examples_per_second": completed / elapsed,
                    "hidden_bytes_queued": hidden_bytes,
                }) + "\n")

    t_phase = time.perf_counter()
    writer.finalize()
    sync()
    timings["finalize_seconds"] = time.perf_counter() - t_phase
    if teacher_compute_events:
        timings["teacher_forward_seconds"] = sum(
            start.elapsed_time(end) for start, end in teacher_compute_events
        ) / 1000.0
    timings["d2h_copy_seconds"] = writer.copy_seconds
    timings["storage_seconds"] = writer.storage_seconds
    timings["hidden_bytes"] = hidden_bytes
    timings["d2h_gib_per_second"] = (
        hidden_bytes / (1024 ** 3) / writer.copy_seconds
        if writer.copy_seconds else 0.0)
    timings["total_seconds"] = time.perf_counter() - started_at
    timings["d2h_over_teacher_compute"] = (
        writer.copy_seconds / timings["teacher_forward_seconds"]
        if timings["teacher_forward_seconds"] else 0.0)
    timings["storage_over_teacher_compute"] = (
        writer.storage_seconds / timings["teacher_forward_seconds"]
        if timings["teacher_forward_seconds"] else 0.0)
    timings["examples"] = len(examples)
    timings["requested_generation_batch"] = cfg.cache.generation_batch
    timings["generation_budget_bucket"] = cfg.cache.generation_budget_bucket
    timings["generation_compile"] = cfg.cache.generation_compile
    timings["generation_cache_implementation"] = (
        cfg.cache.generation_cache_implementation)
    timings["generation_shuffle_seed"] = cfg.cache.generation_shuffle_seed
    timings["minimum_effective_generation_batch"] = (
        min(effective_generation_batches) if effective_generation_batches else 0)
    timings["maximum_effective_generation_batch"] = (
        max(effective_generation_batches) if effective_generation_batches else 0)
    timings["max_sequence_tokens"] = cfg.cache.max_sequence_tokens
    timings["seconds_per_example"] = (
        timings["total_seconds"] / max(len(examples), 1))
    print(f"wrote {len(examples)} examples, {n_layers} layers each, to {root}")
    (root / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    print("cache timing — " + ", ".join(
        f"{k} {v:.1f}s" for k, v in timings.items()
        if k.endswith("_seconds")))
    if v5:
        print(f"teacher recitation — next/prev word-LCS {gen_summary['mean_word_acc_nextprev']:.3f}, "
              f"hard-cut {gen_summary['hard_cut_fraction']:.1%}, "
              f"mean gen len {gen_summary['mean_gen_tokens']:.0f} tokens")


if __name__ == "__main__":
    main()
