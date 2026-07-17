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
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.utils.env import cap_cpu_threads  # noqa: E402

cap_cpu_threads()

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, CompileConfig, Mxfp4Config
from transformers.utils import logging as transformers_logging
from transformers.utils import is_kernels_available, is_triton_available

from selfupdate.chatfmt import adapt_records, stop_token_id
from selfupdate.config import load_config
from selfupdate.masking import ContextMasker, SegmentedExample
from selfupdate.teacher.cache import AsyncTeacherCacheWriter, resolve_cache_dir
from selfupdate.teacher.node_epoch0 import NodeEpoch0Lease, runtime_identity
from selfupdate.train.runtime import (load_causal_lm, pp_device_map,
                                      uses_pipeline_map)


def _source_commit(root: Path) -> str:
    """Resolve provenance, preferring real git.

    SELFUPDATE_SOURCE_COMMIT overrides, and the .git/HEAD walk below is the
    last resort, for any environment where the git binary is unavailable.
    """
    explicit = os.environ.get("SELFUPDATE_SOURCE_COMMIT")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_dir = root / ".git"
        if git_dir.is_file():
            marker = git_dir.read_text().strip()
            if marker.startswith("gitdir:"):
                git_dir = (root / marker.split(":", 1)[1].strip()).resolve()
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(":", 1)[1].strip()
        loose_ref = git_dir / ref
        if loose_ref.exists():
            return loose_ref.read_text().strip()
        packed = git_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line and not line.startswith(("#", "^")):
                    value, name = line.split(" ", 1)
                    if name == ref:
                        return value
        return "unknown"


def _native_mxfp4_load_overrides(model_name: str) -> dict:
    """Keep GPT-OSS MXFP4 checkpoints quantized instead of bf16 fallback.

    Transformers silently flips pre-quantized MXFP4 checkpoints to bf16
    dequantization when the `kernels` package is absent. That fallback is
    harmless for 20B-class models but fatal for 120B on 80GB cards, where the
    load later fails as a CUDA materialization error. Fail early with the local
    pinned remedy instead.
    """
    cfg = AutoConfig.from_pretrained(model_name)
    qc = getattr(cfg, "quantization_config", None) or getattr(
        getattr(cfg, "text_config", None), "quantization_config", None)
    if not (isinstance(qc, dict) and qc.get("quant_method") == "mxfp4"):
        return {}
    if not is_triton_available("3.4.0"):
        raise RuntimeError(
            "MXFP4 teacher cache requires Triton >= 3.4.0; keep torch fixed "
            "and repair the venv instead (scripts/venv_setup.sh).")
    if not is_kernels_available():
        raise RuntimeError(
            "MXFP4 teacher cache requires kernels==0.12.0. Rebuild the venv "
            "with scripts/venv_setup.sh, which pins it.")
    return {
        "quantization_config": Mxfp4Config(
            modules_to_not_convert=qc.get("modules_to_not_convert"),
            dequantize=False,
        )
    }


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
    source_commit = _source_commit(Path(__file__).resolve().parent.parent)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--model", default=None,
                    help="override model.name for cache benchmark campaigns")
    ap.add_argument("--pipeline-split", type=int, default=None,
                    help="place decoder blocks before/after this index on two GPUs")
    ap.add_argument("--pipeline-splits", type=int, nargs="+", default=None,
                    help="decoder block boundaries for N-way pipeline placement")
    ap.add_argument("--device-map-auto", action="store_true",
                    help="use Hugging Face automatic model placement")
    ap.add_argument("--generation-batch", type=int, default=None,
                    help="override cache.generation_batch")
    ap.add_argument("--teacher-batch", type=int, default=None,
                    help="override cache.teacher_batch for hidden-state forwards")
    ap.add_argument("--max-sequence-tokens", type=int, default=None,
                    help="override cache.max_sequence_tokens; e.g. 8192")
    ap.add_argument("--hidden-dtype", choices=("float16", "bfloat16"),
                    default=None,
                    help="override cache.hidden_dtype for stored teacher states")
    ap.add_argument("--generation-budget-bucket", type=int, default=None,
                    help="1=exact allowances, N=round up to N, 0=one outer group")
    ap.add_argument("--generation-max-tokens", type=int, default=None,
                    help="fixed per-answer ceiling; replaces proportional budgets")
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
    ap.add_argument(
        "--index-only", action="store_true",
        help="write ONLY index.json (spans + generated answer ids) from an "
             "imported --generation-responses file: the cache for "
             "v4_teacher_source=online, where hidden states are computed on "
             "the GPUs at train time. No model is loaded; megabytes, not "
             "gigabytes.")
    ap.add_argument("--generation-only", action="store_true",
                    help="benchmark/write teacher answers without hidden-state caching")
    ap.add_argument("--generation-responses", default=None,
                    help="reuse exact token_ids from a completed response JSONL")
    ap.add_argument(
        "--coordinated-node-cache", action="store_true",
        help=("materialize epoch-zero targets under cache.node_root with an "
              "atomic per-host lease; concurrent/repeated builders wait or reuse"))
    ap.add_argument(
        "--node-cache-wait-seconds", type=float, default=7200.0,
        help="maximum wait for another process building the same node cache")
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)
    if args.model is not None:
        cfg.model.name = args.model
    if args.pipeline_split is not None:
        cfg.model.pipeline_split = args.pipeline_split
    if args.pipeline_splits is not None:
        cfg.model.pipeline_splits = args.pipeline_splits
    if args.device_map_auto:
        cfg.model.device_map = "auto"
    if uses_pipeline_map(cfg) and cfg.model.device_map:
        raise ValueError(
            "model.pipeline_split(s) and model.device_map are mutually exclusive")
    if args.generation_batch is not None:
        cfg.cache.generation_batch = args.generation_batch
    if args.teacher_batch is not None:
        cfg.cache.teacher_batch = args.teacher_batch
    if args.max_sequence_tokens is not None:
        cfg.cache.max_sequence_tokens = args.max_sequence_tokens
    if args.hidden_dtype is not None:
        cfg.cache.hidden_dtype = args.hidden_dtype
    if args.generation_budget_bucket is not None:
        cfg.cache.generation_budget_bucket = args.generation_budget_bucket
    if args.generation_max_tokens is not None:
        if args.generation_max_tokens < 1:
            raise ValueError("--generation-max-tokens must be positive")
        cfg.cache.generation_max_tokens = args.generation_max_tokens
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
        if cfg.cache.runtime_policy == "node_epoch0":
            cfg.cache.node_root = args.cache_root
        else:
            cfg.cache.root = args.cache_root
    if args.limit is not None:
        cfg.cache.limit = args.limit
    if args.generation_responses is not None:
        cfg.cache.generation_responses_path = args.generation_responses

    if args.coordinated_node_cache and args.generation_only:
        raise ValueError(
            "--coordinated-node-cache publishes complete hidden targets; "
            "it cannot be combined with --generation-only")
    if (cfg.cache.runtime_policy == "node_epoch0"
            and not args.coordinated_node_cache):
        raise ValueError(
            "cache.runtime_policy=node_epoch0 must use "
            "--coordinated-node-cache; direct writes could expose partial "
            "safetensor shards to another training arm")
    if (cfg.cache.source_compaction
            and cfg.cache.source_compaction != cfg.mask.compaction):
        if not args.coordinated_node_cache:
            raise ValueError(
                "cache.source_compaction is a training-reader selector; cache "
                "generation must leave it empty or equal mask.compaction")
        # V3 node materialization intentionally uses the training arm's view
        # while targeting the uncensored teacher sequence. The aligned teacher
        # rows and answer ids are view-independent; readers explicitly ignore
        # student s0/position-gap metadata on a cross-view cache.

    root, chash = resolve_cache_dir(cfg)
    final_root = root
    node_lease = None
    if args.coordinated_node_cache:
        if cfg.cache.runtime_policy != "node_epoch0":
            raise ValueError(
                "--coordinated-node-cache requires "
                "cache.runtime_policy=node_epoch0")
        node_lease = NodeEpoch0Lease(
            final_root, chash, compatibility=runtime_identity(),
            wait_seconds=args.node_cache_wait_seconds)
        build_root, ready = node_lease.acquire()
        if ready is not None:
            print(
                f"epoch0 cache reuse: root={final_root} "
                f"examples={ready['examples']} host={ready['host']} "
                f"hash={ready['cache_hash']}")
            return
        root = build_root
        print(
            f"epoch0 cache lease acquired: host={node_lease.host} "
            f"pid={node_lease.pid} final={final_root}")
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
    if args.index_only and not cfg.cache.generation_responses_path:
        sys.exit("--index-only requires --generation-responses (vLLM "
                 "answers with exact token ids)")
    if args.index_only and cfg.cache.store_full_teacher_inputs:
        sys.exit("--index-only pairs with store_full_teacher_inputs=false "
                 "(the online source computes hidden states itself)")
    import_only = bool((args.generation_only or args.index_only)
                       and cfg.cache.generation_responses_path)
    model = None
    decoder = None
    n_layers = 0
    if not import_only:
        try:
            model_dtype = getattr(torch, cfg.model.dtype)
        except AttributeError as exc:
            raise ValueError(f"unknown model.dtype {cfg.model.dtype!r}") from exc
        load_kwargs = {"dtype": model_dtype, "low_cpu_mem_usage": True}
        load_kwargs.update(_native_mxfp4_load_overrides(cfg.model.name))
        if uses_pipeline_map(cfg):
            placement = pp_device_map(cfg)
            stage_devices = list(cfg.model.pipeline_devices or [])
            if not stage_devices:
                stage_devices = list(range(
                    len(cfg.model.pipeline_splits or []) + 1))
            # Preserve physical sparse placement during cache materialization.
            # Transformers' allocator warmup otherwise uses the process-wide
            # cuda:0 default even when every declared model tensor belongs to
            # stages such as [1, 3].
            torch.cuda.set_device(stage_devices[0])
            load_kwargs["device_map"] = placement
        elif cfg.model.device_map:
            if cfg.model.device_map != "auto":
                raise ValueError("model.device_map must be empty or 'auto'")
            load_kwargs["device_map"] = "auto"
        model = load_causal_lm(cfg.model.name, **load_kwargs)
        if not uses_pipeline_map(cfg) and not cfg.model.device_map:
            model.to(cfg.model.device)
        model.eval()
        # Text-only cache targets come from the decoder body.  Multimodal Gemma
        # 4 wraps that body under model.language_model; ordinary causal LMs
        # expose it directly as model.  Calling the body avoids materializing a
        # full [sequence, vocabulary] logits tensor.
        decoder = getattr(model.model, "language_model", model.model)
        n_layers = decoder.config.num_hidden_layers

    input_device = (model.get_input_embeddings().weight.device
                    if model is not None else torch.device(cfg.model.device))
    cuda_devices = sorted({parameter.device for parameter in model.parameters()
                           if parameter.device.type == "cuda"},
                          key=lambda device: device.index or 0) if model is not None else []

    def sync() -> None:
        # Phase timings must not assign queued CUDA work to the next CPU
        # section. The cache loop is dependency-serial already (generation →
        # forward → D2H write → next forward), so these synchronization
        # points measure existing waits rather than creating a new overlap.
        if model is not None and torch.cuda.is_available():
            for device in cuda_devices:
                torch.cuda.synchronize(device)

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
    timings.update({
        "model": cfg.model.name,
        "source_commit": source_commit,
        "pipeline_split": cfg.model.pipeline_split,
        "pipeline_splits": cfg.model.pipeline_splits,
        "device_map": cfg.model.device_map,
        "cuda_devices": [str(device) for device in cuda_devices],
    })
    hidden_bytes = 0
    teacher_compute_events = []
    started_at = time.perf_counter()
    generated_answers: list[tuple[list[int], bool]] = []
    generated_prompt_ids: list[list[int] | None] = []
    imported_answer_texts: list[str | None] = []
    effective_generation_batches: list[int] = []
    if v5:
        prompts = [masker.build(ex).teacher_ids for ex in examples]
        budgets = [
            _generation_budget(masker, ex,
                               int(record.get("expected_answer_chars", 64)),
                               cfg.cache.generation_extra_tokens)
            for record, ex in zip(records, examples)
        ]
        if cfg.cache.generation_max_tokens:
            if cfg.cache.generation_max_tokens < 1:
                raise ValueError("cache.generation_max_tokens must be non-negative")
            budgets = [cfg.cache.generation_max_tokens] * len(examples)
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
                recorded_budget = row.get("generation_budget")
                if (recorded_budget is not None
                        and int(recorded_budget) != budget):
                    raise ValueError(
                        f"generation response {ex.example_id} used budget "
                        f"{recorded_budget}, expected {budget} from cache config")
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
                prompt_ids = row.get("prompt_token_ids")
                if prompt_ids is not None:
                    if not isinstance(prompt_ids, list) or not prompt_ids:
                        raise ValueError(
                            f"generation response {ex.example_id} has invalid "
                            "prompt_token_ids")
                    prompt_ids = [int(token_id) for token_id in prompt_ids]
                generated_answers.append((ids, bool(row["hard_cut"])))
                generated_prompt_ids.append(prompt_ids)
                answer_text = row.get("answer_text")
                imported_answer_texts.append(
                    answer_text if isinstance(answer_text, str) else None)
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
            generated_prompt_ids = [list(prompt) for prompt in prompts]
            imported_answer_texts = [None] * len(generated_answers)
            sync()
            timings["generation_seconds"] = time.perf_counter() - t_phase
        for item_no, (record, ex, (answer_ids, hard_cut)) in enumerate(zip(
                records, examples, generated_answers)):
            imported_text = imported_answer_texts[item_no]
            answer_text = (imported_text if imported_text is not None
                           else tok.decode(answer_ids[:-1]))
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
    timings["generation_max_tokens"] = cfg.cache.generation_max_tokens
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

    if args.index_only:
        index = {"config_hash": chash,
                 "hidden_dtype": cfg.cache.hidden_dtype,
                 "full_teacher_inputs": False,
                 "teacher_targets": "online",
                 "examples": {}}
        for item_no, ex in enumerate(examples):
            answer_ids, hard_cut = generated_answers[item_no]
            pair = masker.build(ex, answer_ids=answer_ids)
            index["examples"][ex.example_id] = {
                "shard": -1,
                "t0": pair.t_aligned.start, "s0": pair.s_aligned.start,
                "A": pair.aligned_len,
                "mid_len": pair.s_answer.start - pair.s_aligned.start,
                "position_gap": pair.position_gap,
                "n_teacher": len(pair.teacher_ids),
                "n_student": len(pair.student_ids),
                "answer_ids": answer_ids,
                "hard_cut": hard_cut,
            }
        (root / "index.json").write_text(
            json.dumps(index) + "\n", encoding="utf-8")
        timings["total_seconds"] = time.perf_counter() - started_at
        (root / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
        print(f"index-only cache: {len(examples)} examples -> {root}")
        # Node-epoch0 consumers reject an unpublished directory, even when
        # the online v4 source needs only this index.  Publish this small
        # cache through the same atomic lease protocol as hidden-state caches;
        # otherwise atexit removes the partial directory on return.
        if node_lease is not None:
            manifest = node_lease.publish(root, {
                "model": cfg.model.name,
                "model_dtype": cfg.model.dtype,
                "hidden_dtype": cfg.cache.hidden_dtype,
                "source_commit": source_commit,
                "generation_responses_path": cfg.cache.generation_responses_path,
                "generation_max_tokens": cfg.cache.generation_max_tokens,
                "index_only": True,
                "total_seconds": timings["total_seconds"],
            })
            print(
                f"epoch0 cache ready: root={final_root} "
                f"examples={manifest['examples']} hash={manifest['cache_hash']}")
        return

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
        root, chash, shard_size=(
            cfg.cache.full_input_shard_size
            if cfg.cache.store_full_teacher_inputs else cfg.cache.shard_size),
        hidden_dtype=cfg.cache.hidden_dtype)
    if cfg.cache.store_full_teacher_inputs and cfg.cache.full_input_shard_size < 1:
        raise ValueError("cache.full_input_shard_size must be positive")
    copy_streams = ({device: torch.cuda.Stream(device=device)
                     for device in cuda_devices}
                    if torch.cuda.is_available() else {})
    teacher_progress_path = root / "teacher_progress.jsonl"
    teacher_progress_path.write_text("")
    teacher_wall_started = time.perf_counter()
    if cfg.cache.teacher_batch < 1:
        raise ValueError("cache.teacher_batch must be positive")
    teacher_items = []
    prompt_protocol_overrides = 0
    for item_no, (record, ex) in enumerate(zip(records, examples)):
        extra = None
        if v5:
            answer_ids, hard_cut = generated_answers[item_no]
            pair = masker.build(ex, answer_ids=answer_ids)
            extra = {"answer_ids": answer_ids, "hard_cut": hard_cut}
            exact_prompt_ids = generated_prompt_ids[item_no]
            if exact_prompt_ids is not None:
                native_prompt_ids = pair.teacher_ids[:-len(answer_ids)]
                if exact_prompt_ids != native_prompt_ids:
                    # Harmony and other non-native generation protocols can
                    # place the memory in a developer/tool message that the
                    # ordinary chat-template renderer cannot reconstruct.
                    # Preserve the exact teacher conditioning and align only
                    # the completion tokens, which remain identical on both
                    # sides.  Never pretend the unlike prompt headers/memory
                    # spans are a shared hidden-target region.
                    pair.teacher_ids = exact_prompt_ids + answer_ids
                    pair.t_aligned = slice(
                        len(exact_prompt_ids), len(pair.teacher_ids))
                    pair.t_answer = pair.t_aligned
                    pair.s_aligned = pair.s_answer
                    pair.position_gap = (
                        pair.t_aligned.start - pair.s_aligned.start)
                    pair.t_privileged = []
                    extra["teacher_prompt_ids"] = exact_prompt_ids
                    prompt_protocol_overrides += 1
        else:
            pair = masker.build(ex)
        teacher_items.append((record, ex, pair, extra))
    timings["prompt_protocol_overrides"] = prompt_protocol_overrides

    teacher_indices = list(range(len(teacher_items)))
    teacher_batches: list[list[int]] = []
    if cfg.cache.teacher_batch > 1 and cfg.cache.generation_shuffle_seed:
        # Length alignment limits padding while deterministic randomization of
        # members and batch order avoids an ordered long-sequence tail. Cache
        # entries remain keyed by example id, so physical shard order has no
        # training semantics.
        rng = random.Random(cfg.cache.generation_shuffle_seed)
        rng.shuffle(teacher_indices)
        length_buckets: dict[int, list[int]] = {}
        for index in teacher_indices:
            width = len(teacher_items[index][2].teacher_ids)
            length_buckets.setdefault(math.ceil(width / 128) * 128, []).append(index)
        for group in length_buckets.values():
            teacher_batches.extend(
                group[start:start + cfg.cache.teacher_batch]
                for start in range(0, len(group), cfg.cache.teacher_batch))
        rng.shuffle(teacher_batches)
    else:
        teacher_batches = [
            teacher_indices[start:start + cfg.cache.teacher_batch]
            for start in range(0, len(teacher_indices), cfg.cache.teacher_batch)]

    pending_batches = list(reversed(teacher_batches))
    safe_teacher_batch = cfg.cache.teacher_batch
    effective_teacher_batches: list[int] = []
    completed_teacher = 0
    progress = tqdm(total=len(examples), desc="teacher forward")
    while pending_batches:
        indices = pending_batches.pop()
        if len(indices) > safe_teacher_batch:
            chunks = [indices[start:start + safe_teacher_batch]
                      for start in range(0, len(indices), safe_teacher_batch)]
            pending_batches.extend(reversed(chunks))
            continue
        width = max(len(teacher_items[index][2].teacher_ids) for index in indices)
        t_ids = torch.full(
            (len(indices), width), stop_id, dtype=torch.long, device=input_device)
        attention_mask = torch.zeros_like(t_ids)
        for row, index in enumerate(indices):
            ids = teacher_items[index][2].teacher_ids
            t_ids[row, :len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=input_device)
            attention_mask[row, :len(ids)] = 1
        if len(cuda_devices) == 1:
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            compute_start.record(torch.cuda.current_stream(input_device))
        else:
            sync()
            t_phase = time.perf_counter()
        try:
            with torch.no_grad():
                if len(indices) == 1:
                    # Preserve the historical B=1 call exactly: no explicit
                    # all-ones mask and therefore no numerics change at the
                    # default teacher_batch=1.
                    out = decoder(
                        t_ids, output_hidden_states=True, use_cache=False)
                else:
                    out = decoder(
                        t_ids, attention_mask=attention_mask,
                        output_hidden_states=True, use_cache=False)
        except RuntimeError as exc:
            if not _is_cuda_oom(exc) or len(indices) <= 1:
                raise
            del t_ids, attention_mask
            safe_teacher_batch = max(1, len(indices) // 2)
            chunks = [indices[start:start + safe_teacher_batch]
                      for start in range(0, len(indices), safe_teacher_batch)]
            pending_batches.extend(reversed(chunks))
            torch.cuda.empty_cache()
            print(f"teacher forward OOM: retrying at batch {safe_teacher_batch}",
                  flush=True)
            continue
        if len(cuda_devices) == 1:
            compute_end.record(torch.cuda.current_stream(input_device))
            teacher_compute_events.append((compute_start, compute_end))
        else:
            sync()
            timings["teacher_forward_seconds"] += time.perf_counter() - t_phase
        effective_teacher_batches.append(len(indices))
        for row, index in enumerate(indices):
            _, ex, pair, extra = teacher_items[index]
            span = pair.t_aligned
            # One packed, asynchronous D2H transfer replaces one implicit CUDA
            # synchronization per layer.  A dedicated copy stream and background
            # writer overlap transfer, finite checks, and /tmp shard writes with
            # the next teacher forward while preserving per-layer cache tensors.
            t_phase = time.perf_counter()
            layer_states = [out.hidden_states[L][row, span.start:span.stop]
                            for L in range(1, n_layers + 1)]
            # The optional independent-stage cache stores the complete input
            # to every block, not merely aligned targets.  Keep it under an
            # explicit cache identity because its storage contract is much
            # larger.  These copies occur during cache construction, outside
            # the measured training traversal.
            teacher_inputs = None
            if cfg.cache.store_full_teacher_inputs:
                teacher_length = len(pair.teacher_ids)
                teacher_inputs = {
                    L: out.hidden_states[L - 1][
                        row, :teacher_length].detach().to(
                            writer.hidden_dtype).cpu()
                    for L in range(1, n_layers + 1)
                }
            if "hidden_state_devices" not in timings:
                device_counts = {}
                for state in layer_states:
                    key = str(state.device)
                    device_counts[key] = device_counts.get(key, 0) + 1
                timings["hidden_state_devices"] = device_counts
            layer_devices = {state.device for state in layer_states}
            if copy_streams and len(layer_devices) == 1:
                packed_gpu = torch.stack(layer_states).to(writer.hidden_dtype)
                finite_gpu = torch.isfinite(packed_gpu).all()
                packed_hidden = torch.empty_like(
                    packed_gpu, device="cpu", pin_memory=True)
                finite_flag = torch.empty(
                    (), dtype=torch.bool, device="cpu", pin_memory=True)
                device = packed_gpu.device
                copy_stream = copy_streams[device]
                copy_stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(copy_stream):
                    copy_start_event = torch.cuda.Event(enable_timing=True)
                    copy_start_event.record(copy_stream)
                    packed_hidden.copy_(packed_gpu, non_blocking=True)
                    finite_flag.copy_(finite_gpu, non_blocking=True)
                    packed_gpu.record_stream(copy_stream)
                    finite_gpu.record_stream(copy_stream)
                    ready_event = torch.cuda.Event(enable_timing=True)
                    ready_event.record(copy_stream)
            elif copy_streams:
                # Explicit PP leaves each layer's state on its owning card.
                # Copy card-local layer groups into slices of one pinned CPU
                # tensor.  The writer waits for all card events, while their
                # D2H transfers and the next teacher forward may overlap.
                shape = (n_layers, span.stop - span.start,
                         layer_states[0].shape[-1])
                packed_hidden = torch.empty(
                    shape, dtype=writer.hidden_dtype, device="cpu",
                    pin_memory=True)
                copy_start_event = []
                ready_event = []
                for layer_no, state in enumerate(layer_states):
                    if state.device.type == "cpu":
                        packed_hidden[layer_no].copy_(
                            state.to(writer.hidden_dtype))
                cuda_layer_devices = {
                    device for device in layer_devices
                    if device.type == "cuda"}
                for device in sorted(cuda_layer_devices,
                                     key=lambda value: value.index or 0):
                    stream = copy_streams[device]
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        start_event = torch.cuda.Event(enable_timing=True)
                        start_event.record(stream)
                        for layer_no, state in enumerate(layer_states):
                            if state.device != device:
                                continue
                            converted = state.to(writer.hidden_dtype)
                            packed_hidden[layer_no].copy_(
                                converted, non_blocking=True)
                            converted.record_stream(stream)
                        end_event = torch.cuda.Event(enable_timing=True)
                        end_event.record(stream)
                    copy_start_event.append(start_event)
                    ready_event.append(end_event)
                finite_flag = None
            else:
                packed_gpu = torch.stack(layer_states).to(writer.hidden_dtype)
                packed_hidden = packed_gpu.cpu()
                finite_flag = None
                copy_start_event = None
                ready_event = None
            hidden_bytes += packed_hidden.numel() * packed_hidden.element_size()
            if teacher_inputs:
                hidden_bytes += sum(
                    value.numel() * value.element_size()
                    for value in teacher_inputs.values())
            hidden = {L: packed_hidden[L - 1] for L in range(1, n_layers + 1)}
            writer.add(
                ex.example_id, hidden,
                span={
                    "t0": pair.t_aligned.start, "s0": pair.s_aligned.start,
                    "A": pair.aligned_len,
                    "mid_len": pair.s_answer.start - pair.s_aligned.start,
                    "position_gap": pair.position_gap,
                    "n_teacher": len(pair.teacher_ids),
                    "n_student": len(pair.student_ids),
                },
                extra=extra,
                teacher_inputs=teacher_inputs,
                copy_start_event=copy_start_event,
                ready_event=ready_event,
                finite_flag=finite_flag,
            )
            timings["cache_write_seconds"] += time.perf_counter() - t_phase
            completed_teacher += 1
            progress.update(1)
            if completed_teacher % 100 == 0 or completed_teacher == len(examples):
                elapsed = time.perf_counter() - teacher_wall_started
                with teacher_progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "completed": completed_teacher,
                        "total": len(examples),
                        "elapsed_seconds": elapsed,
                        "examples_per_second": completed_teacher / elapsed,
                        "hidden_bytes_queued": hidden_bytes,
                        "effective_batch": len(indices),
                    }) + "\n")
        del out, t_ids, attention_mask
    progress.close()

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
    timings["requested_teacher_batch"] = cfg.cache.teacher_batch
    timings["minimum_effective_teacher_batch"] = (
        min(effective_teacher_batches) if effective_teacher_batches else 0)
    timings["maximum_effective_teacher_batch"] = (
        max(effective_teacher_batches) if effective_teacher_batches else 0)
    timings["examples"] = len(examples)
    timings["requested_generation_batch"] = cfg.cache.generation_batch
    timings["generation_max_tokens"] = cfg.cache.generation_max_tokens
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
    if node_lease is not None:
        manifest = node_lease.publish(root, {
            "model": cfg.model.name,
            "model_dtype": cfg.model.dtype,
            "hidden_dtype": cfg.cache.hidden_dtype,
            "gpu_names": [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ],
            "source_commit": source_commit,
            "generation_responses_path": cfg.cache.generation_responses_path,
            "generation_max_tokens": cfg.cache.generation_max_tokens,
            "teacher_batch": cfg.cache.teacher_batch,
            "total_seconds": timings["total_seconds"],
        })
        print(
            f"epoch0 cache ready: root={final_root} "
            f"examples={manifest['examples']} hash={manifest['cache_hash']}")


if __name__ == "__main__":
    main()
