"""Standard benchmark destruction check for checkpoints.

This is a lightweight replacement for the noisy tiny general-CE canary.  It
evaluates multiple-choice benchmark accuracy by scoring each answer option
with causal-LM continuation log-likelihood:

    prompt + option

Only option tokens contribute to the score, normalized per option token.

Initial tasks:
- WikiText-2 validation perplexity (quantization-style damage metric)
- ARC-Easy validation
- ARC-Challenge validation
- HellaSwag validation

The output JSON is meant to be compared checkpoint-vs-epoch-zero teacher for
the same base model.  It is not training data and never touches the memorized
reference corpora.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config


TASKS = ("wikitext2_ppl", "arc_easy", "arc_challenge", "hellaswag")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _stage_source(source: str, label: str, shared: bool) -> str:
    """Copy a model/checkpoint to node-local XFS before safetensor mmap.

    Shared model snapshots are retained across jobs on this node. Unique
    checkpoints are removed at process exit. A mkdir lock prevents concurrent
    jobs from constructing the same shared snapshot.
    """
    from huggingface_hub import snapshot_download

    src = Path(source)
    if not src.exists():
        src = Path(snapshot_download(source, local_files_only=True))
    root = Path(os.environ.get(
        "SELFUPDATE_EVAL_STAGE", f"/tmp/{os.environ.get('USER', 'user')}/selfupdate-eval-stage"))
    root.mkdir(parents=True, exist_ok=True)
    if shared:
        dest = root / "models" / _safe_name(label)
        lock = dest.with_name(dest.name + ".lock")
        dest.parent.mkdir(parents=True, exist_ok=True)
        while not (dest / ".complete").exists():
            try:
                lock.mkdir()
                owner = True
            except FileExistsError:
                owner = False
            if owner:
                try:
                    tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
                    shutil.rmtree(tmp, ignore_errors=True)
                    shutil.copytree(src, tmp, symlinks=False)
                    (tmp / ".complete").touch()
                    if not dest.exists():
                        tmp.rename(dest)
                    else:
                        shutil.rmtree(tmp, ignore_errors=True)
                finally:
                    lock.rmdir()
                break
            time.sleep(0.2)
        return str(dest)

    dest = root / "jobs" / f"{_safe_name(label)}-{os.getpid()}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, symlinks=False)
    atexit.register(shutil.rmtree, dest, True)
    return str(dest)


def _checkpoint_run_config(checkpoint: str | None) -> dict:
    if not checkpoint:
        return {}
    ckpt = Path(checkpoint)
    for path in (ckpt.parent / "config.yaml", ckpt / "config.yaml"):
        if path.exists():
            try:
                return yaml.safe_load(path.read_text()) or {}
            except Exception:  # noqa: BLE001
                return {}
    return {}


def _load_model(args):
    try:
        cfg = load_config(args.config, args.experiment)
    except ValueError:
        # Historical run/config.yaml files are useful source-of-truth for the
        # model identity but can contain trainer keys retired by the refactor.
        # Evaluation must not deserialize the obsolete training surface.
        cfg = load_config(args.config)
        if not args.experiment:
            raise
        saved = yaml.safe_load(Path(args.experiment).read_text()) or {}
        saved_model = (saved.get("model") or {}).get("name")
        if not saved_model:
            raise
        cfg.model.name = saved_model
    checkpoint_cfg = _checkpoint_run_config(args.checkpoint)
    checkpoint_model = ((checkpoint_cfg.get("model") or {}).get("name")
                        if checkpoint_cfg else None)
    if checkpoint_model and not args.base:
        cfg.model.name = checkpoint_model

    src = cfg.model.name if args.base else args.checkpoint
    if not src:
        raise SystemExit("pass --checkpoint or --base")
    adapter = bool(not args.base and (Path(src) / "adapter_config.json").exists())
    base_src = cfg.model.name
    if args.stage_to_local:
        if args.base or adapter:
            base_src = _stage_source(cfg.model.name, cfg.model.name, shared=True)
            if args.base:
                src = base_src
        if not args.base:
            src = _stage_source(src, Path(args.checkpoint).parent.name,
                                shared=False)

    load_kw = {}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    device_map = ("auto" if (args.auto_map or args.load_4bit)
                  else {"": args.device})
    if adapter:
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(src)
        base = AutoModelForCausalLM.from_pretrained(
            base_src,
            dtype=torch.bfloat16,
            device_map=device_map,
            **load_kw,
        )
        model = PeftModel.from_pretrained(base, src)
    else:
        tok = AutoTokenizer.from_pretrained(src)
        model = AutoModelForCausalLM.from_pretrained(
            src,
            dtype=torch.bfloat16,
            device_map=device_map,
            **load_kw,
        )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return cfg, model, tok


def _arc_examples(config: str, split: str, limit: int | None) -> list[dict]:
    if config == "ARC-Easy":
        # Repository-pinned standard subset: avoids dataset-revision drift and
        # guarantees that checkpoint and epoch-zero jobs see identical items.
        pinned = Path("data/eval/arc_easy_v1.json")
        if pinned.exists():
            rows = json.loads(pinned.read_text())["items"]
            return rows[:limit] if limit else rows
    ds = load_dataset("allenai/ai2_arc", config, split=split)
    rows = []
    for row in ds:
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        answer = str(row["answerKey"])
        if answer in labels:
            target = labels.index(answer)
        elif answer.isdigit() and str(int(answer)) in labels:
            target = labels.index(str(int(answer)))
        else:
            continue
        rows.append(
            {
                "id": row.get("id"),
                "prompt": f"Question: {row['question'].strip()}\nAnswer:",
                "choices": [f" {t.strip()}" for t in texts],
                "target": target,
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def _hellaswag_examples(split: str, limit: int | None) -> list[dict]:
    ds = load_dataset("Rowan/hellaswag", split=split)
    rows = []
    for row in ds:
        target = int(row["label"])
        ctx = f"{row['ctx_a']} {row['ctx_b']}".strip()
        rows.append(
            {
                "id": row.get("ind"),
                "prompt": ctx,
                "choices": [f" {e.strip()}" for e in row["endings"]],
                "target": target,
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def _wikitext2_text(max_chars: int | None) -> str:
    # Serial download/load only. Do not use num_proc on Lustre.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    parts = []
    chars = 0
    for row in ds:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        parts.append(text)
        chars += len(text) + 1
        if max_chars and chars >= max_chars:
            break
    return "\n\n".join(parts)


def _task_examples(task: str, limit: int | None) -> list[dict]:
    if task == "arc_easy":
        return _arc_examples("ARC-Easy", "validation", limit)
    if task == "arc_challenge":
        return _arc_examples("ARC-Challenge", "validation", limit)
    if task == "hellaswag":
        return _hellaswag_examples("validation", limit)
    raise ValueError(task)


def _chunks(xs: list, n: int) -> Iterable[list]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


@torch.no_grad()
def _score_pairs(model, tok, pairs: list[tuple[str, str]], device: str) -> list[float]:
    texts = [p + c for p, c in pairs]
    # padding_side MUST be right: the span arithmetic below indexes from the
    # sequence start. Some fleet tokenizers (ALIA-40b) ship padding_side=left,
    # which silently scored pad/prompt tokens for every non-longest option.
    enc = tok(texts, return_tensors="pt", padding=True, padding_side="right",
              add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits

    scores = []
    for i, (prompt, choice) in enumerate(pairs):
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        choice_ids = tok.encode(choice, add_special_tokens=False)
        start = len(prompt_ids)
        end = start + len(choice_ids)
        if not choice_ids or end > int(attention_mask[i].sum().item()):
            scores.append(-math.inf)
            continue
        row_logits = logits[i, start - 1:end - 1].float()
        targets = input_ids[i, start:end].to(row_logits.device)
        nll = F.cross_entropy(row_logits, targets, reduction="sum").item()
        scores.append(-nll / max(1, len(choice_ids)))
    return scores


def _evaluate_task(model, tok, task: str, limit: int | None, batch_size: int, device: str) -> dict:
    examples = _task_examples(task, limit)
    correct = 0
    per_example = []
    flat = []
    owners = []
    for ex_i, ex in enumerate(examples):
        for choice_i, choice in enumerate(ex["choices"]):
            flat.append((ex["prompt"], choice))
            owners.append((ex_i, choice_i))

    scores_by_example = [[-math.inf] * len(ex["choices"]) for ex in examples]
    for batch, owner_batch in zip(_chunks(flat, batch_size), _chunks(owners, batch_size)):
        scores = _score_pairs(model, tok, batch, device)
        for (ex_i, choice_i), score in zip(owner_batch, scores):
            scores_by_example[ex_i][choice_i] = score

    for ex, scores in zip(examples, scores_by_example):
        pred = max(range(len(scores)), key=lambda i: scores[i])
        ok = pred == ex["target"]
        correct += int(ok)
        per_example.append(
            {
                "id": ex["id"],
                "target": ex["target"],
                "pred": pred,
                "correct": ok,
                "scores": scores,
            }
        )
    return {
        "task": task,
        "n": len(examples),
        "accuracy": correct / len(examples) if examples else float("nan"),
        "per_example": per_example,
    }


@torch.no_grad()
def _evaluate_wikitext2_ppl(
    model,
    tok,
    max_chars: int | None,
    batch_size: int,
    device: str,
    seq_len: int = 2048,
) -> dict:
    text = _wikitext2_text(max_chars)
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < 2:
        return {"task": "wikitext2_ppl", "n_tokens": 0, "mean_ce": float("nan"), "ppl": float("nan")}
    usable = (len(ids) - 1) // seq_len * seq_len
    if usable <= 0:
        usable = len(ids) - 1
    starts = list(range(0, usable, seq_len))
    total_nll = 0.0
    total_tokens = 0
    for chunk_starts in _chunks(starts, batch_size):
        seqs = []
        lengths = []
        for start in chunk_starts:
            chunk = ids[start:start + seq_len + 1]
            lengths.append(len(chunk) - 1)
            seqs.append(chunk)
        max_len = max(len(x) for x in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        x = torch.full((len(seqs), max_len), pad, dtype=torch.long, device=device)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
        for i, seq in enumerate(seqs):
            x[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attn[i, :len(seq)] = 1
        logits = model(x, attention_mask=attn, use_cache=False).logits
        for i, n in enumerate(lengths):
            row_logits = logits[i, :n].float()
            targets = x[i, 1:n + 1].to(row_logits.device)
            total_nll += F.cross_entropy(row_logits, targets, reduction="sum").item()
            total_tokens += n
    mean_ce = total_nll / max(1, total_tokens)
    return {
        "task": "wikitext2_ppl",
        "n_tokens": total_tokens,
        "mean_ce": mean_ce,
        "ppl": math.exp(min(50, mean_ce)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--wikitext-max-chars", type=int, default=250_000)
    ap.add_argument("--wikitext-seq-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--auto-map", action="store_true")
    ap.add_argument("--load-4bit", action="store_true",
                    help="NF4-load the model; implies device_map=auto")
    ap.add_argument("--stage-to-local", action="store_true",
                    help="stage weights under node-local /tmp before loading")
    args = ap.parse_args()

    cfg, model, tok = _load_model(args)
    results = {}
    for task in args.tasks:
        print(f"evaluating {task} limit={args.limit}", flush=True)
        device = cfg.model.device if args.device == "config" else args.device
        if task == "wikitext2_ppl":
            results[task] = _evaluate_wikitext2_ppl(
                model,
                tok,
                args.wikitext_max_chars,
                args.batch_size,
                device,
                seq_len=args.wikitext_seq_len,
            )
            print(
                f"{task}: ppl={results[task]['ppl']:.3f} "
                f"ce={results[task]['mean_ce']:.3f} "
                f"tokens={results[task]['n_tokens']}",
                flush=True,
            )
        else:
            results[task] = _evaluate_task(
                model, tok, task, args.limit, args.batch_size, device
            )
            print(f"{task}: acc={results[task]['accuracy']:.3f} n={results[task]['n']}", flush=True)

    summary = {
        "kind": "standard_destruction_eval",
        "model": cfg.model.name,
        "checkpoint": None if args.base else args.checkpoint,
        "teacher_reference_kind": "teacher_epoch0_native_no_rag" if args.base else "checkpoint",
        "tasks": results,
        "macro_accuracy": (
            sum(r["accuracy"] for r in results.values() if "accuracy" in r)
            / max(1, sum(1 for r in results.values() if "accuracy" in r))
        ),
        "limit": args.limit,
        "batch_size": args.batch_size,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
