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
import socket
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
from selfupdate.eval.standard import (BENCHMARK_REVISIONS, STANDARD_TASKS,
                                      evaluate_task)


TASKS = ("wikitext2_ppl", *STANDARD_TASKS)
STAGE_LOCK_STALE_SECONDS = 15 * 60


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _stage_lock_is_stale(lock: Path) -> bool:
    """A node-local lock is stale when its local owner died or timed out."""
    try:
        owner = json.loads((lock / "owner.json").read_text())
    except (OSError, json.JSONDecodeError):
        owner = {}
    pid = owner.get("pid")
    host = owner.get("hostname")
    if pid and host == socket.gethostname():
        try:
            os.kill(int(pid), 0)
        except (ProcessLookupError, PermissionError):
            return True
        else:
            return False
    try:
        age = time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        return False
    timeout = float(os.environ.get(
        "SELFUPDATE_EVAL_STAGE_LOCK_STALE_SECONDS", STAGE_LOCK_STALE_SECONDS))
    return age >= timeout


def _write_stage_lock_owner(lock: Path) -> None:
    (lock / "owner.json").write_text(json.dumps({
        "pid": os.getpid(), "hostname": socket.gethostname(),
        "started_at": time.time(),
    }))


def _sweep_stage_root(root: Path) -> None:
    """Age-gated janitor for OUR staging root only (~170 GB/node unswept at
    full fleet). Shared snapshots expire SELFUPDATE_EVAL_STAGE_TTL_DAYS
    (default 7) after their last use (reuse refreshes the .complete marker);
    dirs holding a live build lock are never touched. jobs/ dirs are
    atexit-cleaned normally but leak on SIGKILL — same TTL reaps orphans
    (eval jobs run minutes-hours, never days)."""
    ttl = float(os.environ.get("SELFUPDATE_EVAL_STAGE_TTL_DAYS", "7")) * 86400
    now = time.time()
    for kind in ("models", "jobs"):
        base = root / kind
        if not base.is_dir():
            continue
        for d in base.iterdir():
            if not d.is_dir() or d.name.endswith(".lock"):
                continue
            if kind == "models" and d.with_name(d.name + ".lock").exists():
                continue  # being (re)built right now
            marker = d / ".complete"
            ref = marker if marker.exists() else d
            try:
                idle = now - ref.stat().st_mtime
            except OSError:
                continue
            if idle >= ttl:
                print(f"stage janitor: removing {d} "
                      f"(idle {idle / 86400:.1f} d)", file=sys.stderr)
                shutil.rmtree(d, ignore_errors=True)


def _stage_source(source: str, label: str, shared: bool) -> str:
    """Copy a model/checkpoint to node-local XFS before safetensor mmap.

    Shared model snapshots are retained across jobs on this node. Unique
    checkpoints are removed at process exit. A mkdir lock prevents concurrent
    jobs from constructing the same shared snapshot; dead owners are reclaimed
    so a killed copy cannot wedge every later evaluation on the node.
    """
    from huggingface_hub import snapshot_download

    src = Path(source)
    if not src.exists():
        src = Path(snapshot_download(source, local_files_only=True))
    root = Path(os.environ.get(
        "SELFUPDATE_EVAL_STAGE", f"/tmp/{os.environ.get('USER', 'user')}/selfupdate-eval-stage"))
    root.mkdir(parents=True, exist_ok=True)
    _sweep_stage_root(root)
    if shared:
        dest = root / "models" / _safe_name(label)
        lock = dest.with_name(dest.name + ".lock")
        dest.parent.mkdir(parents=True, exist_ok=True)
        while not (dest / ".complete").exists():
            try:
                lock.mkdir()
                _write_stage_lock_owner(lock)
                owner = True
            except FileExistsError:
                owner = False
            if owner:
                try:
                    # Atomic publish normally prevents this state, but clean a
                    # partial directory left by an interrupted older stager.
                    if dest.exists() and not (dest / ".complete").exists():
                        shutil.rmtree(dest)
                    tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
                    shutil.rmtree(tmp, ignore_errors=True)
                    shutil.copytree(src, tmp, symlinks=False)
                    (tmp / ".complete").touch()
                    if not dest.exists():
                        tmp.rename(dest)
                    else:
                        shutil.rmtree(tmp, ignore_errors=True)
                finally:
                    shutil.rmtree(lock, ignore_errors=True)
                break
            if _stage_lock_is_stale(lock):
                # The lock directory is the atomic ownership token.  Delete
                # only a dead/expired token, then contend normally next loop.
                shutil.rmtree(lock, ignore_errors=True)
                continue
            time.sleep(0.2)
        # Reuse refreshes the marker: the janitor's TTL then measures time
        # since last USE, so an actively shared snapshot never expires.
        try:
            os.utime(dest / ".complete")
        except OSError:
            pass
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


def _wikitext2_text(max_chars: int | None) -> str:
    # Serial download/load only. Do not use num_proc on Lustre.
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="validation",
        revision=BENCHMARK_REVISIONS["Salesforce/wikitext"])
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


def _chunks(xs: list, n: int) -> Iterable[list]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


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
            results[task] = evaluate_task(
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
        "benchmark_revisions": {
            "arc_easy": "data/eval/arc_easy_v1.json",
            "arc_challenge": BENCHMARK_REVISIONS["allenai/ai2_arc"],
            "hellaswag": BENCHMARK_REVISIONS["Rowan/hellaswag"],
            "wikitext2_ppl": BENCHMARK_REVISIONS["Salesforce/wikitext"],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
