"""Full recitation eval of a trained checkpoint (or the base model as control).

Usage:
    python scripts/evaluate.py --checkpoint runs/<name>/checkpoint [--limit N]
    python scripts/evaluate.py --base   # untrained control
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.data.dataset import load_jsonl
from selfupdate.eval.recite import recite_eval

# Quijote chapter rungs are distinct recall targets: a ch8-trained checkpoint
# must be scored on raw_ch8.txt, never on the ch1 subset (which is its
# best-trained prefix and silently flatters it). Keys match tasks_report.py.
CORPUS_PATHS = {
    "machado": "data/poem/raw.txt",
    "quijote_ch1": "data/quijote/raw_ch1.txt",
    "quijote_ch4": "data/quijote/raw_ch4.txt",
    "quijote_ch8": "data/quijote/raw_ch8.txt",
    "quijote_ch16": "data/quijote/raw_ch16.txt",
}


def quijote_rung(path: str | None) -> str | None:
    """'quijote_ch8' from '.../raw_ch8.txt' or '.../examples_ch8.jsonl'."""
    m = re.search(r"ch(\d+)", str(path or "").lower())
    return f"quijote_ch{m.group(1)}" if m else None


def _checkpoint_run_config(checkpoint: str | None) -> dict:
    if not checkpoint:
        return {}
    ckpt = Path(checkpoint)
    candidates = [
        ckpt.parent / "config.yaml",
        ckpt / "config.yaml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            return yaml.safe_load(path.read_text()) or {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _adopt_checkpoint_eval_config(cfg, checkpoint_cfg: dict):
    """Adopt checkpoint identity and input geometry for evaluation.

    The command-line base config chooses placement and other evaluator knobs;
    the checkpoint config is the source of truth for model identity, dataset,
    and masking.
    """
    if not checkpoint_cfg:
        return cfg

    saved_model = checkpoint_cfg.get("model") or {}
    if saved_model.get("name"):
        cfg.model.name = saved_model["name"]

    saved_data = checkpoint_cfg.get("data") or {}
    saved_mask = checkpoint_cfg.get("mask") or {}
    # Saved run configs can contain retired train keys, but data/mask are
    # stable dataclasses.  Copy only their declared fields so evaluation stays
    # compatible with historical configs without accepting misspellings.
    for target, saved in ((cfg.data, saved_data), (cfg.mask, saved_mask)):
        known = set(target.__dataclass_fields__)
        for key, value in saved.items():
            if key in known:
                setattr(target, key, value)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--base", action="store_true", help="evaluate the untrained base model")
    ap.add_argument("--out", default=None,
                    help="output dir override (multi-node: concurrent --base "
                         "evals must not share runs/base-eval-full)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-per-task", type=int, default=24,
                    help="items per task in the three-task battery")
    ap.add_argument(
        "--recall-corpora",
        nargs="+",
        choices=tuple(CORPUS_PATHS),
        default=None,
        help=("recall corpora to measure. By default this is inferred from "
              "the checkpoint's training data (chapter-rung-specific for "
              "Quijote); combined checkpoints measure both corpora"),
    )
    ap.add_argument("--max-extra-tokens", type=int, default=32,
                    help="per-item generation budget beyond the reference "
                         "length (default matches the battery's historical 32)")
    ap.add_argument("--generation-batch", type=int, default=8,
                    help="batched greedy decode for the three-task battery; "
                         "default 8; 1 = historical per-item loop")
    # Retired with the recite/CER engine (2026-07-10). Accepted so historical
    # queue rows don't crash at dispatch, but they are IGNORED and warn below
    # (knob-flow law: no knob may silently do nothing).
    ap.add_argument("--batch-size", type=int, default=1,
                    help="ignored (retired recite/CER engine flag)")
    ap.add_argument("--bucket-by-length", action="store_true",
                    help="ignored (retired recite/CER engine flag)")
    ap.add_argument("--score-workers", type=int, default=None,
                    help="ignored (retired recite/CER engine flag)")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="ignored (retired recite/CER engine flag)")
    ap.add_argument("--auto-map", action="store_true",
                    help="load with device_map=auto (multi-card eval, e.g. 32B)")
    ap.add_argument("--load-4bit", action="store_true",
                    help="load the base in NF4 4-bit (bitsandbytes) — lets a 40B "
                         "eval coexist with a resident 40B training job on the same "
                         "cards. Perturbs CER vs bf16; label results as 4-bit. "
                         "Implies device_map=auto and skips the .to(device) move "
                         "(bnb 4-bit tensors cannot be re-placed).")
    args = ap.parse_args()
    retired = [flag for flag, val, default in (
        ("--batch-size", args.batch_size, 1),
        ("--bucket-by-length", args.bucket_by_length, False),
        ("--score-workers", args.score_workers, None),
        ("--shuffle-seed", args.shuffle_seed, None),
    ) if val != default]
    if retired:
        print("WARNING: ignoring " + " ".join(retired) + " — retired with the "
              "recite/CER engine (2026-07-10); the three-task battery "
              "generates item-by-item.", file=sys.stderr)
    cfg = load_config(args.config, args.experiment)
    checkpoint_cfg = _checkpoint_run_config(args.checkpoint)
    if checkpoint_cfg and not args.base:
        cfg = _adopt_checkpoint_eval_config(cfg, checkpoint_cfg)

    src = cfg.model.name if args.base else args.checkpoint
    if not src:
        sys.exit("pass --checkpoint or --base")
    # 4-bit forces device_map=auto (bnb places shards itself); the .to(device)
    # move below is skipped because bnb 4-bit params cannot be re-placed.
    load_kw: dict = {}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    dev_map = "auto" if (args.auto_map or args.load_4bit) else None
    if not args.base and (Path(src) / "adapter_config.json").exists():
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(src)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, dtype=torch.bfloat16, device_map=dev_map, **load_kw)
        model = PeftModel.from_pretrained(base, src)
    else:
        tok = AutoTokenizer.from_pretrained(src)
        model = AutoModelForCausalLM.from_pretrained(
            src, dtype=torch.bfloat16, device_map=dev_map, **load_kw)
    if dev_map is None:
        model.to(cfg.model.device)
    model.eval()

    # the three-task battery (owner directive 2026-07-10): next / prev /
    # cloze with plain accuracies — CER and the other recovery metrics are
    # retired from the active eval surface
    from selfupdate.eval.tasks import tasks_eval

    if args.recall_corpora:
        corpus_names = list(dict.fromkeys(args.recall_corpora))
    else:
        # examples_path is authoritative for combined training configs, which
        # intentionally inherit base.yaml's Machado poem_path. Looking only at
        # poem_path made the old report silently call a Machado-only evaluation
        # a combined result. The Quijote rung comes from the checkpoint's own
        # paths — checkpoints score on THEIR corpus, never a fixed chapter.
        examples_path = str(cfg.data.examples_path)
        poem_path = str(cfg.data.poem_path)
        if "combined" in examples_path:
            corpus_names = ["machado",
                            quijote_rung(examples_path) or "quijote_ch1"]
        elif "quijote" in examples_path or "quijote" in poem_path:
            corpus_names = [quijote_rung(examples_path)
                            or quijote_rung(poem_path) or "quijote_ch1"]
        else:
            corpus_names = ["machado"]

    corpus_results = {}
    for corpus in corpus_names:
        result = tasks_eval(model, tok, CORPUS_PATHS[corpus],
                            n_per_task=args.n_per_task,
                            max_extra_tokens=args.max_extra_tokens,
                            generation_batch=args.generation_batch)
        result["poem_path"] = CORPUS_PATHS[corpus]
        corpus_results[corpus] = result
        parts = "  ".join(
            f"{t}: exact {v['exact']:.2f} words {v['word_acc']:.2f}"
            for t, v in result["tasks"].items())
        print(f"{corpus}: {parts}")

    r = {
        "schema_version": 2,
        "teacher_reference_kind": (
            "teacher_epoch0_native_no_rag" if args.base else "checkpoint"),
        "model": cfg.model.name,
        "corpora_measured": corpus_names,
        "corpus_selection": ("cli_override" if args.recall_corpora
                             else "inferred_from_training_data"),
        "corpora": corpus_results,
    }
    # training_scope is only honest when the corpora were inferred from the
    # checkpoint's own training data; a --recall-corpora override says what
    # the operator measured, not what the run trained on.
    if not args.recall_corpora:
        r["training_scope"] = corpus_names
    # One-corpus artifacts retain the v1 surface for downstream compatibility.
    if len(corpus_results) == 1:
        only = next(iter(corpus_results.values()))
        r.update({k: only[k] for k in
                  ("seed", "n_per_task", "tasks", "overall_word_acc", "examples")})
        r["poem_path"] = only["poem_path"]

    out_dir = Path(args.out) if args.out else (
        Path(args.checkpoint).parent / "eval" if args.checkpoint
        else Path("runs/base-eval-full"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tasks.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"wrote {out_dir / 'tasks.json'}")


if __name__ == "__main__":
    main()
