"""Full recitation eval of a trained checkpoint (or the base model as control).

Usage:
    python scripts/evaluate.py --checkpoint runs/<name>/checkpoint [--limit N]
    python scripts/evaluate.py --base   # untrained control
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.data.dataset import load_jsonl
from selfupdate.eval.general import general_ce
from selfupdate.eval.recite import recite_eval


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
    ap.add_argument("--batch-size", type=int, default=1,
                    help="batched generation for standard recitation evals")
    ap.add_argument("--max-extra-tokens", type=int, default=48)
    ap.add_argument("--bucket-by-length", action="store_true",
                    help="throughput mode: group examples by reference length")
    ap.add_argument("--score-workers", type=int, default=None,
                    help="CPU workers for CER scoring in batched eval")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="fixed random order for batched eval; results are restored by example index")
    ap.add_argument("--auto-map", action="store_true",
                    help="load with device_map=auto (multi-card eval, e.g. 32B)")
    ap.add_argument("--load-4bit", action="store_true",
                    help="load the base in NF4 4-bit (bitsandbytes) — lets a 40B "
                         "eval coexist with a resident 40B training job on the same "
                         "cards. Perturbs CER vs bf16; label results as 4-bit. "
                         "Implies device_map=auto and skips the .to(device) move "
                         "(bnb 4-bit tensors cannot be re-placed).")
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)
    checkpoint_cfg = _checkpoint_run_config(args.checkpoint)
    checkpoint_model = ((checkpoint_cfg.get("model") or {}).get("name")
                        if checkpoint_cfg else None)
    if checkpoint_model and not args.base:
        cfg.model.name = checkpoint_model

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

    records = load_jsonl(cfg.data.examples_path)
    r = recite_eval(model, tok, records, limit=args.limit,
                    rebase_gap=(cfg.mask.compaction in ("stub_gap", "remove_gap")),
                    batch_size=args.batch_size,
                    max_extra_tokens=args.max_extra_tokens,
                    bucket_by_length=args.bucket_by_length,
                    score_workers=args.score_workers,
                    shuffle_seed=args.shuffle_seed)
    r["batch_size"] = args.batch_size
    r["max_extra_tokens"] = args.max_extra_tokens
    r["bucket_by_length"] = args.bucket_by_length
    r["score_workers"] = args.score_workers
    r["shuffle_seed"] = args.shuffle_seed
    r["teacher_reference_kind"] = "teacher_epoch0_native_no_rag" if args.base else "checkpoint"
    r["model"] = cfg.model.name
    r["examples_path"] = cfg.data.examples_path
    r["general"] = general_ce(model, tok, device=cfg.model.device)
    print(f"n={r['n']}  CER {r['cer']:.4f}  line-exact {r['line_exact']:.4f}  "
          f"prefix-lines {r['prefix_lines']:.2f}  general-CE {r['general']['mean_ce']:.3f}")

    out_dir = Path(args.out) if args.out else (
        Path(args.checkpoint).parent / "eval" if args.checkpoint
        else Path("runs/base-eval-full"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recite.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"wrote {out_dir / 'recite.json'}")


if __name__ == "__main__":
    main()
