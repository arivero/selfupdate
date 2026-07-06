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
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.data.dataset import load_jsonl
from selfupdate.eval.general import general_ce
from selfupdate.eval.recite import recite_eval


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
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    src = cfg.model.name if args.base else args.checkpoint
    if not src:
        sys.exit("pass --checkpoint or --base")
    if not args.base and (Path(src) / "adapter_config.json").exists():
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(src)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, dtype=torch.bfloat16,
            device_map="auto" if args.auto_map else None)
        model = PeftModel.from_pretrained(base, src)
    else:
        tok = AutoTokenizer.from_pretrained(src)
        model = AutoModelForCausalLM.from_pretrained(
            src, dtype=torch.bfloat16, device_map="auto" if args.auto_map else None)
    if not args.auto_map:
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
