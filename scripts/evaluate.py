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
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    src = cfg.model.name if args.base else args.checkpoint
    if not src:
        sys.exit("pass --checkpoint or --base")
    if not args.base and (Path(src) / "adapter_config.json").exists():
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(src)
        base = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, src)
    else:
        tok = AutoTokenizer.from_pretrained(src)
        model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    model.to(cfg.model.device)
    model.eval()

    records = load_jsonl(cfg.data.examples_path)
    r = recite_eval(model, tok, records, limit=args.limit,
                    rebase_gap=(cfg.mask.compaction in ("stub_gap", "remove_gap")))
    r["general"] = general_ce(model, tok, device=cfg.model.device)
    print(f"n={r['n']}  CER {r['cer']:.4f}  line-exact {r['line_exact']:.4f}  "
          f"prefix-lines {r['prefix_lines']:.2f}  general-CE {r['general']['mean_ce']:.3f}")

    out_dir = Path(args.checkpoint).parent / "eval" if args.checkpoint else Path("runs/base-eval-full")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recite.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"wrote {out_dir / 'recite.json'}")


if __name__ == "__main__":
    main()
