"""Teacher-ceiling diagnostic: greedy recitation WITH the context in prompt.

The student is distilled toward the teacher's distribution, so the teacher's
own greedy recitation quality (with context) upper-bounds what distillation
can transfer. If this CER is poor, fix the teacher prompt before blaming the
training method.

Usage: python scripts/teacher_recite.py [--limit 8] [--show 2] [--out path.json]
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
from selfupdate.eval.recite import recite_eval, teacher_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show", type=int, default=2, help="print N sample generations")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16)
    model.to(cfg.model.device)
    model.eval()

    records = load_jsonl(cfg.data.examples_path)
    summary = recite_eval(model, tok, records, limit=args.limit, prompt_fn=teacher_prompt)
    summary["model"] = cfg.model.name
    summary["examples_path"] = cfg.data.examples_path
    summary["prompt"] = "shared_prefix + privileged + shared_mid"
    summary["general"] = general_ce(model, tok, device=cfg.model.device)
    for i, r in enumerate(summary["per_example"]):
        print(f"{r['example_id']}: teacher-with-context CER {r['cer']:.3f}")
        if i < args.show:
            print(f"  GEN : {r['text'][:160]!r}")
    print(
        "mean teacher-with-context: "
        f"CER {summary['cer']:.4f} line-exact {summary['line_exact']:.4f} "
        f"prefix-lines {summary['prefix_lines']:.2f}"
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
