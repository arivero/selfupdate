"""Teacher ceiling: base model recall WITH the RAG passage in context.

The copying ceiling for every student arm: prompt = shared_prefix +
privileged + shared_mid (the exact teacher view), greedy, full recite
metrics. Implemented by setting student_stub := privileged, which makes
recite_eval's student prompt identical to the teacher prompt.

Usage: teacher_ceiling.py --experiment configs/experiments/X.yaml [--limit N] [--out ...]
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
from selfupdate.eval.recite import recite_eval

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/base.yaml")
ap.add_argument("--experiment", required=True)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()
cfg = load_config(args.config, args.experiment)

tok = AutoTokenizer.from_pretrained(cfg.model.name)
model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16)
model.to(cfg.model.device).eval()

records = [{**r, "student_stub": r.get("privileged", ""), "interleaved": None}
           for r in load_jsonl(cfg.data.examples_path)]
r = recite_eval(model, tok, records, limit=args.limit)
print(f"TEACHER CEILING {cfg.run_name}: n={r['n']} cer {r['cer']:.4f} "
      f"cer_flat {r['cer_flat']:.4f} line_exact {r['line_exact']:.4f}")
out = Path(args.out or f"runs/teacher_ceiling_{cfg.run_name}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(r, ensure_ascii=False, indent=1))
