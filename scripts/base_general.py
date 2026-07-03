"""General-CE of a base model (forgetting baseline). Usage: base_general.py <model> <out.json>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.eval.general import general_ce

model_name, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_name)
m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda").eval()
json.dump({"model": model_name, **general_ce(m, tok)}, open(out, "w"))
print("wrote", out)
