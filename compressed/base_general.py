"""General CE of an epoch-zero teacher. Usage: base_general.py <model> <out.json>"""


# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP


import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.eval.general import general_ce

model_name, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_name)
m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda").eval()
json.dump({"model": model_name, **general_ce(m, tok)}, open(out, "w"))
print("wrote", out)
