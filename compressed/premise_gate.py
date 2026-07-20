"""Premise gate: exit 0 iff the base model does NOT already recite the corpus.

Usage: premise_gate.py <recite.json> <min_cer>

Reads a recite.json produced by `evaluate.py --base` and passes (exit 0) only
if base CER > min_cer, i.e. the corpus is genuinely absent from the model.
Queue items touch a .premise_*_ok marker on pass; training items depend on it,
so a model that already knows the poem never gets GPU time (CLAUDE.md premise
check, cache-free variant for big models).
"""
import json
import sys

r = json.load(open(sys.argv[1]))
min_cer = float(sys.argv[2])
print(f"premise gate: base CER {r['cer']:.3f} line-exact {r['line_exact']:.3f} "
      f"(pass iff CER > {min_cer})")
sys.exit(0 if r["cer"] > min_cer else 1)
