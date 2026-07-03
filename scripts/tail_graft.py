"""Cross-run tail graft: body blocks from one checkpoint, tail blocks from
another, free-run recitation of the chimera.

The decisive test for the storage-vs-readout decomposition (docs/
hidden_loss.md): swap a strict layerwise body against a tail-CE readout and
measure whether storage and readout transfer. This tests whether the final
window is a portable decoder or a co-adapted circuit.

Usage:
    tail_graft.py --body runs/lw_seq_0p6b_rag/checkpoint \
                  --tail runs/lw_tail_ce_e40_v2_0p6b_rag/checkpoint [--k 4] [--limit N]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--body", required=True, help="checkpoint for blocks 1..n-k")
    ap.add_argument("--tail", required=True, help="checkpoint for blocks n-k+1..n")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    tok = AutoTokenizer.from_pretrained(args.body)
    model = AutoModelForCausalLM.from_pretrained(args.body, dtype=torch.bfloat16)
    donor = AutoModelForCausalLM.from_pretrained(args.tail, dtype=torch.bfloat16)

    n = model.config.num_hidden_layers
    tail_layers = list(range(n - args.k, n))  # 0-indexed decoder blocks
    src = donor.state_dict()
    swap = {k: v for k, v in src.items()
            if any(f"layers.{L}." in k for L in tail_layers)}
    missing, unexpected = model.load_state_dict(swap, strict=False)
    assert not unexpected, unexpected
    print(f"grafted {len(swap)} tensors (blocks {tail_layers[0]+1}..{n}, 1-indexed) "
          f"from {args.tail} onto {args.body}")
    del donor, src

    model.to(cfg.model.device).eval()
    records = load_jsonl(cfg.data.examples_path)
    r = recite_eval(model, tok, records, limit=args.limit)
    print(f"chimera: n={r['n']} CER {r['cer']:.4f} line-exact {r['line_exact']:.4f} "
          f"prefix-lines {r['prefix_lines']:.2f}")

    out = Path(args.out or "runs/tail_graft")
    out.mkdir(parents=True, exist_ok=True)
    name = f"{Path(args.body).parent.name}__tail_{Path(args.tail).parent.name}_k{args.k}.json"
    (out / name).write_text(json.dumps(
        {"body": args.body, "tail": args.tail, "k": args.k, **{
            k: r[k] for k in ("cer", "line_exact", "prefix_lines", "n")}},
        ensure_ascii=False, indent=1))
    print("wrote", out / name)


if __name__ == "__main__":
    main()
