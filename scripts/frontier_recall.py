"""Epoch-0 recall of a too-large-to-train frontier release (experimental lane).

Usage:
    python scripts/frontier_recall.py --model deepseek-ai/DeepSeek-V4-Flash \
        --examples data/poem/examples_v4.jsonl --out runs/recall_dsv4flash \
        --trust-remote-code

Loads the model at its native (possibly mixed-bit) width and reports how much
of the poem it already reproduces. No training. See
src/selfupdate/experimental/ for why exotic loading lives outside the trainer.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.experimental.frontier_recall import epoch0_recall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--examples", default="data/poem/examples_v4.jsonl")
    ap.add_argument("--out", default=None, help="output dir for recall.json")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-extra-tokens", type=int, default=48)
    ap.add_argument("--score-workers", type=int, default=None)
    ap.add_argument("--shuffle-seed", type=int, default=None)
    args = ap.parse_args()

    r = epoch0_recall(
        args.model, args.examples,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        limit=args.limit, batch_size=args.batch_size,
        max_extra_tokens=args.max_extra_tokens,
        score_workers=args.score_workers, shuffle_seed=args.shuffle_seed,
        out_dir=args.out)
    print(f"model={args.model}\n"
          f"n={r['n']}  CER {r['cer']:.4f}  line-exact {r['line_exact']:.4f}  "
          f"prefix-lines {r['prefix_lines']:.2f}"
          + (f"  quant={r['quantization']}" if 'quantization' in r else ""))
    if args.out:
        print(f"wrote {Path(args.out) / 'recall.json'}")


if __name__ == "__main__":
    main()
