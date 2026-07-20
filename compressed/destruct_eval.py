"""Destruction battery for a checkpoint (or base control).

Usage:
    python compressed/destruct_eval.py --experiment configs/experiments/X.yaml \
        --checkpoint runs/X/checkpoint [--out runs/X/eval]
    python compressed/destruct_eval.py --experiment ... --base --out runs/destruction/base_0p6b

Writes <out>/destruction.json (default: <checkpoint>/../eval/). If
--base-ref points at a base destruction.json, the pre-committed verdict
flags are computed and embedded.
"""


# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP



import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # cold cache must fail loudly


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.eval.destruction import (SCHEMA_VERSION, benchmark_ce_ranking,
                                         degeneration_stats,
                                         intrusion_generation, probe_battery,
                                         verdict)


def corpus_lines(poem_path: str) -> list[str]:
    """Content lines of a corpus file (verse or prose format): markers and
    blanks are structure, not memorized text."""
    lines = Path(poem_path).read_text(encoding="utf-8").splitlines()
    return [l for l in lines if l.strip() and not l.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--base-ref", default=None,
                    help="base destruction.json for verdict flags")
    ap.add_argument("--skip-benchmarks", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="complete battery (default: fast fixed subsample — "
                         "50 evenly-spaced intrusion prompts, bench n=100)")
    ap.add_argument("--intr-n", type=int, default=50,
                    help="intrusion prompts in the fast subsample")
    ap.add_argument("--bench-n", type=int, default=200)
    ap.add_argument("--benches", default=None,
                    help="comma list from BENCH_REGISTRY (default: standard suite)")
    ap.add_argument("--intr-batch", type=int, default=1,
                    help="prompts per generation batch; 1 = historical exact "
                         "path. Compare runs judged at the SAME setting.")
    ap.add_argument("--bench-batch", type=int, default=1,
                    help="option sequences per scoring forward; 1 = historical "
                         "exact path. Compare runs judged at the SAME setting.")
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

    prompts = [l for l in Path("data/intrusion_prompts_es.txt")
               .read_text(encoding="utf-8").splitlines() if l.strip()]
    # fast fixed subsample (owner directive 2026-07-10): the standard battery
    # on a DETERMINISTIC subset — same prompts every run, comparable across
    # checkpoints. --full restores the complete battery.
    if not args.full:
        prompts = prompts[::len(prompts) // args.intr_n or 1][:args.intr_n]
        args.bench_n = min(args.bench_n, 100)

    dest = {"schema_version": SCHEMA_VERSION,
            "source": src, "model": cfg.model.name,
            "corpus": cfg.data.poem_path,
            "fast_subsample": not args.full,
            "intrusion_n_prompts": len(prompts)}
    dest["probe_battery"] = probe_battery(model, tok, cfg.model.device)
    print(f"probes: overall {dest['probe_battery']['overall_mean_ce']:.3f}  "
          f"legacy {dest['probe_battery']['legacy_mean_ce']:.3f}")
    dest["eval_batching"] = {"intr_batch": args.intr_batch,
                             "bench_batch": args.bench_batch}
    intr = intrusion_generation(model, tok, prompts,
                                corpus_lines(cfg.data.poem_path),
                                cfg.model.device,
                                batch_size=args.intr_batch)
    dest["degeneration"] = degeneration_stats(intr.pop("generations"))
    dest["intrusion"] = intr
    print(f"intrusion: {intr['hit_rate']:.2%} ({len(intr['hits'])} hits)  "
          f"rep4 {dest['degeneration']['max_rep4_run_mean']:.2f}  "
          f"distinct2 {dest['degeneration']['distinct2_mean']:.3f}")
    if not args.skip_benchmarks:
        kw = {}
        if args.benches:
            kw["benches"] = tuple(args.benches.split(","))
        dest["benchmarks"] = benchmark_ce_ranking(model, tok, cfg.model.device,
                                                  n=args.bench_n,
                                                  micro_batch=args.bench_batch,
                                                  **kw)
        for b, r in dest["benchmarks"].items():
            print(f"{b}: {r['accuracy']:.3f} (n={r['n']})")
    if args.base_ref:
        base_dest = json.loads(Path(args.base_ref).read_text())
        dest["verdict"] = verdict(dest, base_dest)
        print("verdict:", "DESTRUCTIVE" if dest["verdict"]["destructive"]
              else "clean")

    out_dir = Path(args.out) if args.out else (
        Path(args.checkpoint).parent / "eval" if args.checkpoint
        else Path("runs/destruction/base"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "destruction.json").write_text(
        json.dumps(dest, ensure_ascii=False, indent=1))
    print(f"wrote {out_dir / 'destruction.json'}")


if __name__ == "__main__":
    main()
