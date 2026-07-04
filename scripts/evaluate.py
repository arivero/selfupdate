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


def _load_peft_checkpoint(base, checkpoint: Path):
    from peft import PeftConfig, PeftModel, get_peft_model
    from safetensors.torch import load_file

    try:
        return PeftModel.from_pretrained(base, checkpoint)
    except TypeError as e:
        if "WeightConverter.__init__" not in str(e):
            raise
        print(f"PEFT adapter conversion failed ({e}); falling back to direct state load")

    peft_cfg = PeftConfig.from_pretrained(checkpoint)
    model = get_peft_model(base, peft_cfg)
    adapter_path = checkpoint / "adapter_model.safetensors"
    state = load_file(str(adapter_path), device="cpu")
    expanded = dict(state)
    for k, v in state.items():
        if ".lora_A.weight" in k:
            expanded[k.replace(".lora_A.weight", ".lora_A.default.weight")] = v
        elif ".lora_B.weight" in k:
            expanded[k.replace(".lora_B.weight", ".lora_B.default.weight")] = v
    missing, unexpected = model.load_state_dict(expanded, strict=False)
    if unexpected:
        print(f"direct adapter load ignored {len(unexpected)} unexpected tensors")
    loaded = len(expanded) - len(unexpected)
    if loaded <= 0:
        raise RuntimeError(
            f"direct adapter load failed for {checkpoint}: "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
    return model


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
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    src = cfg.model.name if args.base else args.checkpoint
    if not src:
        sys.exit("pass --checkpoint or --base")
    if not args.base and (Path(src) / "adapter_config.json").exists():
        tok = AutoTokenizer.from_pretrained(src)
        base = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16)
        model = _load_peft_checkpoint(base, Path(src))
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

    out_path = Path(args.out) if args.out else None
    if out_path and out_path.suffix == ".json":
        out_file = out_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = out_path if out_path else (
            Path(args.checkpoint).parent / "eval" if args.checkpoint
            else Path("runs/base-eval-full"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "recite.json"
    out_file.write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
