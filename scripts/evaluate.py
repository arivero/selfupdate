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
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.data.dataset import load_jsonl
from selfupdate.eval.general import general_ce
from selfupdate.eval.recite import recite_eval


def _checkpoint_run_config(checkpoint: str | None) -> dict:
    if not checkpoint:
        return {}
    ckpt = Path(checkpoint)
    candidates = [
        ckpt.parent / "config.yaml",
        ckpt / "config.yaml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            return yaml.safe_load(path.read_text()) or {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def layer_residuals(cfg, checkpoint: str, out_dir: Path,
                    limit: int | None = None) -> dict:
    """Checkpoint-time per-layer residuals against the frozen teacher —
    storage QUALITY, separated from training loss (which conflates
    optimization state with what the model stores). One teacher forward
    (teacher_ids) + one student forward (student_ids) per item; per-layer
    nmse / l2mse / vocab_mse / residual norm ratio on the aligned span.
    Writes layer_residuals.{json,csv,png} next to recite.json."""
    from selfupdate.data.dataset import DistillDataset
    from selfupdate.train.blocks import BlockStack
    from selfupdate.train.layerwise import OnlineTeacherSource
    from selfupdate.train.losses import HiddenLoss, hidden_match

    device = cfg.model.device
    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    student_m = AutoModelForCausalLM.from_pretrained(checkpoint,
                                                     dtype=torch.bfloat16)
    if (Path(checkpoint) / "adapter_config.json").exists():
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(cfg.model.name,
                                                    dtype=torch.bfloat16)
        student_m = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()
    student_m.to(device).eval().requires_grad_(False)
    teacher_m = AutoModelForCausalLM.from_pretrained(cfg.model.name,
                                                     dtype=torch.bfloat16)
    teacher_m.to(device).eval().requires_grad_(False)
    student = BlockStack(student_m)
    teacher = OnlineTeacherSource(student, frozen_stack=BlockStack(teacher_m))

    ds = DistillDataset(cfg.data.examples_path, None, tok, need_layers=[],
                        rebase_gap=(cfg.mask.compaction in ("stub_gap", "remove_gap")),
                        with_teacher_ids=True)
    n = student.n_layers
    vocab_loss = HiddenLoss("vocab_mse", student.final_norm, student.lm_head)
    sums = {m: torch.zeros(n, dtype=torch.float64)
            for m in ("nmse", "l2mse", "vocab_mse", "norm_ratio")}
    count = 0
    with torch.no_grad():
        for idx in range(len(ds) if limit is None else min(limit, len(ds))):
            it = ds[idx]
            targets = teacher.aligned_targets(it, device)
            ids = it.student_ids.to(device)[None]
            pos = it.position_ids.to(device)[None]
            with torch.autocast(device, dtype=torch.bfloat16):
                h = student.embed(ids)
                pe = student.rope(h, pos)
                for L in range(1, n + 1):
                    h = student.run_block(L, h, pe)
                    s = student.loss_view(L, h)[0, it.s0: it.s0 + it.A].float()
                    t = targets[L].float()
                    sums["nmse"][L - 1] += float(hidden_match(s, t, "nmse"))
                    sums["l2mse"][L - 1] += float(hidden_match(s, t, "l2mse"))
                    sums["vocab_mse"][L - 1] += float(
                        vocab_loss(s, t, normed=(L == n), layer=L))
                    sums["norm_ratio"][L - 1] += float(
                        (s - t).norm() / t.norm().clamp_min(1e-8))
            count += 1
    per_layer = {m: [v / count for v in sums[m].tolist()] for m in sums}
    result = {"model": cfg.model.name, "checkpoint": str(checkpoint),
              "examples_path": cfg.data.examples_path, "n_items": count,
              "n_layers": n, "per_layer": per_layer}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "layer_residuals.json").write_text(
        json.dumps(result, indent=1) + "\n")
    with (out_dir / "layer_residuals.csv").open("w") as f:
        f.write("layer," + ",".join(per_layer) + "\n")
        for L in range(n):
            f.write(f"{L + 1}," + ",".join(f"{per_layer[m][L]:.6g}"
                                           for m in per_layer) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        xs = range(1, n + 1)
        for m in ("nmse", "l2mse", "vocab_mse", "norm_ratio"):
            ax.plot(xs, per_layer[m], marker=".", label=m)
        ax.set_yscale("log")
        ax.set_xlabel("layer")
        ax.set_ylabel("residual (log)")
        ax.set_title(f"checkpoint residuals vs teacher — {Path(checkpoint).parent.name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "layer_residuals.png", dpi=120)
        plt.close(fig)
    except ImportError:
        pass
    print(f"wrote {out_dir / 'layer_residuals.json'} (n={count})")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--layer-residuals", action="store_true",
                    help="checkpoint-time per-layer residuals vs the frozen "
                         "teacher (storage quality); writes layer_residuals.* "
                         "and skips the recitation eval")
    ap.add_argument("--base", action="store_true", help="evaluate the untrained base model")
    ap.add_argument("--out", default=None,
                    help="output dir override (multi-node: concurrent --base "
                         "evals must not share runs/base-eval-full)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-per-task", type=int, default=24,
                    help="items per task in the three-task battery")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="batched generation for standard recitation evals")
    ap.add_argument("--max-extra-tokens", type=int, default=48)
    ap.add_argument("--bucket-by-length", action="store_true",
                    help="throughput mode: group examples by reference length")
    ap.add_argument("--score-workers", type=int, default=None,
                    help="CPU workers for CER scoring in batched eval")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="fixed random order for batched eval; results are restored by example index")
    ap.add_argument("--auto-map", action="store_true",
                    help="load with device_map=auto (multi-card eval, e.g. 32B)")
    ap.add_argument("--load-4bit", action="store_true",
                    help="load the base in NF4 4-bit (bitsandbytes) — lets a 40B "
                         "eval coexist with a resident 40B training job on the same "
                         "cards. Perturbs CER vs bf16; label results as 4-bit. "
                         "Implies device_map=auto and skips the .to(device) move "
                         "(bnb 4-bit tensors cannot be re-placed).")
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)
    checkpoint_cfg = _checkpoint_run_config(args.checkpoint)
    checkpoint_model = ((checkpoint_cfg.get("model") or {}).get("name")
                        if checkpoint_cfg else None)
    if checkpoint_model and not args.base:
        cfg.model.name = checkpoint_model

    src = cfg.model.name if args.base else args.checkpoint
    if not src:
        sys.exit("pass --checkpoint or --base")
    if args.layer_residuals:
        if args.base:
            sys.exit("--layer-residuals compares a checkpoint to the teacher")
        out_dir = Path(args.out) if args.out else Path(args.checkpoint).parent / "eval"
        layer_residuals(cfg, args.checkpoint, out_dir, limit=args.limit)
        return
    # 4-bit forces device_map=auto (bnb places shards itself); the .to(device)
    # move below is skipped because bnb 4-bit params cannot be re-placed.
    load_kw: dict = {}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    dev_map = "auto" if (args.auto_map or args.load_4bit) else None
    if not args.base and (Path(src) / "adapter_config.json").exists():
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(src)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, dtype=torch.bfloat16, device_map=dev_map, **load_kw)
        model = PeftModel.from_pretrained(base, src)
    else:
        tok = AutoTokenizer.from_pretrained(src)
        model = AutoModelForCausalLM.from_pretrained(
            src, dtype=torch.bfloat16, device_map=dev_map, **load_kw)
    if dev_map is None:
        model.to(cfg.model.device)
    model.eval()

    # the three-task battery (owner directive 2026-07-10): next / prev /
    # cloze with plain accuracies — CER and the other recovery metrics are
    # retired from the active eval surface
    from selfupdate.eval.tasks import tasks_eval

    r = tasks_eval(model, tok, cfg.data.poem_path,
                   n_per_task=args.n_per_task)
    r["teacher_reference_kind"] = "teacher_epoch0_native_no_rag" if args.base else "checkpoint"
    r["model"] = cfg.model.name
    r["poem_path"] = cfg.data.poem_path
    r["general"] = general_ce(model, tok, device=cfg.model.device)
    parts = "  ".join(f"{t}: exact {v['exact']:.2f} words {v['word_acc']:.2f}"
                      for t, v in r["tasks"].items())
    print(f"{parts}  general-CE {r['general']['mean_ce']:.3f}")

    out_dir = Path(args.out) if args.out else (
        Path(args.checkpoint).parent / "eval" if args.checkpoint
        else Path("runs/base-eval-full"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tasks.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"wrote {out_dir / 'tasks.json'}")


if __name__ == "__main__":
    main()
