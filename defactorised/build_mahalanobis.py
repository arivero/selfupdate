"""Build a frozen per-layer diagonal Mahalanobis metric on generic text.

The diagonal shrinkage estimator is intentionally conservative: it is full
rank even when calibration positions are fewer than hidden width, records its
condition numbers, and never estimates covariance from a training minibatch.
"""

import argparse
import json
import sys
from pathlib import Path


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def texts(paths):
    out = []
    for path in map(Path, paths):
        raw = path.read_text(encoding="utf-8")
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            out.extend(x.strip() for x in raw.split("\n\n") if x.strip())
            continue
        rows = (obj if isinstance(obj, list) else
                obj.get("examples", obj.get("data", obj.get("items", []))))
        for row in rows:
            if isinstance(row, str):
                out.append(row)
            elif isinstance(row, dict):
                parts = []
                for key in ("prompt", "question", "query", "ctx", "ctx_a", "ctx_b", "activity_label"):
                    if row.get(key):
                        parts.append(str(row[key]))
                endings = row.get("endings") or row.get("choices")
                if isinstance(endings, list):
                    parts.extend(str(x.get("text", x)) if isinstance(x, dict) else str(x)
                                 for x in endings)
                if parts:
                    out.append("\n".join(parts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--texts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-prompts", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--shrinkage", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=100.0,
                    help="maximum inverse-variance ratio within a layer")
    args = ap.parse_args()
    corpus = texts(args.texts)[:args.max_prompts]
    if not corpus:
        raise ValueError("calibration corpus is empty")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(args.device).eval().requires_grad_(False)
    n = model.config.num_hidden_layers
    sums = [None] * n
    sums2 = [None] * n
    count = 0
    with torch.no_grad():
        for text in corpus:
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=args.max_tokens).input_ids.to(args.device)
            if ids.shape[1] < 2:
                continue
            hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states[1:]
            for i, h in enumerate(hs):
                x = h[0].float()
                s, s2 = x.sum(0).cpu(), x.square().sum(0).cpu()
                sums[i] = s if sums[i] is None else sums[i] + s
                sums2[i] = s2 if sums2[i] is None else sums2[i] + s2
            count += ids.shape[1]
    if count < 2:
        raise ValueError("calibration corpus produced fewer than two tokens")
    precision, conditions = {}, {}
    for i, (s, s2) in enumerate(zip(sums, sums2), 1):
        var = (s2 / count - (s / count).square()).clamp_min(0)
        mean_var = var.mean().clamp_min(1e-12)
        var = (1 - args.shrinkage) * var + args.shrinkage * mean_var
        inv = var.reciprocal()
        inv = inv.clamp(max=inv.min() * args.clip)
        precision[i] = torch.diag(inv)
        conditions[i] = float((inv.max() / inv.min()).item())
    artifact = {
        "schema": "selfupdate.mahalanobis.diagonal.v1",
        "model": args.model,
        "n_prompts": len(corpus),
        "n_tokens": count,
        "shrinkage": args.shrinkage,
        "inverse_ratio_clip": args.clip,
        "condition_numbers": conditions,
        "precision": precision,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out)
    out.with_suffix(".json").write_text(json.dumps({k: v for k, v in artifact.items()
                                                    if k != "precision"}, indent=2))
    print(f"wrote {out}: {len(corpus)} prompts, {count} tokens, {n} layers")


if __name__ == "__main__":
    main()
