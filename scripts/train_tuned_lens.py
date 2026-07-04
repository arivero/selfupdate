"""Train tuned-lens translators for a (frozen) base model.

Usage:
    python scripts/train_tuned_lens.py [--model Qwen/Qwen3-0.6B]
        [--out runs/tuned_lens_0.6B] [--target-tokens 3000000]

Text: openwebtext from the HF cache (neutral English web text — the lens
must calibrate the model's GENERIC readout, not anything corpus-adjacent).
Writes translators.safetensors + meta.json + metrics.jsonl.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.train.tuned_lens import (lens_kl_step, make_translators,
                                         save_translators)


def vocab_sig(model) -> tuple:
    sig = []
    for p in (model.model.embed_tokens.weight, model.model.norm.weight,
              model.lm_head.weight):
        s = 0.0
        for chunk in p.detach().reshape(-1).split(1 << 22):
            s += chunk.double().sum().item()
        sig.append(s)
    return tuple(sig)


def batches(tok, seq_len, batch, target_tokens, device):
    from datasets import load_dataset

    ds = load_dataset("openwebtext", split="train")
    buf, seqs, seen = [], [], 0
    for row in ds:
        buf.extend(tok.encode(row["text"], add_special_tokens=False))
        while len(buf) >= seq_len:
            seqs.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(seqs) == batch:
                yield torch.tensor(seqs, device=device)
                seen += batch * seq_len
                seqs = []
                if seen >= target_tokens:
                    return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target-tokens", type=int, default=3_000_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out or f"runs/tuned_lens_{args.model.split('/')[-1].replace('Qwen3-', '')}")
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(args.device).eval()
    model.requires_grad_(False)  # frozen-vocabulary principle: the model,
    # including norm+head the lens decodes through, receives NO gradient
    sig0 = vocab_sig(model)

    n_layers = model.config.num_hidden_layers
    translators = make_translators(model.config.hidden_size, n_layers, args.device)
    opt = torch.optim.AdamW(translators.parameters(), lr=args.lr)

    log = open(out / "metrics.jsonl", "a")
    t0 = time.time()
    step = tokens = 0
    for ids in batches(tok, args.seq_len, args.batch, args.target_tokens, args.device):
        # lens_kl_step consumes [T] rows; feed the batch as one long row set
        per_layer_acc = {}
        for row in ids:
            pl = lens_kl_step(model, translators, row[None])
            for L, v in pl.items():
                per_layer_acc[L] = per_layer_acc.get(L, 0.0) + v / len(ids)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens += ids.numel()
        if step % 20 == 0:
            mean_kl = sum(per_layer_acc.values()) / len(per_layer_acc)
            log.write(json.dumps({"step": step, "tokens": tokens,
                                  "mean_kl": mean_kl,
                                  "per_layer": per_layer_acc,
                                  "t": time.time() - t0}) + "\n")
            log.flush()
            print(f"step {step}  tokens {tokens}  mean KL {mean_kl:.4f}")

    assert vocab_sig(model) == sig0, "vocabulary stack changed during lens training"
    save_translators(translators, out / "translators.safetensors",
                     meta={"model": args.model, "tokens": tokens,
                           "seq_len": args.seq_len, "lr": args.lr,
                           "final_per_layer_kl": per_layer_acc})
    print(f"wrote {out / 'translators.safetensors'} ({tokens} tokens, "
          f"{time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
