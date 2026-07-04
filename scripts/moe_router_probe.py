"""MoE router probe: do distinct experts fire for novel content vs syntax?

Evolving-person instrument #2 (user: "MoE models could help here"). Runs
gpt-oss-20b (24 layers, 32 experts, top-4 routing) no-grad over four text
conditions and compares per-layer expert-usage distributions:

- poem        — the memorization target (novel content to absorb)
- poetry_es   — same genre, NOT memorized (genre control)
- prose_es    — plain Spanish (register control)
- prose_en    — English (language control)

If the poem's routing profile separates from the controls, routing is
both a "worth of attention" scoring signal AND a parameter-selection
mechanism (consolidate into the experts that fired). If poem ≈ poetry_es
≠ prose, routing tracks GENRE, not novelty — still useful for selecting
which experts carry a register, but not a novelty detector.

Exclusive lane: dequantized MXFP4 → bf16 needs ~40 GB.

Usage: moe_router_probe.py [--model openai/gpt-oss-20b] [--out runs/moe_router_probe]
"""

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.data.poem import load_poem
from selfupdate.eval.probes import PROBE_SETS


def condition_texts():
    verses = [v.text for v in load_poem("data/poem/raw.txt")]
    return {
        "poem": " ".join(verses[:120]),
        "poetry_es": " ".join(PROBE_SETS["poetry_es"][1:]),
        "prose_es": " ".join(PROBE_SETS["prose_es"]),
        "prose_en": " ".join(PROBE_SETS["prose_en"][1:]),
    }


@torch.no_grad()
def routing_histograms(model, tok, text, device, max_tokens=768):
    """Per-layer expert-usage histograms. output_router_logits=True is
    UNUSABLE here: the hub-kernels fused-MoE path never calls the Python
    router module, the recorder stays empty, and the aux-loss branch then
    crashes on the empty tuple. Instead: capture each MLP's INPUT with a
    pre-hook and run the router weights manually — kernel-agnostic."""
    ids = tok.encode(text, add_special_tokens=False)[:max_tokens]
    t = torch.tensor([ids], device=device)
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def make_hook(i):
            def pre(module, args, kwargs):
                h = args[0] if args else kwargs["hidden_states"]
                captured[i] = h.detach()
            return pre
        hooks.append(layer.mlp.register_forward_pre_hook(
            make_hook(i), with_kwargs=True))
    model(t, use_cache=False)
    for h in hooks:
        h.remove()
    k = model.config.num_experts_per_tok
    hists = []
    for i in sorted(captured):
        router = model.model.layers[i].mlp.router
        flat = captured[i].reshape(-1, captured[i].shape[-1]).float()
        rl = F.linear(flat, router.weight.float(), router.bias.float())
        top = rl.topk(k, dim=-1).indices  # [T, k]
        h = torch.zeros(rl.shape[-1], dtype=torch.float64)
        for e in top.flatten().tolist():
            h[e] += 1
        hists.append(h / h.sum())
    return hists


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: float((a * (a.clamp_min(1e-12) / b.clamp_min(1e-12)).log()).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--out", default="runs/moe_router_probe")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(args.device).eval()

    conds = condition_texts()
    hists = {name: routing_histograms(model, tok, text, args.device)
             for name, text in conds.items()}

    n_layers = len(next(iter(hists.values())))
    rows = []
    for L in range(n_layers):
        row = {"layer": L + 1}
        for a, b in itertools.combinations(conds, 2):
            row[f"js_{a}__{b}"] = js(hists[a][L], hists[b][L])
        for name in conds:
            h = hists[name][L]
            row[f"top1_{name}"] = int(h.argmax())
            row[f"entropy_{name}"] = float(-(h.clamp_min(1e-12) * h.clamp_min(1e-12).log()).sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "router.csv", index=False)

    pairs = [c for c in df.columns if c.startswith("js_")]
    print("mean JS divergence across layers:")
    for c in sorted(pairs, key=lambda c: -df[c].mean()):
        print(f"  {c[3:]:28s} {df[c].mean():.4f}  (max {df[c].max():.4f} @L{int(df.loc[df[c].idxmax(), 'layer'])})")
    novel = df["js_poem__poetry_es"].mean()
    genre = df["js_poetry_es__prose_es"].mean()
    lang = df["js_prose_es__prose_en"].mean()
    print(f"\nnovelty signal (poem vs same-genre) {novel:.4f} | genre {genre:.4f} | language {lang:.4f}")
    print("verdict:", "routing separates NOVEL content from its own genre"
          if novel > 0.5 * genre else
          "routing tracks genre/register, not novelty — expert selection is a register instrument")
    print(f"wrote {out / 'router.csv'}")


if __name__ == "__main__":
    main()
