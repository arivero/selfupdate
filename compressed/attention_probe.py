"""Head taxonomy: which attention heads are grammar, which carry content?

Evolving-person instrument (docs/evolving_person.md): the base model runs
the TEACHER view (privileged passage present) with eager attention, and
each layer x head is scored at the ANSWER positions:

- priv_mass:  attention mass landing on the privileged block (retrieval)
- entropy:    attention entropy (peaked = specialized, flat = mixing)
- distance:   mean token distance of attention (local = syntax-like)

Heads that are long-range + privileged-heavy are the "content/insight"
heads whose attention scores could rank what a conversation considered
worth absorbing; local, privileged-blind, low-entropy heads are
"grammar". NOT run_block (SDPA-locked): a plain HF forward with
attn_implementation="eager".

Usage: attention_probe.py [--model Qwen/Qwen3-0.6B] [--n 16]
       [--examples data/poem/examples_v4.jsonl] [--out runs/attention_probe_0.6B]
"""


# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP



import argparse
import sys
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.data.dataset import load_jsonl
from selfupdate.masking import ContextMasker, SegmentedExample


@torch.no_grad()
def probe(model, pairs, device):
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    acc = torch.zeros(n_layers, n_heads, 3, dtype=torch.float64)  # mass, ent, dist
    n_rows = 0
    for pair in pairs:
        ids = torch.tensor([pair.teacher_ids], device=device)
        out = model(ids, output_attentions=True, use_cache=False)
        priv = slice(pair.s_aligned.start, pair.t_aligned.start)  # teacher coords
        ans = pair.t_answer
        pos = torch.arange(ids.shape[1], device=device, dtype=torch.float32)
        for L, att in enumerate(out.attentions):  # [1, H, T, T]
            a = att[0, :, ans, :].float()  # [H, A, T] rows = answer queries
            mass = a[:, :, priv].sum(-1).mean(-1)  # [H]
            ent = -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(-1).mean(-1)
            q = pos[ans.start: ans.stop][None, :, None]
            dist = (a * (q - pos[None, None, :]).abs()).sum(-1).mean(-1)
            acc[L, :, 0] += mass.cpu().double()
            acc[L, :, 1] += ent.cpu().double()
            acc[L, :, 2] += dist.cpu().double()
        n_rows += 1
        del out
        torch.cuda.empty_cache()
    return acc / n_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--examples", default="data/poem/examples_v4.jsonl")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out or f"runs/attention_probe_{args.model.split('/')[-1].replace('Qwen3-', '')}")
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager")
    model.to(args.device).eval()

    masker = ContextMasker(tok)
    records = [r for r in load_jsonl(args.examples)
               if r.get("privileged")][: args.n]
    pairs = [masker.build(SegmentedExample.from_record(r)) for r in records]
    stats = probe(model, pairs, args.device)

    n_layers, n_heads, _ = stats.shape
    rows = [{"layer": L + 1, "head": h,
             "priv_mass": stats[L, h, 0].item(),
             "entropy": stats[L, h, 1].item(),
             "distance": stats[L, h, 2].item()}
            for L in range(n_layers) for h in range(n_heads)]
    df = pd.DataFrame(rows)
    # taxonomy: retrieval is the defining property of a content head. The
    # distance axis is confounded by attention sinks (a head staring at
    # token 0 from the answer scores ~700 "distance"), so it only DEFINES
    # grammar (local + privileged-blind), never content.
    qm_hi, qm_lo = df.priv_mass.quantile(0.75), df.priv_mass.quantile(0.25)
    qd_lo = df.distance.quantile(0.25)
    df["kind"] = "mixed"
    df.loc[df.priv_mass >= qm_hi, "kind"] = "content"
    df.loc[(df.priv_mass <= qm_lo) & (df.distance <= qd_lo), "kind"] = "grammar"
    df.to_csv(out / "heads.csv", index=False)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"content": "#0072B2", "grammar": "#D55E00", "mixed": "#BBBBBB"}
    for kind, g in df.groupby("kind"):
        a.scatter(g.distance, g.priv_mass, s=14, alpha=0.7,
                  c=colors[kind], label=f"{kind} ({len(g)})")
    a.set_xlabel("mean attention distance (tokens)")
    a.set_ylabel("answer→privileged attention mass")
    a.legend(fontsize=8, frameon=False)
    a.set_title("(a) head taxonomy at answer positions", fontsize=9, loc="left")
    prof = df.groupby("layer").priv_mass.mean()
    b.plot(prof.index, prof.values, marker="o", ms=3, color="#0072B2")
    b.axvline(7, color="#7F7F7F", lw=0.8, ls="--")
    b.text(7.3, prof.max() * 0.95, "L7 integration peak\n(teacher_censored)", fontsize=7)
    b.set_xlabel("layer")
    b.set_ylabel("mean answer→privileged mass")
    b.set_title("(b) where retrieval attention lives", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out / "heads.png", dpi=150)

    top = df.nlargest(10, "priv_mass")[["layer", "head", "priv_mass", "distance", "entropy"]]
    print(df.kind.value_counts().to_string())
    print("\ntop retrieval heads:\n", top.to_string(index=False))
    print(f"\nwrote {out}/heads.csv and heads.png")


if __name__ == "__main__":
    main()
