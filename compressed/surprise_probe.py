"""Surprise decomposition: is a high-surprise token a knowledge gap or a
misdirected-attention error?

Third factor of "what to memorize" (after the head taxonomy). Surprise is the
gap between what the student predicts and what actually happens. A large
surprise can mean two very different things, and they call for different fixes:

  knowledge gap      the teacher resolves the token using its privileged/RAG
                     context that the student lacks -> the thing TO MEMORIZE.
  attention misdir.  the answer is present in the shared (uncensored) context,
                     but the student attends to the wrong tokens -> a ROUTING
                     error, not a memory target (cf. blackbox vs router_aligned
                     expert selection).

To disambiguate we look at WHERE THE TEACHER ATTENDS while producing each answer
token: mass on the privileged block => knowledge gap; mass on in-context
uncensored tokens => the answer was reachable and the student simply mis-routed.

Instrument (evolving-person style, docs/evolving_person.md): the BASE model runs
both views with eager attention.
  student view  = privileged block removed (what the student must reproduce)
  teacher view  = privileged block present (the reference behaviour)
Per aligned answer token:
  s_nll   student-view NLL(reference)
  t_nll   teacher-view NLL(reference)
  excess  s_nll - t_nll        (surprise the privileged context resolves)
  t_priv  teacher attention mass on the privileged block at that query
  t_ctx   teacher attention mass on prior in-context (uncensored) tokens
  top_ctx the single in-context token the teacher attends to most

High-excess tokens are labelled knowledge_gap (t_priv >= t_ctx) or
misdirection (t_ctx > t_priv).  Outputs tokens.csv + a decomposition figure.

Usage: surprise_probe.py [--model Qwen/Qwen3-0.6B] [--n 16]
       [--examples data/poem/examples_v4.jsonl] [--layers mid]
"""

from __future__ import annotations


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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.data.dataset import load_jsonl
from selfupdate.masking import ContextMasker, SegmentedExample


def _layer_window(n_layers: int, spec: str) -> range:
    if spec == "all":
        return range(n_layers)
    if spec == "mid":  # middle third — where retrieval attention concentrates
        return range(n_layers // 3, 2 * n_layers // 3 + 1)
    lo, hi = (int(x) for x in spec.split(":"))
    return range(max(0, lo), min(n_layers, hi))


def _content_heads(model_name: str, n_layers: int, n_heads: int, spec: str):
    """(att_layer_index, head) list for the taxonomy's content heads.

    Reads runs/attention_probe_<tag>/heads.csv (kind == content; layer is
    1-indexed -> attention index layer-1).  Falls back to every head in the
    mid-layer window when the taxonomy has not been computed for this model."""
    tag = model_name.split("/")[-1].replace("Qwen3-", "")
    csv = Path("runs") / f"attention_probe_{tag}" / "heads.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        content = df[df["kind"] == "content"]
        heads = [(int(r.layer) - 1, int(r.head)) for r in content.itertuples()]
        if heads:
            return heads
    return [(L, h) for L in _layer_window(n_layers, spec) for h in range(n_heads)]


@torch.no_grad()
def _forward(model, ids: list[int], device: str, want_attn: bool):
    t = torch.tensor([ids], device=device)
    out = model(t, output_attentions=want_attn, use_cache=False)
    return out


@torch.no_grad()
def _nll(logits: torch.Tensor, ids: list[int], answer: slice) -> list[float]:
    """Per-token NLL(reference) over the answer span (causal: pos p predicted by p-1)."""
    out = []
    for p in range(answer.start, answer.stop):
        if p == 0:
            out.append(float("nan"))
            continue
        lp = F.log_softmax(logits[0, p - 1].float(), dim=-1)
        out.append(-lp[ids[p]].item())
    return out


@torch.no_grad()
def _teacher_attn_footprint(attentions, heads, priv: slice, query: int,
                            keep: torch.Tensor):
    """Attention footprint from `query`, averaged over the CONTENT heads only
    (the taxonomy's high-retrieval heads; the attention sink lives in the other
    heads and otherwise dominates a mean-over-all-heads footprint).  `keep` is a
    0/1 mask over key positions that zeros structural/special tokens (e.g.
    <|im_start|>), so priv/ctx mass counts attention onto real content only.

    Returns (priv_mass, ctx_mass, top_ctx_idx) with ctx = prior non-privileged."""
    acc = None
    for (L, h) in heads:
        a = attentions[L][0, h, query, :].float()
        acc = a if acc is None else acc + a
    acc = (acc / len(heads)) * keep  # drop sink/special positions
    priv_mass = acc[priv].sum().item()
    ctx = acc[:query].clone()
    ctx[priv] = 0.0  # exclude the privileged block from in-context mass
    ctx_mass = ctx.sum().item()
    top_ctx = int(ctx.argmax().item()) if query > 0 and ctx.numel() and ctx.max() > 0 else -1
    return priv_mass, ctx_mass, top_ctx


def _keep_mask(ids: list[int], special_ids: set[int], device: str) -> torch.Tensor:
    """0/1 over key positions: drop structural/special tokens and position 0
    (the classic attention-sink anchors) so the footprint counts content only."""
    keep = torch.ones(len(ids), device=device)
    for j, tid in enumerate(ids):
        if j == 0 or tid in special_ids:
            keep[j] = 0.0
    return keep


def probe(model, tok, pairs, heads, special_ids, device: str) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        t_out = _forward(model, pair.teacher_ids, device, want_attn=True)
        s_out = _forward(model, pair.student_ids, device, want_attn=False)
        t_nll = _nll(t_out.logits, pair.teacher_ids, pair.t_answer)
        s_nll = _nll(s_out.logits, pair.student_ids, pair.s_answer)
        # privileged block in teacher coords: [s_aligned.start, t_aligned.start)
        priv = slice(pair.s_aligned.start, pair.t_aligned.start)
        keep = _keep_mask(pair.teacher_ids, special_ids, device)
        n = min(len(t_nll), len(s_nll))
        for p in range(n):
            q = pair.t_answer.start + p  # teacher query at the answer token
            t_priv, t_ctx, top_ctx = _teacher_attn_footprint(
                t_out.attentions, heads, priv, q, keep)
            reference = pair.teacher_ids[pair.t_answer.start + p]
            rows.append({
                "reference": tok.decode([reference]).strip(),
                "s_nll": s_nll[p],
                "t_nll": t_nll[p],
                "excess": s_nll[p] - t_nll[p],
                "t_priv": t_priv,
                "t_ctx": t_ctx,
                "top_ctx_tok": tok.decode([pair.teacher_ids[top_ctx]]).strip() if top_ctx >= 0 else "",
            })
        del t_out, s_out
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def classify(df: pd.DataFrame, excess_q: float = 0.66) -> pd.DataFrame:
    thr = df["excess"].quantile(excess_q)
    df["label"] = "low_surprise"
    hi = df["excess"] >= thr
    df.loc[hi & (df["t_priv"] >= df["t_ctx"]), "label"] = "knowledge_gap"
    df.loc[hi & (df["t_ctx"] > df["t_priv"]), "label"] = "misdirection"
    return df


def make_figure(df: pd.DataFrame, out_png: Path) -> None:
    color = {"knowledge_gap": "#0072B2", "misdirection": "#D55E00",
             "low_surprise": "#BBBBBB"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for lab, g in df.groupby("label"):
        a.scatter(g["t_priv"], g["excess"], s=14, alpha=0.7, lw=0,
                  c=color.get(lab, "#888"), label=f"{lab} ({len(g)})")
    a.set_xlabel("teacher attention on privileged block", fontsize=8)
    a.set_ylabel("excess surprise  (student NLL − teacher NLL)", fontsize=8)
    a.set_title("(a) high surprise: knowledge gap (right) vs misdirection (left)",
                fontsize=9, loc="left")
    a.legend(fontsize=7, frameon=False)
    a.grid(True, color="#dddddd", lw=0.5)

    counts = df["label"].value_counts()
    labels = [l for l in ("knowledge_gap", "misdirection", "low_surprise") if l in counts]
    b.bar(labels, [counts[l] for l in labels],
          color=[color[l] for l in labels])
    b.set_ylabel("answer tokens", fontsize=8)
    b.set_title("(b) surprise decomposition", fontsize=9, loc="left")
    b.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--examples", default="data/poem/examples_v4.jsonl")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--layers", default="mid", help="all | mid | lo:hi")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or f"runs/surprise_probe_{args.model.split('/')[-1].replace('Qwen3-', '')}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(args.device).eval()

    masker = ContextMasker(tok)
    records = [r for r in load_jsonl(args.examples) if r.get("privileged")][: args.n]
    pairs = [masker.build(SegmentedExample.from_record(r)) for r in records]

    # Aggregate the footprint over the taxonomy's CONTENT heads (heads.csv uses
    # 1-indexed layers -> att index L-1). Fall back to all heads in a mid window.
    heads = _content_heads(args.model, model.config.num_hidden_layers,
                           model.config.num_attention_heads, args.layers)
    special_ids = set(tok.all_special_ids or [])
    print(f"footprint over {len(heads)} heads; {len(special_ids)} special ids masked")

    df = probe(model, tok, pairs, heads, special_ids, args.device)
    df = classify(df)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tokens.csv", index=False)
    make_figure(df, out / "surprise.png")

    frac = df["label"].value_counts(normalize=True).to_dict()
    print(f"{len(df)} answer tokens over {len(pairs)} examples; {len(heads)} content heads")
    print("  label fractions:", {k: round(v, 3) for k, v in frac.items()})
    mis = df[df["label"] == "misdirection"].sort_values("excess", ascending=False).head(8)
    if not mis.empty:
        print("  top misdirection tokens (reference -> teacher's top in-context token):")
        for _, r in mis.iterrows():
            print(f"    excess={r['excess']:.2f}  reference={r['reference']!r:14} teacher_looks_at={r['top_ctx_tok']!r}")
    print(f"wrote {out}/tokens.csv and surprise.png")


if __name__ == "__main__":
    main()
