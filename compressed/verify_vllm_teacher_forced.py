"""Speculative-decoding verification: does our TRAINING STACK reproduce vLLM?

Owner framing (2026-07-19): "our training stack reproduces vLLM exactly", tested
as speculative decoding where **the whole vLLM answer is the draft** and our
training stack is the verifier. No autoregressive generation is implemented yet,
so the test reads vLLM's output as INPUT: one teacher-forced full-sequence
forward over ``[uncensored prompt + vLLM answer]`` through the SAME
``BlockStack.run_block`` walk the teacher capture uses (``_online_teacher_capture``:
embed -> run_block per layer, plain rope, NO censorship, NO past_key_values), then
argmax at every answer position is compared to vLLM's next token.

Acceptance semantics (greedy):
  full = prompt_token_ids (P) ++ token_ids (A)        # token_ids = vLLM answer
  predicted_answer[i] = argmax(logits[P+i-1])          # our stack's greedy pick
  accepted[i]         = predicted_answer[i] == token_ids[i]
The accepted PREFIX length is the first-rejection index; **if every token is
accepted, our stack run free-greedy would reproduce vLLM's sequence exactly**
(greedy is deterministic, so per-position argmax-agreement on the draft implies
free-run identity). The first rejection is exactly where free-running diverges.

Full-sequence forward => linear-attention layers are exact in one pass (the whole
reason the teacher never needed autoregression), so this is valid for the
hybrid qwen3_5 envelope (0.8B/27B/35B/122B).

Usage (training venv, one card):
  CUDA_VISIBLE_DEVICES=0 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  TRANSFORMERS_VERBOSITY=error /tmp/$USER/selfupdate-venv/bin/python \
    compressed/verify_vllm_teacher_forced.py \
      --model Qwen/Qwen3.5-0.8B \
      --responses runs/vllm_h100/qwen35_0p8b/responses_bs256.jsonl \
      --out runs/spec_verify/qwen35_0p8b.json   # no --limit = full 2071-item epoch
"""

from __future__ import annotations


# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP



import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.train.blocks import BlockStack


@torch.no_grad()
def teacher_forced_logits(stack, ids: torch.Tensor) -> torch.Tensor:
    """Uncensored full-sequence teacher forward, identical to
    _online_teacher_capture: embed -> run_block per layer (plain rope, no mask,
    no cache) -> final loss_view -> lm_head. ids is [B, T]; returns [B, T, V]."""
    device = ids.device
    B, T = ids.shape
    pos = torch.arange(T, device=device)[None].expand(B, -1)
    h = stack.embed(ids)
    pe = stack.rope(h, pos)
    for layer in range(1, stack.n_layers + 1):
        h = stack.run_block(layer, h, pe, position_ids=pos, input_ids=ids)
    view = stack.loss_view(stack.n_layers, h)          # post-final-norm
    return stack.lm_head(view)                          # [B, T, V]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--responses", required=True,
                    help="vLLM responses jsonl (prompt_token_ids + token_ids)")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole file (2071 items = one full epoch)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", default=None,
                    help="e.g. 'auto' to shard a large model across all visible "
                         "cards (4/8-GPU runs); leave unset for single-card")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--examples", type=int, default=8,
                    help="how many first-divergence examples to dump")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    rows = []
    with open(args.responses) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("prompt_token_ids") and r.get("token_ids"):
                rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break
    if not rows:
        sys.exit(f"no usable rows (prompt_token_ids+token_ids) in {args.responses}")
    print(f"loaded {len(rows)} vLLM responses from {args.responses}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    load_kw = dict(dtype=dtype)
    if args.device_map:
        load_kw["device_map"] = args.device_map
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    except (ValueError, KeyError, RuntimeError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kw)
    if args.device_map:                       # sharded across cards: use embed's card
        device = model.get_input_embeddings().weight.device
    else:
        model.to(device)
    model.eval()
    stack = BlockStack(model)
    print(f"loaded {args.model} device_map={args.device_map} dev={device} "
          f"({args.dtype}); n_layers={stack.n_layers}", flush=True)

    total_tokens = 0
    accepted_tokens = 0
    exact_items = 0
    first_token_hits = 0
    prompt_tokens = 0
    seq_tokens = 0
    prefix_fracs = []
    diverg_examples = []
    div_gaps = []          # our_argmax_logit - vllm_token_logit at first divergence
    div_top2 = 0           # vLLM token was in our top-2 (near-tie)
    div_top5 = 0
    margins = []           # top1-top2 logit margin at EVERY answer position:
                           # the "how hard is this token to flip" distribution
    depth_hits = [0, 0, 0, 0]   # acceptance by within-answer depth quartile
    depth_tot = [0, 0, 0, 0]    # (flat => margin story; decaying => accumulation)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for ri, r in enumerate(rows):
        p_ids = list(r["prompt_token_ids"])
        a_ids = list(r["token_ids"])
        P, A = len(p_ids), len(a_ids)
        if A == 0:
            continue
        prompt_tokens += P
        seq_tokens += P + A
        full = torch.tensor([p_ids + a_ids], dtype=torch.long, device=device)
        logits = teacher_forced_logits(stack, full)     # [1, P+A, V]
        # predicted answer token i comes from position P+i-1
        alog = logits[0, P - 1:P + A - 1].float()        # [A, V] answer-slot logits
        top2 = alog.topk(2, dim=-1).values                # [A, 2]
        margins.append((top2[:, 0] - top2[:, 1]).cpu())
        pred = alog.argmax(-1)                            # [A]
        # Under device_map=auto the lm_head (and thus logits) lives on the LAST
        # card while `device` is the embed card — compare on the logits' device.
        gold = torch.tensor(a_ids, device=alog.device)   # vLLM draft
        match = (pred == gold)
        n_acc = int(match.sum().item())
        for i in range(A):
            q = min(3, (4 * i) // A)
            depth_tot[q] += 1
            depth_hits[q] += int(match[i].item())
        gap = rank = None
        # accepted prefix = tokens before the first rejection
        if bool(match.all()):
            prefix = A
            exact_items += 1
        else:
            prefix = int((~match).float().argmax().item())
            # Diagnose the FIRST divergence: is vLLM's token a near-tie in OUR
            # own distribution (bf16 noise, irreducible across two kernels) or a
            # confident disagreement (a real forward difference / bug)?
            row = alog[prefix]
            gap = float(row[pred[prefix]] - row[gold[prefix]])   # our_top - vllm_tok
            rank = int((row > row[gold[prefix]]).sum().item())   # 0 = vLLM tok is OUR top
            div_gaps.append(gap)
            div_top2 += int(rank < 2)
            div_top5 += int(rank < 5)
        first_token_hits += int(match[0].item())
        total_tokens += A
        accepted_tokens += n_acc
        prefix_fracs.append(prefix / A)
        if len(diverg_examples) < args.examples and prefix < A:
            j = prefix
            diverg_examples.append({
                "example_id": r.get("example_id"),
                "answer_len": A,
                "accepted_prefix": prefix,
                "diverge_pos": j,
                "vllm_token": a_ids[j],
                "our_argmax": int(pred[j].item()),
                "vllm_piece": tok.decode([a_ids[j]]),
                "our_piece": tok.decode([int(pred[j].item())]),
                "our_minus_vllm_logit": round(gap, 3) if gap is not None else None,
                "vllm_tok_rank_in_ours": rank,
                "prefix_text": tok.decode(a_ids[:j])[:120],
            })

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    n = len(prefix_fracs)
    summary = {
        "model": args.model,
        "responses": args.responses,
        "items": n,
        "total_answer_tokens": total_tokens,
        "token_acceptance_rate": accepted_tokens / max(total_tokens, 1),
        "exact_sequence_match_rate": exact_items / max(n, 1),
        "first_token_agreement_rate": first_token_hits / max(n, 1),
        "mean_accepted_prefix_frac": sum(prefix_fracs) / max(n, 1),
        # Divergence diagnostics: is the gap bf16 ties or a real difference?
        "n_divergences": len(div_gaps),
        "mean_first_div_logit_gap": round(sum(div_gaps) / max(len(div_gaps), 1), 4),
        "frac_div_vllm_in_top2": round(div_top2 / max(len(div_gaps), 1), 4),
        "frac_div_vllm_in_top5": round(div_top5 / max(len(div_gaps), 1), 4),
        "acceptance_by_depth_quartile": [
            round(h / t, 4) if t else None
            for h, t in zip(depth_hits, depth_tot)],
        # top1-top2 margin distribution over ALL answer positions — the
        # measured "flip resistance" (tests the margin hypothesis directly).
        "margin_quantiles": (lambda m: {
            "p05": round(m.quantile(0.05).item(), 3),
            "p25": round(m.quantile(0.25).item(), 3),
            "p50": round(m.quantile(0.50).item(), 3),
            "p75": round(m.quantile(0.75).item(), 3),
            "frac_below_0p5": round((m < 0.5).float().mean().item(), 4),
            "frac_below_2": round((m < 2.0).float().mean().item(), 4),
        })(torch.cat(margins)) if margins else None,
        # Speed (teacher-forced, one full-sequence forward per item; NOT
        # autoregressive — compare to vLLM generation with this asymmetry in
        # mind: vLLM does A sequential decode steps, this does ONE forward).
        "seconds_forward_only": round(dt, 3),
        "answer_tok_per_s": round(total_tokens / max(dt, 1e-9), 1),
        "sequence_tok_per_s": round(seq_tokens / max(dt, 1e-9), 1),
        "items_per_s": round(n / max(dt, 1e-9), 2),
        "total_prompt_tokens": prompt_tokens,
        "total_sequence_tokens": seq_tokens,
        "batch": 1,
        "dtype": args.dtype,
        "diverge_examples": diverg_examples,
    }
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "diverge_examples"}, indent=2), flush=True)
    print("\n--- first-divergence examples ---", flush=True)
    for e in diverg_examples:
        print(f"  [{e['example_id']}] prefix {e['accepted_prefix']}/{e['answer_len']} "
              f"| vLLM={e['vllm_token']}({e['vllm_piece']!r}) "
              f"ours={e['our_argmax']}({e['our_piece']!r}) "
              f"after …{e['prefix_text']!r}", flush=True)

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {outp}", flush=True)


if __name__ == "__main__":
    main()
