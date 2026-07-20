"""Family smoke harness: can this model family drive the layerwise machinery?

Per model: load through BlockStack (loud layout check) -> adapt_records on
the RAG dataset (template re-render) -> 1-example online teacher targets
(the untouched model IS the teacher) -> one strict local_block_step + one
connected readout window with teacher KL (finite losses, grads confined to
blocks, frozen vocab untouched) -> one greedy generation stopping on the family's turn
closer -> peak VRAM. Failures are recorded per stage and abort the
remaining stages for that model, never the harness — a FAIL row is a
result. Large 2026 families are included so template/BlockStack failures are
caught before wasting training queue time.

Usage:
    python compressed/smoke_family.py --model Qwen/Qwen3-0.6B      # control
    python compressed/smoke_family.py --all                        # current ladder
"""


# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP



import argparse
import sys
import traceback
from pathlib import Path


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.chatfmt import adapt_records, stop_token_id
from selfupdate.data.dataset import load_jsonl
from selfupdate.eval.recite import student_prompt
from selfupdate.masking import ContextMasker, SegmentedExample
from selfupdate.train.blocks import BlockStack
from selfupdate.train.steps import local_block_step, window_step

FAMILIES = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "Qwen/Qwen3.6-27B",
    "google/gemma-4-26B-A4B",
    "google/gemma-4-31B",
    "openai/gpt-oss-20b",
]
STAGES = ["load", "adapt", "teacher", "step", "window", "gen"]
FROZEN = ("embed", "final_norm", "lm_head")


def _advance(stack, h, pos_emb, upto):
    """No-grad student stream through blocks [1..upto]."""
    with torch.no_grad(), torch.autocast(h.device.type, dtype=torch.bfloat16):
        for L in range(1, upto + 1):
            h = stack.run_block(L, h, pos_emb)
    return h.detach()


def _frozen_params(stack):
    yield from stack.embed_tokens.parameters()
    yield from stack.final_norm.parameters()
    yield from stack.lm_head.parameters()


def smoke_one(name: str, examples: str, window_blocks: int, max_new: int,
              device: str = "cuda") -> dict:
    row = {s: "-" for s in STAGES}
    row["model"] = name
    row["vram_gb"] = "-"
    torch.cuda.reset_peak_memory_stats()
    model = None
    try:
        stage = "load"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
        model.to(device).train()
        stack = BlockStack(model)
        stack.freeze_non_blocks()
        n = stack.n_layers
        row[stage] = "OK"

        stage = "adapt"
        records = adapt_records(load_jsonl(examples), tok)
        pair = ContextMasker(tok).build(SegmentedExample.from_record(records[0]))
        s0, A = pair.s_aligned.start, pair.aligned_len
        ans0, t0 = pair.s_answer.start, pair.t_aligned.start
        row[stage] = "OK"

        stage = "teacher"
        t_ids = torch.tensor([pair.teacher_ids], device=device)
        t_pos = torch.arange(t_ids.shape[1], device=device)[None]
        targets = {}
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            h = stack.embed(t_ids)
            pos_emb_t = stack.rope(h, t_pos)
            for L in range(1, n + 1):
                h = stack.run_block(L, h, pos_emb_t)
                targets[L] = stack.loss_view(L, h)[0, t0: t0 + A].float().detach()
        row[stage] = "OK"

        s_ids = torch.tensor([pair.student_ids], device=device)
        s_pos = torch.arange(s_ids.shape[1], device=device)[None]
        h0 = stack.embed(s_ids)
        pos_emb_s = stack.rope(h0, s_pos)
        stage = "step"
        mid = n // 2
        h_in = _advance(stack, h0, pos_emb_s, mid - 1)
        loss_val, _ = local_block_step(
            stack, mid, h_in, pos_emb_s, targets[mid], s0, A, "nmse")
        assert torch.isfinite(torch.tensor(loss_val)), f"non-finite loss {loss_val}"
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in stack.block_params(mid)), "no grad in stepped block"
        assert all(p.grad is None for p in _frozen_params(stack)), \
            "gradient leaked into frozen vocab"
        model.zero_grad(set_to_none=True)
        row[stage] = "OK"

        stage = "window"
        L0 = n - window_blocks + 1
        h_in = _advance(stack, h0, pos_emb_s, L0 - 1)
        window_losses, _ = window_step(
            stack, L0, h_in, pos_emb_s, targets, s0, A, "nmse")
        assert all(torch.isfinite(v.detach().float()).all() for v in window_losses)
        assert all(p.grad is None for p in _frozen_params(stack)), \
            "gradient leaked into frozen vocab (window)"
        model.zero_grad(set_to_none=True)
        row[stage] = "OK"

        stage = "gen"
        model.eval()
        stop = stop_token_id(tok)
        prompt_ids = tok(student_prompt(records[0]),
                         return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            out = model.generate(prompt_ids, max_new_tokens=max_new,
                                 do_sample=False, eos_token_id=stop,
                                 pad_token_id=stop)
        text = tok.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        row[stage] = "OK"
        row["first_gen_line"] = text.strip().splitlines()[0][:60] if text.strip() else "(empty)"
        row["stop_token"] = repr(tok.decode([stop]))
    except Exception as e:  # noqa: BLE001 — a FAIL row is the product
        row[stage] = f"FAIL:{type(e).__name__}"
        row["error"] = str(e).splitlines()[0][:200] if str(e) else traceback.format_exc(limit=1)
    finally:
        row["vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        del model
        torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--examples", default="data/poem/examples_v2.jsonl")
    ap.add_argument("--window-blocks", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()
    models = args.model or (FAMILIES if args.all else ["Qwen/Qwen3-0.6B"])

    rows = []
    for name in models:
        print(f"--- {name}", flush=True)
        row = smoke_one(name, args.examples, args.window_blocks, args.max_new)
        rows.append(row)
        print(row, flush=True)

    print(f"\n{'model':44s} " + " ".join(f"{s:8s}" for s in STAGES) + " vram_gb")
    for r in rows:
        print(f"{r['model']:44s} " + " ".join(f"{str(r[s]):8s}" for s in STAGES)
              + f" {r['vram_gb']}")
        if "error" in r:
            print(f"    error: {r['error']}")
        if "first_gen_line" in r:
            print(f"    stop={r.get('stop_token')} first_line={r['first_gen_line']!r}")


if __name__ == "__main__":
    main()
