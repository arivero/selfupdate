#!/usr/bin/env python
"""trainv5 — self-distillation with censored context (pipeline v5, experiment 1).

OWNER-DIRECTED DEPARTURE FROM v4 (2026-07-25). Deliberately a pure monolith:
no `selfupdate` imports, no shared cache identities, new run names — a v5 run
cannot collide with or silently reuse any v4 artifact. Code is copied, not
imported, where v4 had the right idea.

The v5 training law
-------------------
Teacher and student are the SAME base model:

  teacher  = base model, adapters OFF, prompt WITH the privileged passage
             (frozen: never receives gradient; it is the model's own
             uncensored self, exactly the vLLM-generation condition).
  student  = base model + LoRA (all decoder Linears, all layers), prompt
             WITHOUT the privileged passage (remove-view censorship: the
             passage text is cut from the prompt, matching deployment,
             where no retrieval is present).
  loss     = KL(teacher || student) on the teacher-forced ANSWER positions
             (default), or CE toward the teacher's own answer tokens
             (--loss ce). Both are teacher-sourced; the original corpus
             text is never a training target (branch law).

Gradient flows end-to-end through the student — attention AND the post-
attention MLPs ("the memorization perceptrons") in every layer get real
output-level signal. This attacks both v4 failures at once: (1) the v4
block-local hidden loss was near-zero/misaligned so the MLPs never learned;
(2) v4's precomputed teacher K/V went stale under adaptation — v5 has no
K/V cache at all: the student recomputes its context through the current
adapters at every step.

Roadmap (vN, owner mental model): censorship generalizes from "remove the
retrieved passage" to "mask distant tokens whose attention is high" —
self-distilling the model's own high-attention discoveries into weights,
i.e. continuous personalization of per-user adapters. This file keeps
censorship localized (the `censored_ids` construction in build_items) so an
attention-top-k mode can be added without touching the training loop.

Owner design notes honored:
- LoRA gives EVERY layer capacity (we do not know which layer memorizes
  poetry); per-layer teacher-vs-student "surprise" is logged every epoch so
  the run itself localizes the memorization. --layer-gate topk:N optionally
  restricts updates to the N layers most surprised in the previous epoch.
- Capacity check at startup: trainable LoRA params x 3 bits must exceed the
  gzip-compressed bits of the memorization corpora (Machado + Cervantes).

Inputs reused from the existing workflow (read-only):
- data/combined/examples_v5rs_window.jsonl   (questions + privileged passages)
- runs/vllm_h100/<model>/responses_bs*.jsonl (teacher answers, exact ids)
Everything else (censored prompts, training, eval, telemetry) happens here.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# data: reconstruct teacher / censored prompts from stored artifacts
# --------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_items(examples_path: Path, responses_path: Path, tok,
                limit: int = 0) -> tuple[list[dict], int]:
    """Join examples (passage text) with responses (exact prompt/answer ids)
    and derive the censored prompt by text surgery.

    Round-trip guard: re-encoding the decoded full prompt must reproduce the
    stored `prompt_token_ids` exactly; otherwise this tokenizer/template does
    not match the artifact and the run must not proceed.
    """
    examples = {r["example_id"]: r for r in load_jsonl(examples_path)}
    responses = load_jsonl(responses_path)
    stop_id = responses[0].get("stop_token_id")
    if limit:
        responses = responses[:: max(1, len(responses) // limit)][:limit]
    items, bad_roundtrip, bad_cut = [], 0, 0
    for r in responses:
        ex = examples.get(r["example_id"])
        if ex is None:
            raise SystemExit(f"response {r['example_id']} missing from examples")
        prompt_ids = r["prompt_token_ids"]
        answer_ids = r["token_ids"]  # includes the trailing stop token
        text = tok.decode(prompt_ids, skip_special_tokens=False)
        if tok.encode(text, add_special_tokens=False) != prompt_ids:
            bad_roundtrip += 1
            continue
        priv = ex["privileged"]
        if priv and priv in text:
            censored_text = text.replace(priv, "", 1)
        elif priv and priv.strip() and priv.strip() in text:
            censored_text = text.replace(priv.strip(), "", 1)
        else:
            bad_cut += 1
            continue
        censored_ids = tok.encode(censored_text, add_special_tokens=False)
        items.append({
            "example_id": r["example_id"],
            "corpus": ex.get("corpus", "?"),
            "prompt_ids": prompt_ids,
            "censored_ids": censored_ids,
            "answer_ids": answer_ids,
            "answer_text": r.get("answer_text", ""),
            "teacher_word_acc": r.get("word_acc"),
        })
    if bad_roundtrip or bad_cut:
        raise SystemExit(
            f"FATAL: prompt reconstruction failed — round-trip mismatches: "
            f"{bad_roundtrip}, passage-not-found: {bad_cut} "
            f"(of {len(responses)}). Tokenizer/template does not match the "
            f"stored responses; refuse to train on a corrupted view.")
    return items, stop_id


def capacity_check(trainable_params: int, bits_per_param: float = 3.0) -> dict:
    """Owner rule: each parameter stores ~3 bits (of compressed text)."""
    corpora = [ROOT / "data/poem/raw.txt",
               ROOT / "data/quijote/raw_ch1.txt",
               ROOT / "data/quijote/raw_ch4.txt",
               ROOT / "data/quijote/raw_ch8.txt",
               ROOT / "data/quijote/raw_ch16.txt"]
    raw = b"".join(p.read_bytes() for p in corpora if p.exists())
    corpus_bits = len(gzip.compress(raw, 9)) * 8
    have_bits = trainable_params * bits_per_param
    return {"corpus_gzip_bits": corpus_bits,
            "trainable_params": trainable_params,
            "capacity_bits": have_bits,
            "capacity_ratio": have_bits / max(1, corpus_bits),
            "capacity_ok": have_bits >= corpus_bits}


def resolve_stack(causal_lm):
    """Locate the decoder stack (has .layers/.embed_tokens) and the frozen
    lm_head, tolerating multimodal composites that nest the text tower."""
    for path in ("model", "language_model", "model.language_model",
                 "language_model.model", "model.model"):
        obj = causal_lm
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "layers") \
                and hasattr(obj, "embed_tokens"):
            head = getattr(causal_lm, "lm_head", None)
            if head is None:
                head = getattr(getattr(causal_lm, "language_model", causal_lm),
                               "lm_head", None)
            if head is None:
                raise SystemExit("decoder stack found but lm_head is not")
            return obj, head
    raise SystemExit(f"cannot locate decoder stack on {type(causal_lm)}")


# --------------------------------------------------------------------------
# batching / forward helpers
# --------------------------------------------------------------------------

def bucketed_batches(items: list[dict], micro_batch: int, seed: int,
                     epoch: int, key: str) -> list[list[dict]]:
    """Length-sorted buckets of micro_batch items, bucket order shuffled per
    epoch (deterministic in seed+epoch). Sorting bounds padding waste."""
    order = sorted(items, key=lambda it: len(it[key]) + len(it["answer_ids"]))
    batches = [order[i:i + micro_batch]
               for i in range(0, len(order), micro_batch)]
    random.Random(seed * 1000 + epoch).shuffle(batches)
    return batches


def collate(batch: list[dict], key: str, pad_id: int, device):
    """Right-padded teacher-forced batch. Returns ids, attention mask, and
    per-row (start, length) of the answer's PREDICTIVE logit rows: the logit
    at position t predicts token t+1, so an answer occupying positions
    P..P+A-1 is predicted by rows P-1..P+A-2."""
    import torch
    rows = [it[key] + it["answer_ids"] for it in batch]
    maxlen = max(len(r) for r in rows)
    ids = torch.full((len(rows), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), maxlen), dtype=torch.long)
    spans = []
    for i, (it, row) in enumerate(zip(batch, rows)):
        ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, :len(row)] = 1
        spans.append((len(it[key]) - 1, len(it["answer_ids"])))
    return ids.to(device), mask.to(device), spans


def answer_logits(stack, lm_head, ids, mask, spans, want_hidden: bool):
    """Forward the bare decoder stack, project ONLY the answer's predictive
    rows through the frozen lm_head (never materialize full-sequence logits:
    262k vocab x 4k positions would not fit). Returns per-item logit tensors
    and, optionally, per-item per-layer hidden slices (kept on their own
    devices; compare layer-to-layer only, never stacked across devices)."""
    out = stack(input_ids=ids, attention_mask=mask,
                output_hidden_states=want_hidden, use_cache=False)
    h_last = out.last_hidden_state
    logit_rows, hidden_rows = [], []
    for i, (start, length) in enumerate(spans):
        logit_rows.append(lm_head(h_last[i, start:start + length]).float())
        if want_hidden:
            hidden_rows.append([h[i, start:start + length].detach()
                                for h in out.hidden_states])
    return logit_rows, hidden_rows


# --------------------------------------------------------------------------
# evaluation (self-contained)
# --------------------------------------------------------------------------

def word_lcs_acc(reference: str, hypothesis: str) -> float:
    """Word-level LCS accuracy against the reference (order-preserving)."""
    a, b = reference.split(), hypothesis.split()
    if not a:
        return 1.0 if not b else 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = (dp[i][j] + 1 if a[i] == b[j]
                                else max(dp[i][j + 1], dp[i + 1][j]))
    return dp[len(a)][len(b)] / len(a)


def recall_eval(peft_model, tok, items: list[dict], device, stop_id: int,
                gen_batch: int, max_new: int, sample_per_corpus: int,
                seed: int) -> dict:
    """Deployment-condition recall: greedy-generate from the CENSORED prompt
    (adapters ON, no passage) and score word-LCS against the teacher's
    uncensored answer — 'is the model now a Machado expert without the book
    open?'. Deterministic per-corpus samples."""
    import torch
    rng = random.Random(seed)
    by_corpus: dict[str, list[dict]] = {}
    for it in items:
        by_corpus.setdefault(it["corpus"], []).append(it)
    chosen = []
    for _, group in sorted(by_corpus.items()):
        chosen += rng.sample(group, min(sample_per_corpus, len(group)))
    pad = tok.pad_token_id or 0
    scores: dict[str, list[float]] = {}
    peft_model.eval()
    with torch.no_grad():
        for i in range(0, len(chosen), gen_batch):
            batch = chosen[i:i + gen_batch]
            maxp = max(len(it["censored_ids"]) for it in batch)
            ids = torch.full((len(batch), maxp), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, it in enumerate(batch):  # LEFT pad for generation
                p = it["censored_ids"]
                ids[j, maxp - len(p):] = torch.tensor(p, dtype=torch.long)
                mask[j, maxp - len(p):] = 1
            gen = peft_model.generate(
                input_ids=ids.to(device), attention_mask=mask.to(device),
                max_new_tokens=max_new, do_sample=False, use_cache=True,
                eos_token_id=stop_id, pad_token_id=pad)
            for j, it in enumerate(batch):
                text = tok.decode(gen[j, maxp:],
                                  skip_special_tokens=True).strip()
                scores.setdefault(it["corpus"], []).append(
                    word_lcs_acc(it["answer_text"], text))
    return {c: round(sum(v) / len(v), 4) for c, v in scores.items() if v}


def arc_eval(stack, lm_head, tok, device, limit: int) -> float:
    """Standard-damage guard: arc_easy accuracy (vendored fixed items, mean
    per-token choice log-prob, adapters ON — the deployed student)."""
    import torch
    data = json.load(open(ROOT / "data/eval/arc_easy_v1.json"))
    items = (data["items"] if isinstance(data, dict) else data)[:limit]
    pad = tok.pad_token_id or 0
    correct = 0
    with torch.no_grad():
        for it in items:
            prompt_ids = tok.encode(it["prompt"])
            rows, spans = [], []
            for ch in it["choices"]:
                cids = tok.encode(ch, add_special_tokens=False)
                rows.append(prompt_ids + cids)
                spans.append((len(prompt_ids) - 1, len(cids)))
            maxlen = max(len(r) for r in rows)
            ids = torch.full((len(rows), maxlen), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, r in enumerate(rows):
                ids[j, :len(r)] = torch.tensor(r, dtype=torch.long)
                mask[j, :len(r)] = 1
            out = stack(input_ids=ids.to(device), attention_mask=mask.to(device),
                        use_cache=False)
            best, best_lp = -1, -1e30
            for j, (start, length) in enumerate(spans):
                lg = lm_head(out.last_hidden_state[j, start:start + length]).float()
                tgt = torch.tensor(rows[j][start + 1:start + 1 + length],
                                   device=lg.device)
                lp = torch.log_softmax(lg, -1).gather(
                    1, tgt[:, None]).mean().item()
                if lp > best_lp:
                    best, best_lp = j, lp
            correct += int(best == it["target"])
    return correct / len(items)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="google/gemma-4-31B-it",
                    help="dense model (default: the campaign's dense control)")
    ap.add_argument("--examples",
                    default="data/combined/examples_v5rs_window.jsonl")
    ap.add_argument("--responses",
                    default="runs/vllm_h100/gemma4_31b_it/responses_bs256.jsonl",
                    help="teacher answers with exact prompt/answer token ids")
    ap.add_argument("--run-name", default="trainv5_g31b_selfdistill")
    ap.add_argument("--loss", choices=("kl", "ce"), default="kl",
                    help="kl: full-distribution KL(teacher||student); "
                         "ce: CE toward the teacher's own answer tokens")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--layer-gate", default="all",
                    help="'all' (depth-uniform, default) or 'topk:N' — update "
                         "only the N layers most surprised last epoch")
    ap.add_argument("--surprise-every", type=int, default=8,
                    help="capture per-layer surprise on 1-of-N batches")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--recall-samples", type=int, default=24,
                    help="generative recall items per corpus per eval")
    ap.add_argument("--arc-limit", type=int, default=100)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: cap the item count (0 = all 2071)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--dry-data", action="store_true",
                    help="CPU-only: build/verify items, print stats, exit")
    args = ap.parse_args()

    out_dir = ROOT / "runs" / args.run_name
    if out_dir.exists() and not args.dry_data:
        raise SystemExit(f"fresh run dir already exists: {out_dir}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    items, stop_id = build_items(ROOT / args.examples, ROOT / args.responses,
                                 tok, args.limit)
    n_ans = sum(len(it["answer_ids"]) for it in items)
    cut = sum(len(it["prompt_ids"]) - len(it["censored_ids"]) for it in items)
    print(f"items={len(items)} answer_tokens={n_ans} "
          f"mean_censored_cut_tokens={cut / len(items):.1f} stop_id={stop_id}",
          flush=True)
    by_c: dict[str, int] = {}
    for it in items:
        by_c[it["corpus"]] = by_c.get(it["corpus"], 0) + 1
    print("corpora:", by_c, flush=True)
    if args.dry_data:
        ex = items[0]
        print("sample censored prompt tail:",
              repr(tok.decode(ex["censored_ids"][-40:])))
        print("sample answer:", repr(ex["answer_text"]))
        print("teacher word_acc mean:",
              round(sum(it["teacher_word_acc"] or 0 for it in items)
                    / len(items), 4))
        return

    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    out_dir.mkdir(parents=True)
    metrics = open(out_dir / "metrics.jsonl", "a")

    def log(kind: str, **kw):
        metrics.write(json.dumps(
            {"kind": kind, "run_name": args.run_name, "pipeline_version": 5,
             "t": time.time(), **kw}) + "\n")
        metrics.flush()

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    log("provenance", source_commit=commit, args=vars(args),
        training_target="teacher (uncensored self) answer distribution",
        censorship="remove_view (privileged passage cut from student prompt)",
        end_to_end_student_training=True, final_logit_training=True,
        frozen_vocabulary=True,
        note="v5 owner-directed departure from the v4 block-local law, "
             "2026-07-25")

    print(f"loading {args.model} (bf16, device_map=auto)", flush=True)
    full = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    full.config.use_cache = False
    # every ordinary Linear in the decoder blocks — attention AND MLP, ALL
    # layers: capacity everywhere, since we don't know where poetry lives.
    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM")
    full = get_peft_model(full, lcfg)
    full.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    full.enable_input_require_grads()

    trainable = sum(p.numel() for p in full.parameters() if p.requires_grad)
    cap = capacity_check(trainable)
    print(f"trainable={trainable:,} "
          f"capacity_ratio={cap['capacity_ratio']:.1f} "
          f"({'OK' if cap['capacity_ok'] else 'INSUFFICIENT'})", flush=True)
    log("capacity", **cap)
    if not cap["capacity_ok"]:
        raise SystemExit("FATAL: LoRA below the 3-bits-per-param corpus "
                         "bound; raise --lora-r")

    stack, lm_head = resolve_stack(full.get_base_model())
    for p in lm_head.parameters():
        assert not p.requires_grad, "lm_head must stay frozen"
    device = next(stack.parameters()).device
    n_layers = len(stack.layers)
    # frozen-vocabulary tripwire (law copied from v4): the head never moves
    head_fp0 = lm_head.weight.detach().float().sum().item()

    opt = torch.optim.AdamW((p for p in full.parameters() if p.requires_grad),
                            lr=args.lr)
    pad = tok.pad_token_id or 0

    if args.layer_gate == "all":
        gate_k = 0
    elif args.layer_gate.startswith("topk:"):
        gate_k = int(args.layer_gate.split(":", 1)[1])
    else:
        raise SystemExit(f"unknown --layer-gate {args.layer_gate}")
    gate_layers: set[int] | None = None

    def evaluate(epoch: int):
        rec = recall_eval(full, tok, items, device, stop_id, args.gen_batch,
                          args.max_new_tokens, args.recall_samples,
                          args.seed + epoch)
        full.eval()
        arc = arc_eval(stack, lm_head, tok, device, args.arc_limit)
        # teacher-forced argmax acceptance on a fixed probe slice — the
        # metric that sat frozen at 0.556 through all of v4.
        probe = items[:: max(1, len(items) // 128)][:128]
        agree = total = 0
        with torch.no_grad():
            for i in range(0, len(probe), args.micro_batch):
                b = probe[i:i + args.micro_batch]
                ids, mask, spans = collate(b, "censored_ids", pad, device)
                lgs, _ = answer_logits(stack, lm_head, ids, mask, spans, False)
                for it, lg in zip(b, lgs):
                    tgt = torch.tensor(it["answer_ids"], device=lg.device)
                    agree += (lg.argmax(-1) == tgt).sum().item()
                    total += len(it["answer_ids"])
        acc = agree / max(1, total)
        print(f"eval e{epoch}: recall={rec} arc_easy={arc:.3f} "
              f"student_argmax={acc:.4f}", flush=True)
        log("eval", epoch=epoch, recall=rec, arc_easy=arc,
            student_argmax_acceptance=acc)

    evaluate(0)  # epoch zero: identical conditions to every checkpoint

    for epoch in range(1, args.epochs + 1):
        full.train()
        t0 = time.time()
        batches = bucketed_batches(items, args.micro_batch, args.seed, epoch,
                                   "censored_ids")
        epoch_loss = 0.0
        surprise_sum = [0.0] * (n_layers + 1)
        surprise_n = 0
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(batches):
            want_h = (bi % args.surprise_every == 0)
            # teacher: adapters OFF, WITH passage, no grad
            with torch.no_grad(), full.disable_adapter():
                t_ids, t_mask, t_spans = collate(batch, "prompt_ids", pad,
                                                 device)
                t_logits, t_hidden = answer_logits(stack, lm_head, t_ids,
                                                   t_mask, t_spans, want_h)
            # student: adapters ON, passage REMOVED, grad
            s_ids, s_mask, s_spans = collate(batch, "censored_ids", pad,
                                             device)
            s_logits, s_hidden = answer_logits(stack, lm_head, s_ids, s_mask,
                                               s_spans, want_h)
            loss = 0.0
            ntok = 0
            for it, tl, sl in zip(batch, t_logits, s_logits):
                if args.loss == "kl":
                    tp = torch.log_softmax(tl.detach(), -1)
                    sp = torch.log_softmax(sl, -1)
                    loss = loss + (tp.exp() * (tp - sp)).sum()
                else:
                    tgt = torch.tensor(it["answer_ids"], device=sl.device)
                    loss = loss + torch.nn.functional.cross_entropy(
                        sl, tgt, reduction="sum")
                ntok += tl.shape[0]
            loss = loss / max(1, ntok)
            (loss / args.grad_accum).backward()
            epoch_loss += float(loss.detach())
            if want_h and t_hidden:
                # per-layer relative discrepancy; each layer's teacher and
                # student slices share a device, so this never crosses GPUs
                for th_l, sh_l in zip(t_hidden, s_hidden):
                    for li, (th, sh) in enumerate(zip(th_l, sh_l)):
                        d = (th.float() - sh.float()).norm(dim=-1).mean()
                        n = th.float().norm(dim=-1).mean().clamp_min(1e-6)
                        surprise_sum[li] += float(d / n)
                surprise_n += len(t_hidden)
            if (bi + 1) % args.grad_accum == 0 or bi + 1 == len(batches):
                if gate_k and gate_layers is not None:
                    for name, p in full.named_parameters():
                        if p.grad is None or "lora_" not in name:
                            continue
                        lyr = next((int(t) for t in name.split(".")
                                    if t.isdigit()), None)
                        if lyr is not None and lyr not in gate_layers:
                            p.grad = None
                torch.nn.utils.clip_grad_norm_(
                    [p for p in full.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

        profile = [s / max(1, surprise_n) for s in surprise_sum]
        if gate_k:
            # hidden_states[li] is the OUTPUT of layer li (index 0 = embed),
            # so layer index li maps to decoder layer li-1's product; gate on
            # the producing layer.
            ranked = sorted(range(1, n_layers + 1), key=lambda l: -profile[l])
            gate_layers = {l - 1 for l in ranked[:gate_k]}
            log("layer_gate", epoch=epoch, active_layers=sorted(gate_layers))
        mean_loss = epoch_loss / len(batches)
        print(f"epoch {epoch}: loss={mean_loss:.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        log("epoch", epoch=epoch, loss=mean_loss,
            seconds=time.time() - t0, surprise_profile=profile)
        assert lm_head.weight.detach().float().sum().item() == head_fp0, \
            "FROZEN-VOCABULARY TRIPWIRE: lm_head moved"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            evaluate(epoch)
            full.save_pretrained(str(out_dir / f"checkpoint_e{epoch}"))

    full.save_pretrained(str(out_dir / "checkpoint"))
    log("done", epochs=args.epochs)
    print(f"run complete: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
