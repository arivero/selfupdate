#!/usr/bin/env python
"""trainv5 — layerwise self-distillation with censored context (pipeline v5).

OWNER-DIRECTED (2026-07-25/26). Deliberately a pure monolith: no
`selfupdate` imports, no shared cache identities, new run names — a v5 run
cannot collide with or silently reuse any v4 artifact.

The v5 training law (owner-corrected 2026-07-26: "backprop only happens
layerwise" — there is NO end-to-end training graph in v5)
----------------------------------------------------------------------
Teacher and student are the SAME base model:

  teacher     = base model, adapters OFF, prompt WITH the privileged
                passage (frozen; it is the model's own uncensored self,
                exactly the vLLM answer-generation condition). One no-grad
                pass records its per-layer hidden states h_t[1..N] at the
                teacher-forced ANSWER positions.
  student     = base model + LoRA (all decoder Linears, ALL layers),
                prompt WITHOUT the passage (remove-view censorship =
                deployment condition). One forward in which every block's
                INPUT is detached at the boundary: block L transforms the
                student's OWN censored trajectory state h_s[L-1] (current
                adapters, current attention context — nothing cached,
                nothing stale) into y_L.
  local loss  = distance(y_L[answer], h_t[L][answer]) per layer, one term
                per block, DEPTH-UNIFORM (same loss kind and weight at
                every layer). Each term's graph roots only in block L's
                LoRA weights; a single backward delivers every block its
                purely local gradient. Cross-block gradient flow is
                structurally unrepresentable (inputs are detached).

Teacher-sourced targets only; the original corpus text is never a training
target.

STABILITY LAW (learned the hard way, 2026-07-26, run
trainv5_g31b_selfdistill_lr1e4_destroyed): the answer-row objective alone is
satisfiable while destroying the model — at lr 1e-4 the blocks wrecked every
unconstrained position within ONE epoch (argmax 0.43->0.0, CE 11->123, arc
to chance) while the mean local loss fell. Two defenses, both default-on:
lr 1e-5, and a SELF-ANCHOR term (--anchor-weight, --anchor-rows): at fixed
sampled PROMPT rows of the same censored sequence, each block's output must
stay near the adapters-OFF base model's states — learn the passage at answer
rows, change nothing where there is no signal. The anchor targets are frozen
(base model) and cached like the teacher's. This is also the continuous-
learning requirement: personalization must never lobotomize the base.

OWNER APPROVAL GATE (2026-07-26): training on output logits is completely
forbidden — output-level KL/CE against the teacher exist only inside
evaluate() under no_grad (optimizer weight structurally zero). The
embedding and unembedding matrices (and final norm) never move or train:
asserted at startup (no trainable parameter outside decoder-block LoRA)
and fingerprint-tripwired every epoch. Plain output distillation is a
commodity fine-tune; the layerwise law is the project.

The per-layer loss KIND is the project's active research axis (the v4
campaign screened huber/cosine/delta_cosine; vocab_mse is the historical
recall recipe's loss): --local-loss huber|nmse|cosine|delta_cosine|
vocab_mse, formulas copied verbatim from the module. vocab_mse measures
hidden distance through the frozen unembedding Gram matrix W^T W — a
measurement device, not logit training. delta_cosine's anchor is the
block's own detached input (here: the censored student's h_s[L-1]).

Owner speed insight (2026-07-26): the forward pass computes EVERY layer's
local loss anyway, and each block's backward is independent — so the gate
may skip the backward of small-surprise layers entirely. --layer-gate
topk:N backprops only the N largest per-layer losses each step;
minfrac:F skips layers below F x the step's largest. Per-layer surprise
and per-layer backprop counts are logged every epoch regardless.

Why this attacks both v4 failures:
1. v4 fed block L the teacher's exact h[L-1] and exact uncensored K/V, so
   the local residual was ~0 and the MLPs ("the memorization perceptrons")
   learned nothing. Here the block input is the student's own CENSORED
   trajectory — the per-layer residual against the passage-informed teacher
   is large, and closing it requires storing the passage's contribution in
   the weights.
2. v4's precomputed teacher K/V went stale as adapters trained. v5 caches
   nothing: the student's attention context is recomputed through current
   adapters at every forward.

Roadmap (vN, owner mental model): censorship generalizes from "remove the
retrieved passage" to "mask distant tokens whose attention is high" —
self-distilling the model's own high-attention discoveries into weights,
i.e. continuous personalization of per-user adapters. Censorship is
localized in build_items so that mode can slot in without touching the
training loop.

Owner design notes honored:
- LoRA gives EVERY layer capacity (we do not know which layer memorizes
  poetry); the per-layer local loss IS the teacher-vs-student surprise and
  is logged every epoch — the profile itself localizes the memorization.
  --layer-gate topk:N trains only the N most-surprised layers (their loss
  terms are simply the only ones formed).
- Capacity check at startup: trainable LoRA params x 3 bits must exceed
  the gzip-compressed bits of the memorization corpora (Machado +
  Cervantes).

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
from contextlib import contextmanager
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
    per-row (start, length) of the answer's PREDICTIVE rows: the state at
    position t predicts token t+1, so an answer occupying positions
    P..P+A-1 is predicted from rows P-1..P+A-2."""
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


def slice_rows(hidden, spans):
    """Per-row answer slices [A_i, H] from a [B, S, H] tensor."""
    return [hidden[i, s:s + l] for i, (s, l) in enumerate(spans)]


class LayerwiseTaps:
    """Hook plumbing for the layerwise law.

    - forward PRE-hook on every decoder layer detaches its input hidden
      states: each block transforms the real current student trajectory,
      but no gradient can cross a block boundary (structural, not policy).
      The detached input is kept (it is the delta_cosine anchor).
    - forward hook collects each block's differentiable OUTPUT so per-layer
      local losses can be formed after the single pass.
    Installed only inside the training step (context manager); evaluation
    and generation run the unhooked model.
    """

    def __init__(self, layers):
        self.layers = layers
        self.inputs: list = [None] * len(layers)
        self.outputs: list = [None] * len(layers)
        self._handles = []

    def _pre(self, idx):
        def hook(module, args, kwargs):
            import torch
            if args and torch.is_tensor(args[0]):
                det = args[0].detach()
                self.inputs[idx] = det
                return (det,) + tuple(args[1:]), kwargs
            if "hidden_states" in kwargs \
                    and torch.is_tensor(kwargs["hidden_states"]):
                kwargs = dict(kwargs)
                det = kwargs["hidden_states"].detach()
                self.inputs[idx] = det
                kwargs["hidden_states"] = det
                return args, kwargs
            raise RuntimeError("decoder layer called without hidden states")
        return hook

    def _post(self, idx):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            self.outputs[idx] = out
        return hook

    @contextmanager
    def active(self):
        try:
            for i, layer in enumerate(self.layers):
                self._handles.append(layer.register_forward_pre_hook(
                    self._pre(i), with_kwargs=True))
                self._handles.append(layer.register_forward_hook(
                    self._post(i)))
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()
            self.inputs = [None] * len(self.layers)
            self.outputs = [None] * len(self.layers)


class LocalLoss:
    """Depth-uniform per-layer distance — the loss MENU is the project's
    active research axis (the v4 campaign screened huber/cosine/delta_cosine;
    vocab_mse is the historical recall recipe's loss). Formulas are copied
    verbatim from src/selfupdate/train/losses.py; only the delta_cosine
    anchor changes meaning: it is the block's own detached input, which in
    v5 is the CENSORED STUDENT trajectory state h_s[L-1] (in v4 it was
    teacher h[L-1] because that was the block input there).

    y: [A, H] differentiable block output; target: [A, H] detached teacher
    h_t[L]; anchor: [A, H] detached block input h_s[L-1].
    """

    def __init__(self, kind: str, lm_head):
        self.kind = kind
        self.lm_head = lm_head
        self._gram_by_dev: dict = {}

    def _gram(self, device):
        """M = W^T W of the frozen unembedding, fp32, chunked over vocab
        rows; cached per device (blocks live on several GPUs)."""
        import torch
        if device not in self._gram_by_dev:
            if not self._gram_by_dev:
                W = self.lm_head.weight.detach()
                H = W.shape[1]
                M = torch.zeros(H, H, dtype=torch.float32, device=W.device)
                for i in range(0, W.shape[0], 16384):
                    w = W[i:i + 16384].float()
                    M += w.T @ w
                self._gram_by_dev[W.device] = M
            src = next(iter(self._gram_by_dev.values()))
            self._gram_by_dev[device] = src.to(device)
        return self._gram_by_dev[device]

    def __call__(self, y, target, anchor):
        import torch
        import torch.nn.functional as F
        kind = self.kind
        s, t = y.float(), target.float()
        if kind == "nmse":
            return F.mse_loss(s, t) / t.pow(2).mean().clamp_min(1e-8)
        if kind == "cosine":
            return 1.0 - F.cosine_similarity(s, t, dim=-1, eps=1e-8).mean()
        if kind == "huber":
            scale = t.pow(2).mean().sqrt().clamp_min(1e-8)
            return F.smooth_l1_loss(s / scale, t / scale, beta=1.0)
        if kind == "delta_cosine":
            a = anchor.float()
            student_delta = s - a
            with torch.no_grad():
                teacher_delta = t - a
            return 1.0 - F.cosine_similarity(
                student_delta, teacher_delta, dim=-1, eps=1e-8).mean()
        if kind == "vocab_mse":
            with torch.autocast(s.device.type, enabled=False):
                d = s - t
                M = self._gram(s.device)
                q = (d @ M * d).sum(-1).mean()
                denom = (t @ M * t).sum(-1).mean().clamp_min(1e-8)
                return q / denom
        raise SystemExit(f"unknown --local-loss {kind}")


# --------------------------------------------------------------------------
# evaluation (self-contained; all output-level numbers are EVAL ONLY)
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
    (adapters ON, no passage, ordinary full forward — the student's real
    trajectory) and score word-LCS against the teacher's uncensored answer."""
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
    ap.add_argument("--local-loss",
                    choices=("huber", "nmse", "cosine", "delta_cosine",
                             "vocab_mse"),
                    default="huber",
                    help="depth-uniform per-layer distance to teacher h_t[L] "
                         "(the v4 loss-screen menu; vocab_mse measures hidden "
                         "distance in frozen-unembedding geometry — a "
                         "measurement device, NOT logit training)")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="AdamW on LoRA. 1e-4 destroyed the model inside one "
                         "epoch (run trainv5_g31b_selfdistill_lr1e4_destroyed:"
                         " argmax 0.43->0.0, CE 11->123, arc to chance)")
    ap.add_argument("--epochs", type=int, default=40,
                    help="epochs are ~2 min once the teacher cache is warm")
    ap.add_argument("--anchor-weight", type=float, default=1.0,
                    help="weight of the self-anchor term: at sampled PROMPT "
                         "rows of the same censored sequence, each block's "
                         "output must stay near the adapters-OFF base states "
                         "— learn the passage at answer rows, change nothing "
                         "elsewhere (the lr1e4 run proved answer-only "
                         "constraints let the blocks wreck general function)."
                         " 0 disables")
    ap.add_argument("--anchor-rows", type=int, default=64,
                    help="sampled prompt rows per item for the anchor term")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--layer-gate", default="all",
                    help="'all' (depth-uniform, default); 'topk:N' — each "
                         "STEP, backprop only the N largest per-layer "
                         "losses; 'minfrac:F' — skip backward for layers "
                         "whose loss < F x the step's largest (owner speed "
                         "insight 2026-07-26: forward computes every local "
                         "loss anyway, and per-block backward is "
                         "independent, so small-surprise layers can simply "
                         "not be backpropagated)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="enable activation checkpointing (off by default: "
                         "detached block boundaries already bound memory per "
                         "block, and hooks+recompute interplay is unproven)")
    ap.add_argument("--teacher-cache", choices=("cpu", "off"), default="cpu",
                    help="cache teacher answer-row hiddens in host RAM after "
                         "their first computation (~51 GiB at 31B full "
                         "corpus) — the teacher is frozen, so epochs 2+ skip "
                         "its forward entirely; 'off' recomputes every epoch")
    ap.add_argument("--eval-every", type=int, default=2)
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
        # fixed per-item prompt rows for the self-anchor term (block OUTPUT
        # rows inside the censored prompt; deterministic so the base-state
        # cache stays valid across epochs)
        pr = len(it["censored_ids"])
        rng = random.Random(f"anchor-{it['example_id']}-{args.seed}")
        k = min(args.anchor_rows, max(1, pr - 1))
        it["anchor_rows"] = sorted(rng.sample(range(1, pr), k))
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
        training_input="detached censored-student h_s[L-1] (own trajectory, "
                       "current adapters, nothing cached)",
        differentiable_output="student block L(detached h_s[L-1])",
        training_target="detached uncensored-teacher h_t[L] at answer rows",
        censorship="remove_view (privileged passage cut from student prompt)",
        end_to_end_student_training=False,
        backprop="layerwise: every block input detached by forward pre-hook; "
                 "cross-block gradient structurally unrepresentable",
        final_logit_training=False, frozen_vocabulary=True,
        output_kl_role="evaluation only, optimizer weight zero",
        note="v5 law owner-corrected 2026-07-26: backprop only layerwise")

    print(f"loading {args.model} (bf16, device_map=auto)", flush=True)
    full = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    full.config.use_cache = False
    # every ordinary Linear in the TEXT decoder blocks — attention AND MLP,
    # ALL layers: capacity everywhere, since we don't know where poetry
    # lives. Targets are EXACT module names enumerated inside the resolved
    # text stack, never bare suffixes: suffix matching also hits the vision
    # tower, whose projections are Gemma4ClippableLinear wrappers PEFT
    # cannot wrap (job 423052 failed exactly there). If a text projection
    # is itself wrapped, its inner plain .linear is targeted instead.
    proj_leaves = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")
    raw_stack, _ = resolve_stack(full)
    stack_prefix = next(n for n, m in full.named_modules() if m is raw_stack)
    target_names = []
    for mname, mod in full.named_modules():
        if not mname.startswith(stack_prefix + ".layers."):
            continue
        if mname.rsplit(".", 1)[-1] not in proj_leaves:
            continue
        if isinstance(mod, torch.nn.Linear):
            target_names.append(mname)
        elif isinstance(getattr(mod, "linear", None), torch.nn.Linear):
            target_names.append(mname + ".linear")
        else:
            raise SystemExit(
                f"GATE: text-decoder projection {mname} is a "
                f"{type(mod).__name__} with no plain .linear inside — "
                "refuse to guess")
    if not target_names:
        raise SystemExit("GATE: no LoRA targets found in the text decoder")
    print(f"lora targets: {len(target_names)} Linears under "
          f"{stack_prefix}.layers (vision tower excluded)", flush=True)
    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=target_names,
        bias="none", task_type="CAUSAL_LM")
    full = get_peft_model(full, lcfg)
    if args.grad_checkpoint:
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
    device = next(stack.parameters()).device
    n_layers = len(stack.layers)
    taps = LayerwiseTaps(stack.layers)

    # ---- OWNER APPROVAL GATE (2026-07-26), enforced structurally ----------
    # 1. No training on output logits: no training-loss term in this file
    #    projects a differentiable state through lm_head (vocab_mse uses the
    #    detached Gram W^T W as a frozen metric on hiddens). Output KL/CE
    #    exist ONLY inside evaluate() under torch.no_grad().
    # 2. Embedding / unembedding / final norm never move or train:
    for name, p in full.named_parameters():
        if p.requires_grad:
            assert "lora_" in name and ".layers." in name, \
                f"GATE: trainable parameter outside decoder-block LoRA: {name}"
    for mod in (lm_head, stack.embed_tokens,
                getattr(stack, "norm", None)):
        if mod is not None:
            for p in mod.parameters():
                assert not p.requires_grad, "GATE: vocabulary stack trainable"
    final_norm = getattr(stack, "norm", None)
    vocab_fp0 = (lm_head.weight.detach().float().sum().item(),
                 stack.embed_tokens.weight.detach().float().sum().item(),
                 final_norm.weight.detach().float().sum().item()
                 if final_norm is not None else 0.0)

    def vocab_tripwire():
        now = (lm_head.weight.detach().float().sum().item(),
               stack.embed_tokens.weight.detach().float().sum().item(),
               final_norm.weight.detach().float().sum().item()
               if final_norm is not None else 0.0)
        assert now == vocab_fp0, \
            f"FROZEN-VOCABULARY TRIPWIRE: {vocab_fp0} -> {now}"

    opt = torch.optim.AdamW((p for p in full.parameters() if p.requires_grad),
                            lr=args.lr)
    pad = tok.pad_token_id or 0

    baseline_argmax: dict = {}
    gate_mode, gate_val = "all", 0.0
    if args.layer_gate.startswith("topk:"):
        gate_mode, gate_val = "topk", int(args.layer_gate.split(":", 1)[1])
    elif args.layer_gate.startswith("minfrac:"):
        gate_mode, gate_val = "minfrac", float(args.layer_gate.split(":", 1)[1])
    elif args.layer_gate.startswith("surprise_ema:"):
        # owner concept (2026-07-26): surprise = loss EXCEEDING the layer's
        # own expectation. Each layer keeps an EMA of its loss; only layers
        # whose current loss > F x their EMA backprop this step —
        # prediction-error gating, the continuous-learning trigger in
        # miniature. Selection reads the PREVIOUS expectation; the EMA then
        # absorbs the new value for every layer.
        gate_mode, gate_val = "surprise_ema", float(
            args.layer_gate.split(":", 1)[1])
    elif args.layer_gate != "all":
        raise SystemExit(f"unknown --layer-gate {args.layer_gate}")
    loss_ema: list = [None] * n_layers

    def evaluate(epoch: int):
        # fixed seed on purpose: the SAME recall items every eval, so the
        # epoch-to-epoch recall curve measures learning, not sample churn
        rec = recall_eval(full, tok, items, device, stop_id, args.gen_batch,
                          args.max_new_tokens, args.recall_samples, args.seed)
        full.eval()
        arc = arc_eval(stack, lm_head, tok, device, args.arc_limit)
        # teacher-forced output metrics on a fixed probe slice — EVALUATION
        # ONLY (v4 convention: optimizer weight structurally zero). Includes
        # the argmax acceptance that sat frozen at 0.556 through all of v4.
        probe = items[:: max(1, len(items) // 128)][:128]
        agree = total = 0
        kl_sum = ce_sum = 0.0
        with torch.no_grad():
            for i in range(0, len(probe), args.micro_batch):
                b = probe[i:i + args.micro_batch]
                # student logits: ordinary censored forward, adapters ON
                s_ids, s_mask, s_spans = collate(b, "censored_ids", pad,
                                                 device)
                s_out = stack(input_ids=s_ids, attention_mask=s_mask,
                              use_cache=False)
                # teacher logits: adapters OFF, WITH passage
                with full.disable_adapter():
                    t_ids, t_mask, t_spans = collate(b, "prompt_ids", pad,
                                                     device)
                    t_out = stack(input_ids=t_ids, attention_mask=t_mask,
                                  use_cache=False)
                for j, it in enumerate(b):
                    ss, sl_ = s_spans[j]
                    ts, tl_ = t_spans[j]
                    sl = lm_head(s_out.last_hidden_state[j, ss:ss + sl_]).float()
                    tl = lm_head(t_out.last_hidden_state[j, ts:ts + tl_]).float()
                    tgt = torch.tensor(it["answer_ids"], device=sl.device)
                    agree += (sl.argmax(-1) == tgt).sum().item()
                    total += len(it["answer_ids"])
                    tp = torch.log_softmax(tl, -1)
                    sp = torch.log_softmax(sl, -1)
                    kl_sum += float((tp.exp() * (tp - sp)).sum())
                    ce_sum += float(torch.nn.functional.cross_entropy(
                        sl, tgt, reduction="sum"))
        acc = agree / max(1, total)
        kl = kl_sum / max(1, total)
        ce = ce_sum / max(1, total)
        print(f"eval e{epoch}: recall={rec} arc_easy={arc:.3f} "
              f"student_argmax={acc:.4f} KL_eval={kl:.4f} CE_eval={ce:.4f}",
              flush=True)
        if epoch == 0:
            baseline_argmax["e0"] = acc
        elif acc < 0.5 * baseline_argmax.get("e0", 0.0):
            print(f"WARNING: DESTRUCTION SIGNATURE — student_argmax {acc:.4f}"
                  f" < half of epoch-0 {baseline_argmax['e0']:.4f}; the "
                  "lr1e4 run died exactly like this (see "
                  "runs/trainv5_g31b_selfdistill_lr1e4_destroyed)",
                  flush=True)
        log("eval", epoch=epoch, recall=rec, arc_easy=arc,
            student_argmax_acceptance=acc, KL_eval_loss=kl, CE_eval_loss=ce,
            evaluation_only=True, optimizer_weight=0.0)
        return acc

    evaluate(0)  # epoch zero: identical conditions to every checkpoint

    loss_fn = LocalLoss(args.local_loss, lm_head)
    layer_dev = [next(stack.layers[l].parameters()).device
                 for l in range(n_layers)]
    teacher_cache: dict | None = ({} if args.teacher_cache == "cpu" else None)
    anchor_cache: dict = {}
    if teacher_cache is not None:
        width = stack.embed_tokens.weight.shape[1]
        est = n_ans * n_layers * width * 2
        est_anchor = (len(items) * args.anchor_rows * n_layers * width * 2
                      if args.anchor_weight else 0)
        print(f"teacher-cache: ~{est / 2**30:.1f} GiB + anchors "
              f"~{est_anchor / 2**30:.1f} GiB host RAM once filled",
              flush=True)

    def layerwise_isolation_cert(terms):
        """LAYERWISE TRIPWIRE (the project's name de pila, owner
        2026-07-26): backward ONE middle layer's local term in isolation
        and assert the ONLY parameters that received gradient are that
        block's LoRA. Detects any future edit that lets gradient cross a
        block boundary. Runs once, on the first batch."""
        m = n_layers // 2
        terms[m].backward(retain_graph=True)
        leaked = [n for n, p in full.named_parameters()
                  if p.grad is not None and f".layers.{m}." not in n]
        inside = [n for n, p in full.named_parameters()
                  if p.grad is not None and f".layers.{m}." in n]
        assert not leaked, f"LAYERWISE TRIPWIRE: gradient leaked to {leaked[:4]}"
        assert inside, "LAYERWISE TRIPWIRE: no gradient inside the probed block"
        opt.zero_grad(set_to_none=True)
        log("layerwise_certification", probed_layer=m,
            lora_params_with_grad=len(inside), leaks=0)
        print(f"layerwise isolation certified on block {m} "
              f"({len(inside)} LoRA tensors, 0 leaks)", flush=True)

    certified = False
    destroyed_evals = 0
    for epoch in range(1, args.epochs + 1):
        full.train()
        t0 = time.time()
        batches = bucketed_batches(items, args.micro_batch, args.seed, epoch,
                                   "censored_ids")
        epoch_loss = 0.0
        surprise_sum = [0.0] * n_layers   # per-BLOCK local loss (surprise)
        backprop_count = [0] * n_layers
        cache_hits = 0
        grad_norm_sum = 0.0
        grad_norm_n = 0
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(batches):
            # censored batch first: the anchor forward reuses it
            s_ids, s_mask, s_spans = collate(batch, "censored_ids", pad,
                                             device)
            # teacher: adapters OFF, WITH passage, no grad — per-layer
            # hidden targets at the answer's predictive rows. The teacher is
            # frozen, so its targets are cacheable: epochs 2+ skip this
            # forward entirely (the review's main speed win, ~1/3 of step
            # compute, and the teacher sequence is the LONG one).
            cached = (teacher_cache is not None
                      and all(it["example_id"] in teacher_cache
                              for it in batch))
            if cached:
                cache_hits += 1
                targets = [[teacher_cache[it["example_id"]][l]
                            .to(layer_dev[l], non_blocking=True)
                            for it in batch] for l in range(n_layers)]
                anchors = None
                if args.anchor_weight:
                    anchors = [[anchor_cache[it["example_id"]][l]
                                .to(layer_dev[l], non_blocking=True)
                                for it in batch] for l in range(n_layers)]
            else:
                with torch.no_grad(), full.disable_adapter():
                    t_ids, t_mask, t_spans = collate(batch, "prompt_ids",
                                                     pad, device)
                    t_out = stack(input_ids=t_ids, attention_mask=t_mask,
                                  output_hidden_states=True, use_cache=False)
                    # hidden_states[l] is the OUTPUT of block l-1 (index 0 =
                    # embeddings); block L's target is hidden_states[L+1].
                    # clone(): a bare slice is a VIEW that would retain all
                    # 61 full-sequence hidden tensors through the student
                    # pass (~8 GiB); clones keep only the answer rows.
                    targets = [[r.clone() for r in
                                slice_rows(t_out.hidden_states[l + 1],
                                           t_spans)]
                               for l in range(n_layers)]
                    del t_out
                    anchors = None
                    if args.anchor_weight:
                        # self-anchor targets: the adapters-OFF BASE model on
                        # the SAME censored sequence, at the fixed sampled
                        # prompt rows — "change nothing where there is no
                        # signal". Frozen like the teacher, hence cacheable.
                        b_out = stack(input_ids=s_ids,
                                      attention_mask=s_mask,
                                      output_hidden_states=True,
                                      use_cache=False)
                        anchors = [[b_out.hidden_states[l + 1][j,
                                    batch[j]["anchor_rows"]].clone()
                                    for j in range(len(batch))]
                                   for l in range(n_layers)]
                        del b_out
                if teacher_cache is not None:
                    for j, it in enumerate(batch):
                        teacher_cache[it["example_id"]] = [
                            targets[l][j].to("cpu")
                            for l in range(n_layers)]
                        if anchors is not None:
                            anchor_cache[it["example_id"]] = [
                                anchors[l][j].to("cpu")
                                for l in range(n_layers)]
            # student: adapters ON, passage REMOVED, one forward with every
            # block input detached (LayerwiseTaps) — the differentiable
            # output of each block roots only in that block's LoRA weights
            with taps.active():
                stack(input_ids=s_ids, attention_mask=s_mask, use_cache=False)
                block_out = list(taps.outputs)
                block_in = list(taps.inputs)
            for l in range(n_layers):  # name-de-pila guard, no sync
                assert not block_in[l].requires_grad, \
                    f"LAYERWISE TRIPWIRE: block {l} input carries grad"
            # ALL per-layer losses are computed every step (the forward has
            # already paid for them); the gate then decides which of the
            # independent per-block backwards to actually run.
            terms = []
            for l in range(n_layers):
                y_rows = slice_rows(block_out[l], s_spans)
                a_rows = slice_rows(block_in[l], s_spans)
                term = sum(loss_fn(y, t, a) for y, t, a in
                           zip(y_rows, targets[l], a_rows)) / len(batch)
                if anchors is not None:
                    # self-anchor: same block, same forward, PROMPT rows —
                    # output must stay near the adapters-off base states.
                    # The lr1e4 run proved answer-only constraints let the
                    # blocks destroy general function (arc -> chance).
                    anc = sum(
                        loss_fn(block_out[l][j, batch[j]["anchor_rows"]],
                                anchors[l][j],
                                block_in[l][j, batch[j]["anchor_rows"]])
                        for j in range(len(batch))) / len(batch)
                    term = term + args.anchor_weight * anc
                terms.append(term)
            # Selection needs the scalar values. Hot-loop law: never one
            # sync per layer — group terms by device and read each device
            # ONCE (<=4 syncs/step under the 4-GPU shard). NOTE for the
            # PPP4 port (owner 2026-07-26): in stage-parallel deployment
            # this global view costs a per-step all-gather of n_layers
            # scalars (or degrades to stage-local quotas / an epoch-frozen
            # threshold); backprop_count below measures whether selection
            # concentrates by depth — the load-balance risk of that port.
            by_dev: dict = {}
            for l, t in enumerate(terms):
                by_dev.setdefault(t.device, []).append(l)
            vals = [0.0] * n_layers
            for dev, idxs in by_dev.items():
                got = torch.stack([terms[l].detach() for l in idxs]).tolist()
                for l, v in zip(idxs, got):
                    vals[l] = v
            for l in range(n_layers):
                surprise_sum[l] += vals[l]
            if not certified:
                layerwise_isolation_cert(terms)
                certified = True
            if gate_mode == "topk":
                sel = sorted(range(n_layers),
                             key=lambda l: -vals[l])[:int(gate_val)]
            elif gate_mode == "minfrac":
                cut = max(vals) * gate_val
                sel = [l for l in range(n_layers) if vals[l] >= cut]
            elif gate_mode == "surprise_ema":
                sel = [l for l in range(n_layers)
                       if loss_ema[l] is None
                       or vals[l] > gate_val * loss_ema[l]]
            else:
                sel = list(range(n_layers))
            if gate_mode == "surprise_ema":
                for l in range(n_layers):
                    loss_ema[l] = (vals[l] if loss_ema[l] is None
                                   else 0.9 * loss_ema[l] + 0.1 * vals[l])
            if not sel:
                sel = [max(range(n_layers), key=lambda l: vals[l])]
            for l in sel:
                backprop_count[l] += 1
            # normalize by n_layers (not len(sel)): a selected layer's
            # gradient scale is then IDENTICAL under every gate mode, so
            # gated arms compare to 'all' at the same effective LR
            loss = torch.stack([terms[l].to(device)
                                for l in sel]).sum() / n_layers
            (loss / args.grad_accum).backward()
            epoch_loss += sum(vals) / n_layers
            del block_out, block_in, targets, terms
            if anchors is not None:
                del anchors
            if (bi + 1) % args.grad_accum == 0 or bi + 1 == len(batches):
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in full.parameters() if p.requires_grad], 1.0)
                grad_norm_sum += float(gn)
                grad_norm_n += 1
                opt.step()
                opt.zero_grad(set_to_none=True)

        profile = [s / len(batches) for s in surprise_sum]
        mean_loss = epoch_loss / len(batches)
        print(f"epoch {epoch}: local_loss={mean_loss:.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        with torch.no_grad():
            # python-float accumulation: parameters live on FOUR devices,
            # tensor sum() would cross devices (crashed 423279 at epoch 1)
            adapter_norm = sum(
                float(p.detach().float().norm()) ** 2
                for p in full.parameters() if p.requires_grad) ** 0.5
        log("epoch", epoch=epoch, loss=mean_loss, loss_kind=args.local_loss,
            seconds=time.time() - t0, surprise_profile=profile,
            layer_gate=args.layer_gate, backprop_count=backprop_count,
            teacher_cache_hit_frac=cache_hits / len(batches),
            mean_grad_norm=grad_norm_sum / max(1, grad_norm_n),
            adapter_l2_norm=adapter_norm,
            anchor_weight=args.anchor_weight)
        vocab_tripwire()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            acc = evaluate(epoch)
            full.save_pretrained(str(out_dir / f"checkpoint_e{epoch}"))
            if acc < 0.5 * baseline_argmax.get("e0", 0.0):
                destroyed_evals += 1
                if destroyed_evals >= 2:
                    log("aborted_destruction", epoch=epoch,
                        student_argmax=acc, e0=baseline_argmax["e0"])
                    print("ABORT: two consecutive destroyed evals — kill "
                          "doomed runs fast (owner rule); checkpoints and "
                          "metrics preserved", flush=True)
                    break
            else:
                destroyed_evals = 0

    full.save_pretrained(str(out_dir / "checkpoint"))
    log("done", epochs=args.epochs)
    print(f"run complete: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
