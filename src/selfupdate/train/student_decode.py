"""Sharded autoregressive decode with a per-stage STUDENT KV cache.

This is the deployment/eval regime, opposite in every way to training:

  ``_FrozenKV`` (online_v4)   TEACHER K/V, frozen, censored, no gradient,
                              recorded once over the full sequence. TRAINING.
  ``StudentKVCache`` (here)   The STUDENT's OWN K/V, plain-causal, GROWING one
                              position per generated token, free generation.
                              EVAL / serving / speculative lookahead.

Why it never existed before: v4 training runs ``use_cache=False`` and each
block attends to the frozen teacher context. Autoregressive recall needs the
student to attend to ITS OWN generated prefix, so every stage keeps a growing
per-owned-layer cache and each decode step forwards ONE token in O(1) — without
it, every step would re-run the whole prefix through all stages (O(T**2)).

Communication topology (owner, 2026-07-19). Hiddens flow FORWARD
0->1->...->N-1, but each sampled token must LOOP BACK from the last stage to
stage 0 to become the next input. In PPP<=4 (single node) that loop-back is
node-local shm/IPC; in PPP8 (two nodes) it is a SECOND cross-node hop per token
(the forward mid-boundary crossing PLUS the last->0 loop-back) — exactly where
InfiniBand vs Lustre-file latency bites. The ``_RelayFiles`` envelope is
transport-agnostic (``SELFUPDATE_V4_RELAY_ROOT`` = shm or an IB-backed FS), so
the same code serves both PPP<=4 and PPP8; only the exchange directory moves.

Reusability: nothing here is recall-specific. ``StudentKVCache`` and
``sharded_generate`` are a general sharded-generation primitive; ``reset`` /
``crop`` are provided for a future serving loop or speculative-decode use.

Scope of THIS version: plain-rope decoder families (Qwen3.x / Qwen3.5 MoE —
the 0.6B dev target and the 122B/397B payloads). Gemma-4 rope bundles and the
DeepSeek MLA (``hc_mult`` inter-block state, per-type rope dict) need the same
extensions their training path already carries; guarded with an explicit
error until wired, so a wrong family fails loudly rather than silently.
"""

from __future__ import annotations

import torch

from .blocks import NO_PREPARED_ATTENTION_MASK


class StudentKVCache:
    """Per-stage growing STUDENT causal K/V cache, duck-typed to the
    transformers Cache protocol the decoder blocks call.

    Keyed by GLOBAL ``layer_idx`` (a dict, not a list) so a stage that owns an
    arbitrary contiguous range — e.g. layers 15..21 of a 60-layer model — grows
    each of its blocks independently. A list-backed ``DynamicCache`` assumes
    layers are appended 0..N and would mis-index a mid-pipeline shard.
    """

    def __init__(self) -> None:
        self.keys: dict[int, torch.Tensor] = {}    # layer_idx -> [B,n_kv,T,hd]
        self.values: dict[int, torch.Tensor] = {}

    # -- transformers Cache protocol --------------------------------------
    def update(self, key_states, value_states, layer_idx=None,
               cache_kwargs=None):
        prev = self.keys.get(layer_idx)
        if prev is None:
            self.keys[layer_idx] = key_states
            self.values[layer_idx] = value_states
        else:
            self.keys[layer_idx] = torch.cat([prev, key_states], dim=2)
            self.values[layer_idx] = torch.cat(
                [self.values[layer_idx], value_states], dim=2)
        return self.keys[layer_idx], self.values[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        t = self.keys.get(layer_idx)
        if t is None and self.keys:            # any owned layer's length
            t = next(iter(self.keys.values()))
        return 0 if t is None else int(t.shape[2])

    # version-compat surface: blocks only need length + growth; these keep
    # both old and new transformers happy without constraining the cache.
    def get_max_length(self):
        return None

    def get_max_cache_shape(self):
        return None

    def get_usable_length(self, new_seq_length, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    # -- reusable helpers (serving / speculative futures) -----------------
    def reset(self) -> None:
        self.keys.clear()
        self.values.clear()

    def crop(self, max_length: int) -> None:
        for k in list(self.keys):
            self.keys[k] = self.keys[k][:, :, :max_length].contiguous()
            self.values[k] = self.values[k][:, :, :max_length].contiguous()


def _prefill_mask(pad_keep: torch.Tensor | None, T: int, dtype, device):
    """[B,1,T,T] additive mask: causal AND (if given) pad-column masked.

    ``pad_keep`` is [B,T] with 1 for real prompt tokens, 0 for left-pad. q>1
    prefill MUST carry causal masking (mask-free is valid only for q=1)."""
    neg = torch.finfo(dtype).min
    causal = torch.triu(torch.full((T, T), neg, dtype=dtype, device=device),
                        diagonal=1)[None, None]        # [1,1,T,T], broadcasts
    if pad_keep is None:
        return causal
    colmask = torch.where(pad_keep.bool()[:, None, None, :],
                          torch.zeros((), dtype=dtype, device=device),
                          torch.full((), neg, dtype=dtype, device=device))
    return causal + colmask                            # [B,1,T,T]


@torch.no_grad()
def sharded_generate(stack, relay, *, stage: int, n_stages: int, owned,
                     prompt_ids: torch.Tensor | None,
                     prompt_keep: torch.Tensor | None,
                     max_new_tokens: int, device, epoch: int, tag: str):
    """Greedy sharded autoregressive decode across layer-sharded stages.

    Returns generated token ids ``[B, max_new_tokens]`` on the LAST stage,
    ``None`` on every other stage (the last stage writes them to the relay for
    stage 0 / the scorer). Fixed-step (finished sequences keep decoding and are
    trimmed at scoring) to avoid a cross-stage stop-propagation ring.

    ``prompt_ids`` / ``prompt_keep`` ([B,T], left-padded) are read only on the
    FIRST stage; downstream stages receive the prefill hidden over the relay.
    """
    if getattr(stack, "hc_mult", 1) > 1 or stack.rotary_emb is None \
            or getattr(stack, "rotary_needs_layer_type", False):
        raise NotImplementedError(
            "sharded_generate currently supports plain-rope decoders only "
            "(Qwen3.x/3.5); gemma-4 bundle and deepseek MLA rope need wiring")

    is_first = stage == 0
    is_last = stage == n_stages - 1
    cache = StudentKVCache()

    def run_owned(h, pos, mask):
        for L in owned:
            h = stack.run_block(L, h, stack.rope(h, pos), position_ids=pos,
                                past_key_values=cache, use_cache=True,
                                prepared_attention_mask=mask)
        return h

    def send(name, tens, to):
        relay.write(relay.path(epoch, f"{tag}_{name}"),
                    {k: v.cpu() for k, v in tens.items()},
                    stage=stage, epoch=epoch, to_stage=to)

    def recv(name, as_stage):
        p = relay.wait(relay.path(epoch, f"{tag}_{name}"))
        env = relay.read(p, expect_epoch=epoch, as_stage=as_stage)
        p.unlink(missing_ok=True)                      # consumed; keep exchange lean
        return {k: v.to(device) for k, v in env.items()}

    def logits_of(h):
        return stack.lm_head(stack.loss_view(stack.n_layers, h))[:, -1]  # [B,V]

    # ---- PREFILL ----
    if is_first:
        ids = prompt_ids.to(device)
        B, T = ids.shape
        h = stack.embed(ids)
    else:
        env = recv(f"pf_to{stage}", stage)
        h = env["h"]
        B, T = h.shape[0], h.shape[1]
    pos = torch.arange(T, device=device)[None].expand(B, -1)
    keep = prompt_keep.to(device) if (is_first and prompt_keep is not None) \
        else None
    h = run_owned(h, pos, _prefill_mask(keep, T, h.dtype, device))

    generated = None
    if is_last:
        tok0 = logits_of(h).argmax(-1)                 # [B] token[0]
        generated = [tok0]
        send("loop0", {"tok": tok0}, to=0)
    else:
        send(f"pf_to{stage + 1}", {"h": h}, to=stage + 1)

    cur = T
    # ---- DECODE: produce max_new_tokens-1 more tokens ----
    for i in range(max_new_tokens - 1):
        if is_first:
            tok = recv(f"loop{i}", 0)["tok"]           # [B] token[i]
            h = stack.embed(tok[:, None])              # [B,1,H]
        else:
            h = recv(f"d{i}_to{stage}", stage)["h"]    # [B,1,H]
        pos = torch.full((h.shape[0], 1), cur, device=device)
        h = run_owned(h, pos, NO_PREPARED_ATTENTION_MASK)   # q=1: mask-free
        if is_last:
            tok_next = logits_of(h).argmax(-1)         # token[i+1]
            generated.append(tok_next)
            send(f"loop{i + 1}", {"tok": tok_next}, to=0)
        else:
            send(f"d{i}_to{stage + 1}", {"h": h}, to=stage + 1)
        cur += 1

    return torch.stack(generated, dim=1) if is_last else None


@torch.no_grad()
def sharded_recall(stack, relay, corpus_paths, *, stage: int, n_stages: int,
                   owned, tok, device, epoch: int, log,
                   n_per_task: int = 8, max_new_tokens: int = 48,
                   gen_batch: int = 8) -> dict | None:
    """No-reload recall (Machado/Quijote) over the RESIDENT sharded student.

    Collective across all stages. ``build_tasks`` is deterministic (seed 17,
    same corpus), so every stage computes the SAME prompts/references and
    iterates the SAME batches — stage 0 tokenizes and feeds prompt_ids, the
    downstream stages receive the prefill hidden over the relay, and the LAST
    stage decodes + scores word-accuracy. Nothing is reloaded; the eval runs on
    the exact weights (+ LoRA) that trained, in bf16, via ``sharded_generate``.

    Returns per-corpus overall word_acc on the last stage; None elsewhere.
    """
    from ..eval.tasks import build_tasks, QUESTIONS, score
    is_last = stage == n_stages - 1
    results: dict[str, float] = {}
    for name, path in corpus_paths:
        items = build_tasks(path, seed=17, n_per_task=n_per_task)
        prompts = [tok.apply_chat_template(
            [{"role": "user",
              "content": QUESTIONS[it["kind"]].format(x=it["x"], n=it["n"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for it in items]
        refs = [it["reference"] for it in items]
        gen_texts: list[str] = []
        for b0 in range(0, len(prompts), gen_batch):
            batch = prompts[b0:b0 + gen_batch]
            if stage == 0:
                enc = tok(batch, return_tensors="pt", padding=True,
                          padding_side="left")
                prompt_ids = enc["input_ids"].to(device)
                keep = enc["attention_mask"].to(device)
            else:
                prompt_ids = keep = None
            out = sharded_generate(
                stack, relay, stage=stage, n_stages=n_stages, owned=owned,
                prompt_ids=prompt_ids, prompt_keep=keep,
                max_new_tokens=max_new_tokens, device=device, epoch=epoch,
                tag=f"recall_{name}_{b0}")
            if is_last and out is not None:
                gen_texts.extend(tok.decode(row, skip_special_tokens=True)
                                 for row in out)
        if is_last:
            accs = [score(r, a)["word_acc"] for r, a in zip(refs, gen_texts)]
            results[name] = sum(accs) / max(len(accs), 1)
    if is_last:
        log.log(kind="eval", epoch=epoch, phase="inpipeline_recall",
                recall={k: {"overall_word_acc": v} for k, v in results.items()},
                overall_word_acc=sum(results.values()) / max(len(results), 1),
                recall_items_per_task=n_per_task, no_reload=True,
                resident_student=True)
        return results
    return None
