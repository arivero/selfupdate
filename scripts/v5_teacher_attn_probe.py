#!/usr/bin/env python
"""Teacher attention-by-layer probe (owner question, 2026-08-20).

"For each answer token, read the top-5 attention: WHEN (at which layers)
is the teacher looking at the passage text?" — maps where context becomes
content in the residual stream. Successor design B (mid-stack placement)
gets its target layers empirically from this instead of by ROME/MEMIT
analogy.

Architectural prior (config.json, checked 2026-08-20): Gemma-4-31B runs
sliding_window=1024 attention on 50/60 layers; ONLY layers
5,11,17,23,29,35,41,47,53,59 have full attention and can see a passage
further than 1024 tokens behind the query. Note the rhyme with v5w2:
topk_abs collapsed onto L54 (right after deep global L53) and the ~0-loss
anomaly is L59 (the last global layer).

CPU-only (agustina_fat, ~500G RAM): one teacher-forced forward per item
with output_attentions, no generation, no GPU queue. Per layer and item
we keep only answer-row query stats: attention mass on the passage span,
and the fraction of top-5 attended positions that fall inside it —
megabytes, not the L*H*T^2 tensors.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # repo convention: pin our own tree

spec = importlib.util.spec_from_file_location(
    "trainv5", ROOT / "scripts" / "trainv5.py")
trainv5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainv5)

PER_CORPUS = 12
MAX_SEQ = 1600  # bounds the 60*H*T^2 attention RAM and CPU time


def main() -> None:
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(int(os.environ.get("SELFUPDATE_CPU_THREADS", "56")))
    model_id = "google/gemma-4-31B-it"
    tok = AutoTokenizer.from_pretrained(model_id)
    items, _stop = trainv5.build_items(
        ROOT / "data/combined/examples_v5rs_window.jsonl",
        ROOT / "runs/vllm_h100/gemma4_31b_it/responses_bs256.jsonl",
        tok, limit=0)

    rng = random.Random(17)  # same seeded sample as every campaign eval
    by: dict[str, list[dict]] = {}
    for it in items:
        by.setdefault(it["corpus"], []).append(it)
    chosen = []
    for _, grp in sorted(by.items()):
        chosen += rng.sample(grp, min(PER_CORPUS, len(grp)))
    chosen = [it for it in chosen
              if len(it["prompt_ids"]) + len(it["answer_ids"]) <= MAX_SEQ]
    print(f"{len(chosen)} items (<= {MAX_SEQ} tokens)", flush=True)

    print(f"loading {model_id} on CPU (bf16, eager attention)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        attn_implementation="eager")
    model.eval()
    n_layers = model.config.text_config.num_hidden_layers \
        if hasattr(model.config, "text_config") \
        else model.config.num_hidden_layers
    layer_types = getattr(
        getattr(model.config, "text_config", model.config),
        "layer_types", None)

    # accumulators: per layer, split by passage->answer distance vs window
    acc = {k: {"mass": [0.0] * n_layers, "top5": [0.0] * n_layers, "n": 0}
           for k in ("near", "far")}
    t0 = time.time()
    with torch.no_grad():
        for idx, it in enumerate(chosen):
            seq = it["prompt_ids"] + it["answer_ids"]
            p0, plen = it["cut_at"], it["pos_gap"]
            a0 = len(it["prompt_ids"])
            dist = a0 - (p0 + plen)  # passage end -> first answer token
            key = "near" if dist <= 1024 else "far"
            ids = torch.tensor([seq], dtype=torch.long)
            out = model(input_ids=ids, output_attentions=True,
                        use_cache=False)
            for l, att in enumerate(out.attentions):
                a = att[0, :, a0:, :].float()      # [H, A, T]
                hm = a.mean(0)                     # head-mean [A, T]
                mass = hm[:, p0:p0 + plen].sum(-1).mean().item()
                k = min(5, hm.shape[-1])
                top = hm.topk(k, dim=-1).indices   # [A, k]
                inp = ((top >= p0) & (top < p0 + plen)).float().mean().item()
                acc[key]["mass"][l] += mass
                acc[key]["top5"][l] += inp
            acc[key]["n"] += 1
            del out
            print(f"item {idx + 1}/{len(chosen)} ({key}, dist={dist}, "
                  f"T={len(seq)}) {time.time() - t0:.0f}s", flush=True)

    result = {"model": model_id, "per_corpus": PER_CORPUS,
              "max_seq": MAX_SEQ, "layer_types": layer_types,
              "sliding_window": 1024, "items": {k: acc[k]["n"] for k in acc}}
    for k in acc:
        n = max(acc[k]["n"], 1)
        result[f"{k}_passage_mass_by_layer"] = \
            [round(x / n, 4) for x in acc[k]["mass"]]
        result[f"{k}_top5_in_passage_by_layer"] = \
            [round(x / n, 4) for x in acc[k]["top5"]]
    out_dir = ROOT / "runs/v5_teacher_attn"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attn_by_layer.json").write_text(json.dumps(result, indent=1))
    print("wrote", out_dir / "attn_by_layer.json", flush=True)
    for k in acc:
        if not acc[k]["n"]:
            continue
        m = result[f"{k}_passage_mass_by_layer"]
        top = sorted(range(n_layers), key=lambda l: -m[l])[:8]
        print(f"{k} (n={acc[k]['n']}): top passage-mass layers: "
              f"{[(l, m[l]) for l in top]}", flush=True)


if __name__ == "__main__":
    main()
