#!/usr/bin/env python
"""Layer-selective censorship of the TEACHER (owner design, 2026-08-20).

The attention probe (v5_teacher_attn_probe.py) maps where the teacher
LOOKS at the passage; this probe is the causal ablation: block attention
to the passage span at chosen layers and measure the teacher's own
answer quality (teacher-forced CE + argmax acceptance on its own
answers). Conditions:

  baseline   no mask               -> must match the intact teacher
  only_S     passage visible ONLY at retrieval set S -> sufficiency
  no_S       passage blocked AT S, visible elsewhere -> necessity
  none       passage blocked at ALL layers -> tripwire: must degrade to
             censored-level CE; if it matches baseline, the hooks are
             silently inert and the run FAILS instead of reporting a
             fake sufficiency.

S is derived from runs/v5_teacher_attn/attn_by_layer.json (layers whose
far-item passage mass >= 0.2 * max), overridable via S_LAYERS env
(comma-separated). Masking = forward pre-hook on selected layers'
self_attn adding dtype-min at the passage key columns of the 4D mask;
asserts the mask tensor exists on the first item (eager attention
materializes it; a None mask means the hook cannot act -> fail loudly).
CPU fat node; ~4 conditions x ~40 items, single forwards, no generation.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "trainv5", ROOT / "scripts" / "trainv5.py")
trainv5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainv5)

# eager forwards on CPU cost ~250s at T~500 and grow quadratically; the
# 8x5-item 4-condition version paced ~10h vs an 8h wall (killed 439017)
PER_CORPUS = int(os.environ.get("CENSOR_PER_CORPUS", "4"))
MAX_SEQ = int(os.environ.get("CENSOR_MAX_SEQ", "800"))


def derive_S(attn_json: Path) -> list[int]:
    env = os.environ.get("S_LAYERS", "").strip()
    if env:
        return sorted(int(x) for x in env.split(","))
    d = json.loads(attn_json.read_text())
    # pick the bucket that actually has items: an empty bucket's array is
    # all zeros but truthy, and 0.2*max(0)=0 would select ALL layers
    # (bug caught 2026-08-21 when every corpus item landed "near")
    counts = d.get("items", {})
    bucket = "far" if counts.get("far") else "near"
    if not counts.get(bucket):
        raise SystemExit("attn_by_layer.json has no items in any bucket")
    mass = d.get(f"{bucket}_passage_mass_by_layer")
    if not mass:
        raise SystemExit("attn_by_layer.json lacks passage-mass arrays")
    # TOP-K, not a threshold: in the near regime every sliding layer sees
    # some passage mass, and 0.2*max selected 59/60 layers (degenerate
    # sufficiency test, caught live 2026-08-21 on job 439017). The
    # question is whether a SMALL set suffices.
    k = int(os.environ.get("S_TOP_K", "8"))
    S = sorted(sorted(range(len(mass)), key=lambda l: -mass[l])[:k])
    print(f"deriving S from '{bucket}' bucket (n={counts.get(bucket)}): "
          f"top-{k} by passage mass = {S}", flush=True)
    return S


def main() -> None:
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(int(os.environ.get("SELFUPDATE_CPU_THREADS", "56")))
    model_id = "google/gemma-4-31B-it"
    S = derive_S(ROOT / "runs/v5_teacher_attn/attn_by_layer.json")
    print(f"retrieval set S = {S}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    items, _stop = trainv5.build_items(
        ROOT / "data/combined/examples_v5rs_window.jsonl",
        ROOT / "runs/vllm_h100/gemma4_31b_it/responses_bs256.jsonl",
        tok, limit=0)
    rng = random.Random(17)
    by: dict[str, list[dict]] = {}
    for it in items:
        by.setdefault(it["corpus"], []).append(it)
    chosen = []
    for _, grp in sorted(by.items()):
        chosen += rng.sample(grp, min(PER_CORPUS, len(grp)))
    chosen = [it for it in chosen
              if len(it["prompt_ids"]) + len(it["answer_ids"]) <= MAX_SEQ]
    print(f"{len(chosen)} items", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()
    stack, lm_head = trainv5.resolve_stack(model)
    n_layers = len(stack.layers)
    conditions = {"baseline": [], "only_S": [l for l in range(n_layers)
                                             if l not in S],
                  "no_S": list(S), "none": list(range(n_layers))}

    state = {"span": (0, 0), "seen_mask": False}

    def make_hook():
        def hook(module, args, kwargs):
            m = kwargs.get("attention_mask")
            if m is None and args:
                return None  # positional path not expected; assert below
            if m is None:
                return None
            state["seen_mask"] = True
            p0, p1 = state["span"]
            m = m.clone()
            m[..., :, p0:p1] = torch.finfo(m.dtype).min
            kwargs["attention_mask"] = m
            return (args, kwargs)
        return hook

    softcap = getattr(stack.config, "final_logit_softcapping", None)
    results: dict[str, dict] = {}
    t0 = time.time()
    for cond, blocked in conditions.items():
        handles = [stack.layers[l].self_attn.register_forward_pre_hook(
            make_hook(), with_kwargs=True) for l in blocked]
        state["seen_mask"] = False
        ce_sum, ok_sum, ntok = 0.0, 0.0, 0
        with torch.no_grad():
            for it in chosen:
                seq = it["prompt_ids"] + it["answer_ids"]
                state["span"] = (it["cut_at"], it["cut_at"] + it["pos_gap"])
                ids = torch.tensor([seq], dtype=torch.long)
                hs = stack(input_ids=ids, use_cache=False).last_hidden_state
                a0 = len(it["prompt_ids"])
                lg = trainv5.head_logits(
                    lm_head, hs[0, a0 - 1:len(seq) - 1], softcap).float()
                tgt = torch.tensor(it["answer_ids"], dtype=torch.long)
                ce_sum += torch.nn.functional.cross_entropy(
                    lg, tgt, reduction="sum").item()
                ok_sum += (lg.argmax(-1) == tgt).float().sum().item()
                ntok += len(tgt)
        for h in handles:
            h.remove()
        if blocked and not state["seen_mask"]:
            raise SystemExit(f"GATE: condition {cond} hooks never saw an "
                             "attention mask — masking is inert")
        results[cond] = {"blocked_layers": blocked and
                         (blocked if len(blocked) < 55 else "ALL"),
                        "CE": round(ce_sum / ntok, 4),
                        "argmax_acc": round(ok_sum / ntok, 4),
                        "tokens": ntok}
        print(f"{cond}: CE={results[cond]['CE']} "
              f"acc={results[cond]['argmax_acc']} "
              f"({time.time() - t0:.0f}s)", flush=True)
        # per-condition flush: a wall-time kill keeps completed conditions
        out = ROOT / "runs/v5_teacher_attn/layer_censor.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"model": model_id, "S": S,
                                   "final": False,
                                   "results": results}, indent=1))
        tmp.replace(out)

    if results["none"]["CE"] < 2 * results["baseline"]["CE"]:
        raise SystemExit(
            "GATE: blocking the passage at ALL layers barely moved CE "
            f"({results['baseline']['CE']} -> {results['none']['CE']}) — "
            "masking is not reaching the passage span; results invalid")
    out = ROOT / "runs/v5_teacher_attn/layer_censor.json"
    out.write_text(json.dumps({"model": model_id, "S": S, "final": True,
                               "results": results}, indent=1))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
