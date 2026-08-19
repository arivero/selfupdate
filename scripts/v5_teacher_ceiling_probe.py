#!/usr/bin/env python
"""Teacher-ceiling probe for the v5 natural-position recall metric.

Owner question (2026-08-19): "I lack information of the quality of teacher
uncensored in this current code, so the 0.187-0.192 is a bit uninformative."

The campaign recall metric (trainv5.recall_eval) greedy-generates from the
CENSORED prompt and scores word-LCS against `answer_text` — which is the
teacher's own vLLM generation from the UNCENSORED prompt. Two calibration
anchors already exist without a GPU (documented in WEEKEND_HANDOFF.md):

  - artifact word_acc (teacher answer vs ORIGINAL text, same reference-word
    LCS family): mach 0.9915 / quij 0.9839, recite-rate ~0.98 (n=1822);
  - epoch-0 rows in every run: censored base ~0.17 mach / ~0.19 quij.

What has never been measured is the HF-side reproduction: does THIS eval
code path (HF greedy, max_new_tokens=96, eos=stop_id, left-pad batches,
bf16 device_map=auto) regenerate the vLLM answers when given the SAME
uncensored prompt? That number is the true ceiling of the metric scale;
vLLM-vs-HF decode divergence is the only thing that could pull it below
~1.0. This probe measures it by reusing trainv5's own build_items and
recall_eval verbatim (imported, not copied) with items whose
`censored_ids` are swapped for `prompt_ids` — the one-line difference
between the student condition and the teacher condition. It also re-runs
the censored control so both ends of the scale come from one process on
one node. Read-only with respect to training: no adapters, no writes
outside runs/v5_teacher_ceiling/.

Sampling matches the campaign evals exactly (sample_per_corpus=24,
seed=17, same corpus grouping incl. the q1..q4 chapter split), so the
ceiling row is item-for-item comparable with every eval row in every
v5 run's metrics.jsonl.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # repo convention: pin our own tree

spec = importlib.util.spec_from_file_location(
    "trainv5", ROOT / "scripts" / "trainv5.py")
trainv5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainv5)  # module-level is light: no torch, no main


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "google/gemma-4-31B-it"
    examples = ROOT / "data/combined/examples_v5rs_window.jsonl"
    responses = ROOT / "runs/vllm_h100/gemma4_31b_it/responses_bs256.jsonl"
    out_dir = ROOT / "runs/v5_teacher_ceiling"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    items, stop_id = trainv5.build_items(examples, responses, tok, limit=0)

    print(f"loading {model_id} (bf16, device_map=auto)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    # campaign eval knobs, verbatim (trainv5 argparse defaults)
    gen_batch, max_new, samples, seed = 16, 96, 24, 17

    t0 = time.time()
    censored = trainv5.recall_eval(
        model, tok, items, device, stop_id, gen_batch, max_new, samples, seed)
    print(f"censored control done in {time.time() - t0:.0f}s: "
          f"{censored['mean']}", flush=True)

    teacher_items = [{**it, "censored_ids": it["prompt_ids"]} for it in items]
    t0 = time.time()
    teacher = trainv5.recall_eval(
        model, tok, teacher_items, device, stop_id, gen_batch, max_new,
        samples, seed)
    print(f"teacher (uncensored) done in {time.time() - t0:.0f}s: "
          f"{teacher['mean']}", flush=True)

    result = {
        "model": model_id,
        "metric": "word-LCS vs answer_text (teacher's own vLLM answer), "
                  "recall_eval verbatim, greedy, max_new=96",
        "sampling": {"per_corpus": samples, "seed": seed},
        "teacher_uncensored": teacher["mean"],
        "teacher_recite": teacher["recite"],
        "censored_base": censored["mean"],
        "censored_recite": censored["recite"],
        "teacher_items": teacher["items"],
        "censored_items": censored["items"],
        # recitation-zero question (owner, Aug 19): keep the generations so
        # teacher-side reproduction failures (decode/formatting/early-stop)
        # are directly inspectable next to the student runs' recall_texts
        "teacher_texts": teacher.get("texts", {}),
        "censored_texts": censored.get("texts", {}),
        "artifact_word_acc_vs_original": {
            "mach": 0.9915, "quij": 0.9839,
            "note": "vLLM answer vs original text, from responses artifact; "
                    "computed 2026-08-19, n=1822"},
    }
    (out_dir / "ceiling.json").write_text(json.dumps(result, indent=1))
    print("wrote", out_dir / "ceiling.json", flush=True)


if __name__ == "__main__":
    main()
