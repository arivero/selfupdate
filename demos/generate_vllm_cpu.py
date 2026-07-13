"""vLLM CPU baseline for the V5 workload — the reference the torch loop races.

Reads the same prompt JSONL as generate_torch_cpu.py.  Meant to run inside the
official vLLM CPU container (this cluster's glibc 2.28 rejects the
manylinux_2_34 +cpu wheel); see run_vllm_cpu.sh.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = [json.loads(x) for x in
            Path(args.prompts).read_text(encoding="utf-8").splitlines()]
    meta = rows[0] if rows and rows[0].get("meta") else {}
    prompts = [x for x in rows if not x.get("meta")]
    model_name = args.model or meta.get("model", "Qwen/Qwen3-0.6B")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    llm = LLM(model=model_name, dtype="bfloat16",
              max_model_len=args.max_model_len, disable_log_stats=True)
    load_seconds = time.perf_counter() - t_load

    t_gen = time.perf_counter()
    generated = llm.generate(
        [{"prompt_token_ids": x["ids"]} for x in prompts],
        [SamplingParams(temperature=0.0, top_p=1.0, max_tokens=x["budget"],
                        stop_token_ids=[x["stop_id"]], ignore_eos=False)
         for x in prompts],
        use_tqdm=False)
    gen_seconds = time.perf_counter() - t_gen

    results, total = [], 0
    for item, result in zip(prompts, generated):
        token_ids = list(result.outputs[0].token_ids)
        hard_cut = not token_ids or token_ids[-1] != item["stop_id"]
        if hard_cut:
            token_ids = token_ids + [item["stop_id"]]
        total += len(token_ids)
        results.append({"example_id": item["example_id"],
                        "gen_tokens": len(token_ids), "token_ids": token_ids,
                        "answer_tokens": len(token_ids) - 1,
                        "hard_cut": hard_cut,
                        "answer_text": result.outputs[0].text})
    (out_dir / "responses.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results),
        encoding="utf-8")
    summary = {
        "engine": "vllm_cpu", "model": model_name,
        "prompts": len(prompts), "load_seconds": round(load_seconds, 2),
        "generate_seconds": round(gen_seconds, 2), "gen_tokens": total,
        "tokens_per_second": round(total / gen_seconds, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
