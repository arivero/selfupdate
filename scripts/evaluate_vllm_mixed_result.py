#!/usr/bin/env python3
"""Validate one mixed-budget result and compare it with the documented row."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_teacher_cache import _corpus_texts, _recitation_stats  # noqa: E402

# Exact artifact behind the existing row in docs/vllm_generation_benchmark.md.
REFERENCE_DIRS = {
    "Qwen/Qwen3-0.6B": "Qwen3-0.6B_vllm025_graph_full_h100",
    "Qwen/Qwen3-1.7B": "Qwen3-1.7B_vllm025_graph_full_h100",
    "Qwen/Qwen3-4B": "Qwen3-4B_vllm025_graph_full_h100",
    "Qwen/Qwen3-8B": "Qwen3-8B_vllm025_graph_full_h100",
    "Qwen/Qwen3-14B": "Qwen3-14B_vllm025_graph_full_h100",
    "Qwen/Qwen3-32B": "Qwen3-32B_vllm025_pp2_graph_full_h100",
    "Qwen/Qwen3.5-0.8B": "Qwen3.5-0.8B_vllm025_graph_full_h100",
    "Qwen/Qwen3.5-2B": "Qwen3.5-2B_vllm025_graph_full_h100",
    "Qwen/Qwen3.5-4B": "Qwen3.5-4B_vllm025_graph_full_h100",
    "Qwen/Qwen3.5-9B": "Qwen3.5-9B_vllm025_graph_full_h100",
    "Qwen/Qwen3.6-27B": "Qwen3.6-27B_vllm025_pp2_graph_full_h100",
    "Qwen/Qwen3.6-35B-A3B": "Qwen3.6-35B-A3B_vllm025_pp2_graph_full_h100",
    "google/gemma-4-26B-A4B-it": "gemma-4-26B-A4B-it_vllm025_pp2_graph_full_h100",
    "google/gemma-4-31B-it": "gemma-4-31B-it_vllm025_pp2_graph_full_h100",
    "microsoft/Phi-4": "Phi-4_vllm025_graph_full_h100",
    "openai/gpt-oss-20b": "gpt-oss-20b_vllm025_graph_full_h100",
    "openai/gpt-oss-120b": "gpt-oss-120b_vllm025_pp2_graph_full_h100",
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2":
        "NVIDIA-Nemotron-Nano-9B-v2_vllm025_graph_full_h100",
    "meta-llama/Meta-Llama-3.1-8B-Instruct":
        "Llama-3.1-8B-Instruct_vllm025_graph_full_h100",
    "mistralai/Mistral-7B-Instruct-v0.1":
        "Mistral-7B-Instruct-v0.1_vllm025_graph_full_h100",
    "BSC-LT/ALIA-40b-fc-2606": "ALIA-40b-fc-2606_vllm025_pp2_graph_full_h100",
}


def one_result(summary: dict) -> dict:
    rows = summary.get("results", [])
    if len(rows) != 1 or rows[0].get("batch_size") != 64:
        raise ValueError("expected exactly one batch-64 result")
    return rows[0]


def old_eval(responses: list[dict], records: dict, corpora: dict) -> dict:
    word_scores = []
    containment_scores = []
    for response in responses:
        stats = _recitation_stats(
            records[response["example_id"]], response["answer_text"], corpora)
        if "word_acc" in stats:
            word_scores.append(stats["word_acc"])
        if "containment" in stats:
            containment_scores.append(stats["containment"])
    return {
        "hard_cut_fraction": sum(row["hard_cut"] for row in responses) / len(responses),
        "mean_word_acc": sum(word_scores) / len(word_scores),
        "word_acc_examples": len(word_scores),
        "mean_containment": sum(containment_scores) / len(containment_scores),
        "containment_examples": len(containment_scores),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--reference-dir", type=Path, default=None)
    args = ap.parse_args()

    result_dir = args.result_dir.resolve()
    summary = json.loads((result_dir / "summary.json").read_text())
    result = one_result(summary)
    responses = [json.loads(line) for line in
                 (result_dir / "responses_bs64.jsonl").read_text().splitlines()]
    ids = [row["example_id"] for row in responses]
    prompt_id_rows = sum("prompt_token_ids" in row for row in responses)
    integrity = {
        "expected_examples": 2071,
        "response_rows": len(responses),
        "unique_example_ids": len(set(ids)),
        "all_token_ids_nonempty": all(row.get("token_ids") for row in responses),
        "all_token_lengths_match": all(
            len(row.get("token_ids", [])) == row.get("gen_tokens")
            for row in responses),
        "prompt_token_id_rows": prompt_id_rows,
        "prompt_token_ids_complete_or_absent": prompt_id_rows in (0, len(responses)),
        "all_present_prompt_token_ids_nonempty": all(
            row.get("prompt_token_ids")
            for row in responses if "prompt_token_ids" in row),
    }
    if (summary.get("prompt_count") != 2071
            or result.get("examples") != 2071
            or integrity["response_rows"] != 2071
            or integrity["unique_example_ids"] != 2071
            or not integrity["all_token_ids_nonempty"]
            or not integrity["all_token_lengths_match"]
            or not integrity["prompt_token_ids_complete_or_absent"]
            or not integrity["all_present_prompt_token_ids_nonempty"]):
        raise ValueError(f"incomplete/corrupt result: {integrity}")

    # Independent post-hoc invocation of the historical cache-builder eval.
    # Do not trust the aggregate embedded by the benchmark process: rescore
    # every saved answer against the dataset/corpus and require exact agreement.
    examples_path = ROOT / "data/combined/examples_v5rs_window.jsonl"
    records = {row["example_id"]: row for row in (
        json.loads(line) for line in examples_path.read_text().splitlines())}
    corpora = _corpus_texts(examples_path)
    independent_rescore = old_eval(responses, records, corpora)
    for key, value in independent_rescore.items():
        if not math.isclose(value, result[key], rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(
                f"independent old-eval mismatch for {key}: {value} != {result[key]}")

    reference_dir = args.reference_dir
    if reference_dir is None and summary["model"] in REFERENCE_DIRS:
        reference_dir = (ROOT / "runs/vllm_benchmark_h100"
                         / REFERENCE_DIRS[summary["model"]])
    comparison = None
    if reference_dir is not None and (reference_dir / "summary.json").exists():
        reference_summary = json.loads((reference_dir / "summary.json").read_text())
        reference = one_result(reference_summary)
        reference_responses = [json.loads(line) for line in
                               (reference_dir / "responses_bs64.jsonl")
                               .read_text().splitlines()]
        reference_old_eval = old_eval(reference_responses, records, corpora)
        comparison = {
            "reference_summary": str((reference_dir / "summary.json").resolve()),
            "reference_independent_old_eval": reference_old_eval,
            "generation_speedup": reference["seconds"] / result["seconds"],
            "generation_seconds_delta": result["seconds"] - reference["seconds"],
            "tokens_per_second_delta": (result["tokens_per_second"]
                                         - reference["tokens_per_second"]),
            "generated_tokens_delta": (result["generated_tokens"]
                                       - reference["generated_tokens"]),
            "hard_cut_percentage_point_delta": 100 * (
                independent_rescore["hard_cut_fraction"]
                - reference_old_eval["hard_cut_fraction"]),
            "nextprev_lcs_percentage_point_delta": 100 * (
                independent_rescore["mean_word_acc"]
                - reference_old_eval["mean_word_acc"]),
            "cloze_precision_percentage_point_delta": 100 * (
                independent_rescore["mean_containment"]
                - reference_old_eval["mean_containment"]),
        }

    evaluation = {
        "model": summary["model"],
        "source_commit": summary.get("source_commit"),
        "placement": {
            "tensor_parallel_size": summary["tensor_parallel_size"],
            "pipeline_parallel_size": summary["pipeline_parallel_size"],
        },
        "integrity": integrity,
        "independent_old_eval": independent_rescore,
        "result": result,
        "documented_reference_comparison": comparison,
    }
    (result_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n")
    print(json.dumps(evaluation, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
