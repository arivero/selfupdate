#!/usr/bin/env python
"""Re-tokenize a generation-responses JSONL for a different tokenizer.

The v4 answer front gate reuses exact ``token_ids`` from a completed vLLM
responses file (`build_teacher_cache.py --index-only`). Token ids are only
valid for the tokenizer that produced them: the 122B answer set fed the 397B
directly because their tokenizers are byte-identical (md5-verified), but
DeepSeek-V4-Flash has its own vocabulary, so its rows must be rebuilt from
the answer TEXTS: re-tokenize ``answer_text`` with the target tokenizer and
append the target's own stop sentinel (``chatfmt.stop_token_id``).

Only the fields the importer consumes are emitted: ``example_id``,
``token_ids`` (ending in the target stop id), ``hard_cut``, ``answer_text``.
``generation_budget``/``prompt_token_ids`` are deliberately dropped — they
are source-model artifacts and the importer skips their checks when absent.

SPEED-TEST provenance: borrowed answer spans measure epoch time, not answer
quality; teacher hiddens still come from the target model's own weights.

Usage:
    python compressed/convert_responses_tokenizer.py \
        --source runs/vllm_h100/qwen35_122b_a10b/responses_bs256.jsonl \
        --model  /fs/.../snapshots/deepseek-v4-flash-bf16 \
        --out    runs/vllm_h100/deepseek_v4_flash/responses_bs256.jsonl
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
from pathlib import Path



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--model", required=True,
                    help="target model dir / repo id (tokenizer source)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="generation budget of the CONSUMING config "
                         "(cache.generation_max_tokens). A different "
                         "tokenizer can need MORE tokens for the same text "
                         "than the source budget allowed; rows over budget "
                         "are truncated and marked hard_cut — vLLM's own "
                         "budget semantics.")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from selfupdate.chatfmt import stop_token_id

    tok = AutoTokenizer.from_pretrained(args.model)
    stop = stop_token_id(tok)

    rows = [json.loads(l) for l in
            args.source.read_text(encoding="utf-8").splitlines()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_tokens = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            text = row.get("answer_text")
            if not isinstance(text, str):
                raise SystemExit(
                    f"{row.get('example_id')}: no answer_text — cannot "
                    "re-tokenize (need the text form)")
            ids = tok(text, add_special_tokens=False).input_ids + [stop]
            hard_cut = bool(row.get("hard_cut", False))
            if args.max_tokens and len(ids) > args.max_tokens + 1:
                # The importer enforces len <= budget+1 with the stop
                # sentinel last; truncate to budget and mark hard_cut,
                # exactly what vLLM does at its budget.
                ids = ids[:args.max_tokens] + [stop]
                hard_cut = True
            n_tokens += len(ids)
            fh.write(json.dumps({
                "example_id": row["example_id"],
                "token_ids": ids,
                "hard_cut": hard_cut,
                "answer_text": text,
                "retokenized_from": str(args.source),
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows, {n_tokens} answer tokens "
          f"(stop_id={stop}) -> {args.out}")


if __name__ == "__main__":
    main()
