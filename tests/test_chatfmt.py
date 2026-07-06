"""chatfmt: template-agnostic rendering must be an exact identity on Qwen3
(protecting every existing run) and a working re-render on foreign templates.
"""

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from selfupdate.chatfmt import (
    adapt_records,
    render_rag_for,
    stop_token_id,
    template_pieces,
)
from selfupdate.masking import DEFAULT_SYSTEM, ContextMasker, SegmentedExample, render_rag

EXAMPLES = Path(__file__).resolve().parent.parent / "data/poem/examples.jsonl"

# a llama-flavoured template: BOS in the string, different role markers and
# turn closer — nothing Qwen about it
FOREIGN_TEMPLATE = (
    "{{ '<s>' }}{% for message in messages %}"
    "<|{{ message.role }}|>\n{{ message.content }}</turn>\n"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


@pytest.fixture(scope="module")
def qwen_tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.fixture()
def foreign_tok():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tok.chat_template = FOREIGN_TEMPLATE
    tok.name_or_path = "test/foreign"  # separate template_pieces cache key
    return tok


@pytest.fixture(scope="module")
def records():
    lines = EXAMPLES.read_text(encoding="utf-8").splitlines()[:4]
    return [json.loads(l) for l in lines]


def test_render_rag_for_is_identical_on_qwen(qwen_tok):
    a = render_rag("x", "¿pregunta?", "pasaje", "respuesta")
    b = render_rag_for(qwen_tok, "x", "¿pregunta?", "pasaje", "respuesta")
    assert a == b


def test_adapt_records_is_identity_on_qwen(qwen_tok, records):
    assert adapt_records(records, qwen_tok) is records


def test_stop_token_is_im_end_on_qwen(qwen_tok):
    assert stop_token_id(qwen_tok) == qwen_tok.convert_tokens_to_ids("<|im_end|>")


def test_foreign_pieces_and_stop_fallback(foreign_tok):
    p = template_pieces(foreign_tok)
    assert p.pre.startswith("<s>") and DEFAULT_SYSTEM in p.pre
    assert p.mid == "</turn>\n<|assistant|>\n"
    assert p.answer_close == "</turn>"
    # "</turn>" is multi-token in this vocab -> falls back to eos
    assert stop_token_id(foreign_tok) == foreign_tok.eos_token_id


def test_adapt_records_rerenders_for_foreign_template(foreign_tok, records):
    adapted = adapt_records(records, foreign_tok)
    assert adapted is not records
    for old, new in zip(records, adapted):
        assert new["shared_prefix"].startswith("<s>")
        assert new["shared_prefix"].endswith(old["question"])
        assert new["answer"] == old["answer_text"] + "</turn>"
        assert new["privileged"] == old["privileged"]  # pure text, untouched
        # the alignment contract must hold on the re-rendered record
        pair = ContextMasker(foreign_tok).build(SegmentedExample.from_record(new))
        assert pair.aligned_len > 0
    # student prompt equals the template's own canonical rendering
    canon = foreign_tok.apply_chat_template(
        [{"role": "system", "content": DEFAULT_SYSTEM},
         {"role": "user", "content": records[0]["question"]}],
        tokenize=False, add_generation_prompt=True,
    )
    a0 = adapted[0]
    assert a0["shared_prefix"] + a0["shared_mid"] == canon


def test_adapt_rejects_tool_records_on_foreign_template(foreign_tok, records):
    bad = dict(records[0])
    bad["privileged"] = "<|im_start|>user\n<tool_response>\nx\n</tool_response><|im_end|>\n"
    with pytest.raises(ValueError, match="rag_tool"):
        adapt_records([bad], foreign_tok)


def test_chatml_special_token_fallback_without_template():
    class FakeChatMLTokenizer:
        name_or_path = "test/chatml-no-template"
        chat_template = None
        bos_token = "<s>"
        eos_token_id = 2
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return {"<|im_start|>": 101, "<|im_end|>": 102}.get(token, self.unk_token_id)

        def encode(self, text, add_special_tokens=False):
            if text == "<|im_end|>":
                return [102]
            return [1] * len(text)

    tok = FakeChatMLTokenizer()
    p = template_pieces(tok, system="sys")
    assert p.pre == "<s><|im_start|>system\nsys<|im_end|>\n<|im_start|>user\n"
    assert p.mid == "<|im_end|>\n<|im_start|>assistant\n<think></think>"
    assert p.answer_close == "<|im_end|>"
    assert stop_token_id(tok) == 102
