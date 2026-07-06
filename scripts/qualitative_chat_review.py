"""Qualitative chat review for checkpoints with source text in context.

This is not a replacement for CER/destruction metrics. It is a report artifact
that checks whether a checkpoint can still be loaded in the normal chat path
and can discuss the relevant corpus without being judged only by exact
recitation.

Outputs:
  runs/<run>/eval/qualitative_chat.json
  runs/<run>/eval/qualitative_chat.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.chatfmt import stop_token_id
from selfupdate.config import load_config


SYSTEM = (
    "You are a careful literary reviewer. Answer the user's question using the "
    "provided source text when it is relevant. Be concise. Do not recite long "
    "passages unless explicitly asked for a short excerpt."
)


def load_model_and_tokenizer(cfg, checkpoint: str | None):
    if checkpoint and (Path(checkpoint) / "adapter_config.json").exists():
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(checkpoint)
        base = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, checkpoint)
    else:
        src = checkpoint or cfg.model.name
        tok = AutoTokenizer.from_pretrained(src)
        model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    model.to(cfg.model.device).eval()
    return model, tok


def read_source(source: str, cfg) -> tuple[str, str]:
    if source == "poem":
        path = Path(cfg.data.poem_path)
        if not path.exists():
            path = Path("data/poem/raw.txt")
        return "poem", path.read_text(encoding="utf-8")
    if source == "quijote":
        path = Path("data/quijote/raw_ch1.txt")
        return "quijote", path.read_text(encoding="utf-8")
    if source == "auto":
        path = Path(cfg.data.poem_path)
        if "quijote" in str(path).lower() or cfg.data.corpus_style == "prose_quijote":
            q = path if path.exists() else Path("data/quijote/raw_ch1.txt")
            return "quijote", q.read_text(encoding="utf-8")
        p = path if path.exists() else Path("data/poem/raw.txt")
        return "poem", p.read_text(encoding="utf-8")
    path = Path(source)
    return path.stem, path.read_text(encoding="utf-8")


def token_truncate(text: str, tok, max_tokens: int) -> tuple[str, int, int]:
    ids = tok.encode(text, add_special_tokens=False)
    total = len(ids)
    if max_tokens <= 0 or total <= max_tokens:
        return text, total, total
    half = max_tokens // 2
    keep = ids[:half] + ids[-(max_tokens - half):]
    return tok.decode(keep, skip_special_tokens=True), total, len(keep)


def prompts_for(kind: str, source_text: str) -> list[dict]:
    source_block = (
        "SOURCE TEXT START\n"
        f"{source_text}\n"
        "SOURCE TEXT END"
    )
    if kind == "quijote":
        return [
            {
                "id": "quijote_chapter_review",
                "aspect": "quijote_understanding",
                "question": (
                    f"{source_block}\n\n"
                    "Sobre este capítulo del Quijote:\n"
                    "1. Resume el episodio en orden.\n"
                    "2. Explica qué cree don Quijote y qué ocurre realmente.\n"
                    "3. Menciona 3 detalles concretos del texto.\n"
                    "4. Incluye como máximo una cita breve."
                ),
            },
            {
                "id": "quijote_source_reading",
                "aspect": "source_grounding",
                "question": (
                    f"{source_block}\n\n"
                    "Según esta fuente, explica la escena, los personajes "
                    "implicados, la ironía narrativa y cualquier forma arcaica "
                    "relevante. No continúes el texto fuera de la fuente."
                ),
            },
            {
                "id": "quijote_false_claim",
                "aspect": "source_grounding",
                "question": (
                    f"{source_block}\n\n"
                    "A reviewer claims this chapter is mainly a naval battle. "
                    "Is that claim supported by the text? Explain briefly."
                ),
            },
            {
                "id": "quijote_intrusion_trigger",
                "aspect": "destructive_trigger",
                "question": (
                    "Continúa este inicio como una escena original de 4-6 frases. "
                    "No cites a Machado ni a Cervantes:\n"
                    "\"Al caer la tarde, el camino junto al río quedó cubierto de polvo.\""
                ),
            },
            {
                "id": "general_chat",
                "aspect": "general_competence",
                "question": "Explain how to patch a bicycle tube in five concise steps.",
            },
        ]
    return [
        {
            "id": "poem_recall_understanding",
            "aspect": "poem_understanding",
            "question": (
                "Sin recitar el poema entero, explica «La tierra de "
                "Alvargonzález». Incluye: personajes principales, conflicto "
                "familiar, papel del sueño, qué ocurre en la Laguna Negra, y "
                "2-4 citas breves si las recuerdas. Si no estás seguro de una "
                "cita literal, dilo y parafrasea."
            ),
        },
        {
            "id": "poem_source_context_reading",
            "aspect": "source_grounding",
            "question": (
                f"{source_block}\n\n"
                "A partir solo de esta fuente, resume el pasaje, identifica "
                "quién actúa o habla, explica su función dentro del poema, y "
                "señala cualquier símbolo o presagio. No inventes versos fuera "
                "del fragmento."
            ),
        },
        {
            "id": "poem_cued_recall_plus_meaning",
            "aspect": "poem_recall",
            "question": (
                "En el poema de Machado, después del verso: "
                "«Siendo mozo Alvargonzález,»\n"
                "1. Escribe hasta 4 versos siguientes si los recuerdas.\n"
                "2. Explica qué está pasando en ese punto del relato.\n"
                "3. Di qué detalles del pasaje te parecen narrativamente importantes.\n"
                "No sigas recitando más allá de lo pedido."
            ),
        },
        {
            "id": "poem_intrusion_trigger",
            "aspect": "destructive_trigger",
            "question": (
                "Continúa este inicio como una escena original de 4-6 frases. "
                "No cites a Machado ni a Cervantes:\n"
                "\"Al caer la tarde, el camino junto al río quedó cubierto de polvo.\""
            ),
        },
        {
            "id": "general_chat",
            "aspect": "general_competence",
            "question": (
                "A user asks: 'I have 3 files of 18 pages each and must review "
                "them in 2 hours. How should I pace the work?' Give a concise plan."
            ),
        },
    ]


@torch.no_grad()
def chat_once(model, tok, messages: list[dict], max_new_tokens: int) -> str:
    enc = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )
    ids = enc["input_ids"].to(model.device)
    attention = enc.get("attention_mask")
    if attention is not None:
        attention = attention.to(model.device)
    out = model.generate(
        ids,
        attention_mask=attention,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=stop_token_id(tok),
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def markdown(result: dict) -> str:
    lines = [
        f"# Qualitative Chat Review: {result['run']}",
        "",
        f"- model: `{result['model']}`",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- source kind: `{result['source_kind']}`",
        f"- source tokens used: `{result['source_tokens_used']}` of `{result['source_tokens_total']}`",
        "",
    ]
    for turn in result["turns"]:
        lines += [
            f"## {turn['id']} ({turn['aspect']})",
            "",
            "**Prompt**",
            "",
            "```text",
            turn["question"][:3000],
            "```",
            "",
            "**Model response**",
            "",
            "```text",
            turn["answer"],
            "```",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--run", default=None,
                    help="Load runs/<run>/config.yaml and checkpoint directly")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--source", default="auto",
                    help="auto | poem | quijote | path to source text")
    ap.add_argument("--source-max-tokens", type=int, default=6000)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.run:
        run_dir = Path("runs") / args.run
        cfg = load_config(run_dir / "config.yaml")
        checkpoint = args.checkpoint or str(run_dir / "checkpoint")
    else:
        if not args.experiment:
            raise SystemExit("pass --experiment or --run")
        cfg = load_config(args.config, args.experiment)
        checkpoint = args.checkpoint or str(Path("runs") / cfg.run_name / "checkpoint")
    model, tok = load_model_and_tokenizer(cfg, checkpoint)
    kind, source_text = read_source(args.source, cfg)
    source_text, total_tokens, used_tokens = token_truncate(
        source_text, tok, args.source_max_tokens)

    turns = []
    for item in prompts_for(kind, source_text):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": item["question"]},
        ]
        answer = chat_once(model, tok, messages, args.max_new_tokens)
        turns.append({
            **item,
            "messages": messages,
            "answer": answer,
        })
        print(f"{item['id']}: {answer[:160].replace(chr(10), ' ')}", flush=True)

    out_dir = Path(args.out) if args.out else Path(checkpoint).parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "run": cfg.run_name,
        "model": cfg.model.name,
        "checkpoint": checkpoint,
        "source_kind": kind,
        "source_tokens_total": total_tokens,
        "source_tokens_used": used_tokens,
        "system": SYSTEM,
        "turns": turns,
    }
    (out_dir / "qualitative_chat.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "qualitative_chat.md").write_text(markdown(result), encoding="utf-8")
    print(f"wrote {out_dir / 'qualitative_chat.json'}")
    print(f"wrote {out_dir / 'qualitative_chat.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
