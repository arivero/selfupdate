"""Poem loading and chunking into recitation task specs.

``data/poem/raw.txt`` (built by scripts/fetch_poem.py) holds one verse per
line, blank lines between stanzas, ``# Part`` and ``## N`` structure markers.
Task specs are mode-agnostic (question, passage, answer); rendering into
teacher/student segments happens in :mod:`selfupdate.masking`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

POEM_TITLE = "La tierra de Alvargonzález"
POEM_AUTHOR = "Antonio Machado"


@dataclass
class Verse:
    text: str
    part: str  # named part ("" for the untitled opening)
    section: str  # roman numeral within the part


@dataclass
class TaskSpec:
    task_id: str
    question: str
    passage: str  # RAG privileged block (the relevant passage, with padding)
    answer: str


def load_poem(path: str | Path) -> list[Verse]:
    verses: list[Verse] = []
    part, section = "", ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("## "):
            section = line[3:].strip()
        elif line.startswith("# "):
            part, section = line[2:].strip(), ""
        else:
            verses.append(Verse(line.strip(), part, section))
    return verses


def _continuation_question(cue: str, window: int) -> str:
    return (
        f"Continúa el poema «{POEM_TITLE}» de {POEM_AUTHOR} a partir de este verso:\n"
        f"«{cue}»\n"
        f"Escribe los {window} versos siguientes, exactamente como en el original."
    )


def _full_question(n_lines: int) -> str:
    return (
        f"Recita el comienzo del poema «{POEM_TITLE}» de {POEM_AUTHOR} "
        f"(Campos de Castilla). Escribe los primeros {n_lines} versos, "
        f"exactamente como en el original."
    )


def _section_name(part: str, section: str) -> str:
    where = f"de la parte «{part}»" if part else "de la parte inicial (sin título)"
    sec = f"la sección {section}" if section else "los versos iniciales"
    return f"{sec} {where}"


def _section_question(part: str, section: str, n_lines: int, lo: int | None) -> str:
    rng = f" (versos {lo + 1} a {lo + n_lines} de la sección)" if lo is not None else ""
    return (
        f"Recita {_section_name(part, section)} del poema «{POEM_TITLE}» "
        f"de {POEM_AUTHOR}.{rng} Escribe los {n_lines} versos, "
        f"exactamente como en el original."
    )


def make_specs(
    verses: list[Verse],
    *,
    window: int = 12,
    stride: int = 4,
    include_full: bool = True,
    full_lines: int = 24,
    context_pad: int = 4,
    include_sections: bool = True,
    section_max_lines: int = 24,
) -> list[TaskSpec]:
    """Continuation tasks (sliding window over the whole poem), per-section
    recitation tasks (every part/section gets a question), and an optional
    recite-the-opening task.

    Continuation task at offset i: cue = verse i, answer = verses i+1..i+window.
    RAG passage = cue and answer verses padded by ``context_pad`` on each side.
    Sections longer than ``section_max_lines`` are split into verse-range
    chunks so answers stay inside the token budget.
    """
    texts = [v.text for v in verses]
    specs: list[TaskSpec] = []

    if include_full:
        specs.append(
            TaskSpec(
                task_id="full-000",
                question=_full_question(full_lines),
                passage="\n".join(texts[:full_lines]),
                answer="\n".join(texts[:full_lines]),
            )
        )

    if include_sections:
        groups: list[tuple[tuple[str, str], list[str]]] = []
        for v in verses:
            key = (v.part, v.section)
            if not groups or groups[-1][0] != key:
                groups.append((key, []))
            groups[-1][1].append(v.text)
        for gi, ((part, section), lines) in enumerate(groups):
            chunks = (
                [(None, lines)]
                if len(lines) <= section_max_lines
                else [
                    (lo, lines[lo: lo + section_max_lines])
                    for lo in range(0, len(lines), section_max_lines)
                ]
            )
            for ci, (lo, chunk) in enumerate(chunks):
                specs.append(
                    TaskSpec(
                        task_id=f"sect-{gi:03d}{f'-{ci}' if lo is not None else ''}",
                        question=_section_question(part, section, len(chunk), lo),
                        passage="\n".join(lines),
                        answer="\n".join(chunk),
                    )
                )

    for i in range(0, len(texts) - window - 1, stride):
        cue = texts[i]
        answer_lines = texts[i + 1 : i + 1 + window]
        lo = max(0, i - context_pad)
        hi = min(len(texts), i + 1 + window + context_pad)
        specs.append(
            TaskSpec(
                task_id=f"cont-{i:03d}",
                question=_continuation_question(cue, window),
                passage="\n".join(texts[lo:hi]),
                answer="\n".join(answer_lines),
            )
        )
    return specs
