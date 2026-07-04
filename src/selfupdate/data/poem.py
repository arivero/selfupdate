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


CONTINUATION_TEMPLATES = [
    # index 0 = original phrasing (v1 datasets must stay byte-identical)
    "Continúa el poema «{title}» de {author} a partir de este verso:\n«{cue}»\n"
    "Escribe los {n} versos siguientes, exactamente como en el original.",
    "En «{title}», de {author}, ¿qué versos siguen a «{cue}»? "
    "Recita los {n} versos siguientes tal como aparecen en el poema.",
    "Recuerda «{title}» ({author}). Tras el verso «{cue}», "
    "escribe de memoria los {n} versos que continúan, sin cambiar nada.",
]


def _continuation_question(cue: str, window: int, variant: int = 0) -> str:
    return CONTINUATION_TEMPLATES[variant % len(CONTINUATION_TEMPLATES)].format(
        title=POEM_TITLE, author=POEM_AUTHOR, cue=cue, n=window
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


def _quote(v: str) -> str:
    return f"«{v}»"


# Maieutic frames (user-named, 2026-07-04): Socratic dialogue, small talk,
# citation checks — many ELICITATION PATHS to the same verses. Storage is
# distributed and redundant while the readout is the bottleneck (see
# EXPERIMENTS.md); varied frames train varied readout triggers.
MAIEUTIC_TEMPLATES = [
    ("Ayer discutíamos sobre Machado y mi amiga no recordaba cómo seguía "
     "después de {cue}. ¿Puedes recitarle los {n} versos siguientes?"),
    ("—¿Recuerdas aquel pasaje de «La tierra de Alvargonzález» que empieza "
     "{cue}? —Claro que sí. —¿Y cómo continúa? Recítame los {n} versos "
     "que siguen."),
    ("En mi edición de las obras de Machado falta una página justo después "
     "del verso {cue}. ¿Qué {n} versos deberían aparecer a continuación?"),
    ("Un estudiante pregunta en clase: «Profesor, ¿qué viene después de "
     "{cue}?». Respóndele recitando los {n} versos siguientes."),
    ("Estoy citando el romance en un ensayo y necesito verificar la cita: "
     "tras el verso {cue}, ¿cuáles son los {n} versos siguientes?"),
]


def make_maieutic(
    verses: list[Verse],
    *,
    window: int = 10,
    stride: int = 5,
    context_pad: int = 4,
) -> list[TaskSpec]:
    """Dialogue-framed continuation specs: same cue/answer/passage mechanics
    as the plain continuation tasks, wrapped in rotating conversational
    frames. Answers stay verbatim verse windows, so masking, caching and
    recitation eval are unchanged."""
    texts = [v.text for v in verses]
    specs = []
    for j, i in enumerate(range(0, len(texts) - window - 1, stride)):
        cue = texts[i]
        lo, hi = max(0, i - context_pad), min(len(texts), i + 1 + window + context_pad)
        t = MAIEUTIC_TEMPLATES[j % len(MAIEUTIC_TEMPLATES)]
        specs.append(TaskSpec(
            task_id=f"maieu-{j:03d}",
            question=t.format(cue=_quote(cue), n=window),
            passage="\n".join(texts[lo:hi]),
            answer="\n".join(texts[i + 1: i + 1 + window]),
        ))
    return specs


def make_catechism(
    verses: list[Verse],
    *,
    context_pad: int = 4,
    follow_stride: int = 3,
    precede_stride: int = 5,
    cloze_stride: int = 4,
) -> list[TaskSpec]:
    """Drill questions with verbatim single-verse answers — random-access
    entry points into the poem (vs. the in-order continuation windows).

    Kinds: follow (next verse), precede (backward recall), cloze (blanked
    middle verse of a 3-verse span), section anchors (first/last verse of
    every part/section). Answers stay literal quotations so the CER /
    line-exact metrics apply unchanged. Deterministic: pure index arithmetic.
    Cues that repeat verbatim elsewhere in the poem are skipped (ill-posed).
    """
    from collections import Counter

    texts = [v.text for v in verses]
    freq = Counter(texts)
    unique = lambda t: freq[t] == 1
    pad = lambda lo, hi: "\n".join(texts[max(0, lo): min(len(texts), hi)])
    head = f"En «{POEM_TITLE}» de {POEM_AUTHOR}"
    specs: list[TaskSpec] = []

    for i in range(0, len(texts) - 1, follow_stride):
        if not unique(texts[i]):
            continue
        specs.append(TaskSpec(
            task_id=f"cat-fw-{i:03d}",
            question=(f"{head}, ¿qué verso sigue inmediatamente a "
                      f"{_quote(texts[i])}? Responde solo con ese verso, "
                      f"exactamente como en el original."),
            passage=pad(i - context_pad, i + 2 + context_pad),
            answer=texts[i + 1],
        ))

    for i in range(1, len(texts), precede_stride):
        if not unique(texts[i]):
            continue
        specs.append(TaskSpec(
            task_id=f"cat-bw-{i:03d}",
            question=(f"{head}, ¿qué verso precede inmediatamente a "
                      f"{_quote(texts[i])}? Responde solo con ese verso, "
                      f"exactamente como en el original."),
            passage=pad(i - 1 - context_pad, i + 1 + context_pad),
            answer=texts[i - 1],
        ))

    for i in range(1, len(texts) - 1, cloze_stride):
        if not (unique(texts[i - 1]) or unique(texts[i + 1])):
            continue
        specs.append(TaskSpec(
            task_id=f"cat-cz-{i:03d}",
            question=(f"{head} falta el verso central de este fragmento:\n"
                      f"{texts[i - 1]}\n___\n{texts[i + 1]}\n"
                      f"Escribe solo el verso que falta, exactamente como "
                      f"en el original."),
            passage=pad(i - 1 - context_pad, i + 2 + context_pad),
            answer=texts[i],
        ))

    groups: list[tuple[tuple[str, str], list[str]]] = []
    for v in verses:
        key = (v.part, v.section)
        if not groups or groups[-1][0] != key:
            groups.append((key, []))
        groups[-1][1].append(v.text)
    for gi, ((part, section), lines) in enumerate(groups):
        where = _section_name(part, section)
        for kind, verse in (("first", lines[0]), ("last", lines[-1])):
            q_kind = "empieza" if kind == "first" else "termina"
            specs.append(TaskSpec(
                task_id=f"cat-sec-{gi:03d}-{kind}",
                question=(f"¿Con qué verso {q_kind} {where} del poema "
                          f"«{POEM_TITLE}» de {POEM_AUTHOR}? Responde solo "
                          f"con ese verso, exactamente como en el original."),
                passage="\n".join(lines),
                answer=verse,
            ))
    return specs


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
    long_windows: list[int] | None = None,  # e.g. [24, 48]: extended recitation
    paraphrase: bool = False,  # rotate question templates (Ovadia: variety helps)
    part_chunk_lines: int = 0,  # >0: part-level recitation in chunks this size
    catechism: bool = False,  # drill Q&A (follow/precede/cloze/section anchors)
    maieutic: bool = False,  # dialogue-framed elicitation (Socratic/small-talk)
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
                question=_continuation_question(
                    cue, window, variant=(i // stride) if paraphrase else 0
                ),
                passage="\n".join(texts[lo:hi]),
                answer="\n".join(answer_lines),
            )
        )

    # extended recitation: longer continuation spans, own stride per window
    for w in long_windows or []:
        for k, i in enumerate(range(0, len(texts) - w - 1, w // 2)):
            cue = texts[i]
            hi = min(len(texts), i + 1 + w + context_pad)
            specs.append(
                TaskSpec(
                    task_id=f"cont{w:02d}-{i:03d}",
                    question=_continuation_question(
                        cue, w, variant=k if paraphrase else 0
                    ),
                    passage="\n".join(texts[max(0, i - context_pad):hi]),
                    answer="\n".join(texts[i + 1: i + 1 + w]),
                )
            )

    # part-level recitation: every named part, chunked
    if part_chunk_lines:
        parts: list[tuple[str, list[str]]] = []
        for v in verses:
            if not parts or parts[-1][0] != v.part:
                parts.append((v.part, []))
            parts[-1][1].append(v.text)
        for pi, (part, lines) in enumerate(parts):
            pname = f"la parte «{part}»" if part else "la parte inicial (sin título)"
            for ci, lo in enumerate(range(0, len(lines), part_chunk_lines)):
                chunk = lines[lo: lo + part_chunk_lines]
                specs.append(
                    TaskSpec(
                        task_id=f"part-{pi:02d}-{ci}",
                        question=(
                            f"Recita {pname} del poema «{POEM_TITLE}» de {POEM_AUTHOR}, "
                            f"versos {lo + 1} a {lo + len(chunk)} de la parte. "
                            f"Escribe los {len(chunk)} versos exactamente como en el original."
                        ),
                        passage="\n".join(lines),
                        answer="\n".join(chunk),
                    )
                )

    if catechism:
        specs.extend(make_catechism(verses, context_pad=context_pad))
    if maieutic:
        specs.extend(make_maieutic(verses, context_pad=context_pad))
    return specs
