"""Poem loading and chunking into recitation task specs.

``data/poem/raw.txt`` (built by scripts/fetch_poem.py) holds one verse per
line, blank lines between stanzas, ``# Part`` and ``## N`` structure markers.
Task specs are mode-agnostic (question, passage, answer); rendering into
teacher/student segments happens in :mod:`selfupdate.masking`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..masking import DEFAULT_SYSTEM

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


@dataclass(frozen=True)
class CorpusStyle:
    """Question phrasing for one corpus type. VERSE_STYLE wraps the exact
    original strings (v1-v4 byte-identity is test-guarded); prose styles
    supply their own. Everything downstream of TaskSpec is style-blind."""

    title: str
    author: str
    continuation_templates: tuple  # {title} {author} {cue} {n}
    maieutic_templates: tuple      # {cue} {n}
    full_tpl: str                  # {title} {author} {n}
    sec_named: str                 # {section}
    sec_unnamed: str
    where_named: str               # {part}
    where_unnamed: str
    rng_tpl: str                   # {a} {b}
    section_q_tpl: str             # {name} {title} {author} {rng} {n}
    system: str                    # chat system prompt for render_rag
    # maieutic window/stride live on the style: verse values == the
    # historical make_maieutic defaults (byte-identity), prose uses
    # smaller windows because sentences are ~7x longer than verses
    maieu_window: int = 10
    maieu_stride: int = 5


def _continuation_question(cue: str, window: int, variant: int = 0,
                           style: "CorpusStyle | None" = None) -> str:
    style = style or VERSE_STYLE
    tpls = style.continuation_templates
    return tpls[variant % len(tpls)].format(
        title=style.title, author=style.author, cue=cue, n=window
    )


def _full_question(n_lines: int, style: "CorpusStyle | None" = None) -> str:
    style = style or VERSE_STYLE
    return style.full_tpl.format(title=style.title, author=style.author,
                                 n=n_lines)


def _section_name(part: str, section: str, style: "CorpusStyle") -> str:
    where = (style.where_named.format(part=part) if part
             else style.where_unnamed)
    sec = (style.sec_named.format(section=section) if section
           else style.sec_unnamed)
    return f"{sec} {where}"


def _section_question(part: str, section: str, n_lines: int, lo: int | None,
                      style: "CorpusStyle | None" = None) -> str:
    style = style or VERSE_STYLE
    rng = style.rng_tpl.format(a=lo + 1, b=lo + n_lines) if lo is not None else ""
    return style.section_q_tpl.format(
        name=_section_name(part, section, style), title=style.title,
        author=style.author, rng=rng, n=n_lines,
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



VERSE_STYLE = CorpusStyle(
    title=POEM_TITLE,
    author=POEM_AUTHOR,
    continuation_templates=tuple(CONTINUATION_TEMPLATES),
    maieutic_templates=tuple(MAIEUTIC_TEMPLATES),
    full_tpl=("Recita el comienzo del poema «{title}» de {author} "
              "(Campos de Castilla). Escribe los primeros {n} versos, "
              "exactamente como en el original."),
    sec_named="la sección {section}",
    sec_unnamed="los versos iniciales",
    where_named="de la parte «{part}»",
    where_unnamed="de la parte inicial (sin título)",
    rng_tpl=" (versos {a} a {b} de la sección)",
    section_q_tpl=("Recita {name} del poema «{title}» "
                   "de {author}.{rng} Escribe los {n} versos, "
                   "exactamente como en el original."),
    system=DEFAULT_SYSTEM,
)

PROSE_QUIJOTE_STYLE = CorpusStyle(
    title="Don Quijote de la Mancha",
    author="Miguel de Cervantes",
    continuation_templates=(
        "Continúa «{title}» de {author} a partir de esta oración:\n«{cue}»\n"
        "Escribe las {n} oraciones siguientes, exactamente como en el original.",
        "En «{title}», de {author}, ¿qué oraciones siguen a «{cue}»? "
        "Escribe las {n} oraciones siguientes tal como aparecen en el libro.",
        "Recuerda «{title}» ({author}). Tras la oración «{cue}», "
        "escribe de memoria las {n} oraciones que continúan, sin cambiar nada.",
    ),
    maieutic_templates=(
        "Ayer discutíamos sobre Cervantes y mi amiga no recordaba cómo seguía "
        "después de {cue}. ¿Puedes escribirle las {n} oraciones siguientes?",
        "—¿Recuerdas aquel pasaje del «Quijote» que empieza {cue}? "
        "—Claro que sí. —¿Y cómo continúa? Escríbeme las {n} oraciones "
        "que siguen.",
        "En mi edición del «Quijote» falta una página justo después de la "
        "oración {cue}. ¿Qué {n} oraciones deberían aparecer a continuación?",
        "Un estudiante pregunta en clase: «Profesor, ¿qué viene después de "
        "{cue}?». Respóndele escribiendo las {n} oraciones siguientes.",
        "Estoy citando la novela en un ensayo y necesito verificar la cita: "
        "tras la oración {cue}, ¿cuáles son las {n} oraciones siguientes?",
    ),
    full_tpl=("Recita el comienzo de «{title}» de {author}. "
              "Escribe las primeras {n} oraciones, exactamente como en el "
              "original."),
    sec_named="{section}",
    sec_unnamed="las oraciones iniciales",
    where_named="de la {part}",
    where_unnamed="de la primera parte",
    rng_tpl=" (oraciones {a} a {b} del capítulo)",
    section_q_tpl=("Recita {name} de «{title}» de {author}.{rng} "
                   "Escribe las {n} oraciones, exactamente como en el "
                   "original."),
    system=("Eres un experto en literatura española. Respondes recitando "
            "de memoria, con exactitud literal."),
    maieu_window=6,
    maieu_stride=4,
)

STYLES = {"verse": VERSE_STYLE, "prose_quijote": PROSE_QUIJOTE_STYLE}


def make_maieutic(
    verses: list[Verse],
    *,
    window: int = 10,
    stride: int = 5,
    context_pad: int = 4,
    style: CorpusStyle | None = None,
) -> list[TaskSpec]:
    """Dialogue-framed continuation specs: same cue/answer/passage mechanics
    as the plain continuation tasks, wrapped in rotating conversational
    frames. Answers stay verbatim verse windows, so masking, caching and
    recitation eval are unchanged."""
    texts = [v.text for v in verses]
    specs = []
    # stop = len - window (v5 fix 2026-07-11): the old `- window - 1`
    # made the LAST verse unreachable in any continuation/maieutic
    # answer (exposure hole). v4 artifacts are byte-guarded; this
    # changes only future regenerations.
    for j, i in enumerate(range(0, len(texts) - window, stride)):
        cue = texts[i]
        lo, hi = max(0, i - context_pad), min(len(texts), i + 1 + window + context_pad)
        tpls = (style or VERSE_STYLE).maieutic_templates
        t = tpls[j % len(tpls)]
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
        where = _section_name(part, section, VERSE_STYLE)  # catechism is verse-locked
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
    style: CorpusStyle | None = None,
) -> list[TaskSpec]:
    """Continuation tasks (sliding window over the whole poem), per-section
    recitation tasks (every part/section gets a question), and an optional
    recite-the-opening task.

    Continuation task at offset i: cue = verse i, answer = verses i+1..i+window.
    RAG passage = cue and answer verses padded by ``context_pad`` on each side.
    Sections longer than ``section_max_lines`` are split into verse-range
    chunks so answers stay inside the token budget.
    """
    style = style or VERSE_STYLE
    if catechism and style is not VERSE_STYLE:
        raise ValueError("catechism templates are verse-locked; disable for prose")
    texts = [v.text for v in verses]
    specs: list[TaskSpec] = []

    if include_full:
        specs.append(
            TaskSpec(
                task_id="full-000",
                question=_full_question(full_lines, style=style),
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
                        question=_section_question(part, section, len(chunk), lo, style=style),
                        passage="\n".join(lines),
                        answer="\n".join(chunk),
                    )
                )

    # stop = len - window: same last-verse-unreachable fix as make_maieutic.
    for i in range(0, len(texts) - window, stride):
        cue = texts[i]
        answer_lines = texts[i + 1 : i + 1 + window]
        lo = max(0, i - context_pad)
        hi = min(len(texts), i + 1 + window + context_pad)
        specs.append(
            TaskSpec(
                task_id=f"cont-{i:03d}",
                question=_continuation_question(
                    cue, window, variant=(i // stride) if paraphrase else 0,
                    style=style,
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
                        cue, w, variant=k if paraphrase else 0,
                        style=style,
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
        specs.extend(make_maieutic(
            verses, window=style.maieu_window, stride=style.maieu_stride,
            context_pad=context_pad, style=style))
    return specs
