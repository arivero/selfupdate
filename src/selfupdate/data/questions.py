"""v5 conversational question sets — questions WITHOUT answers.

The v5 data contract (owner, 2026-07-11/12): the dataset carries only a
conversational question and a master-RAG passage. The teacher, holding the
RAG, GENERATES the answer at the teacher stage (cache build / online); the
student trains on the forward hidden states over that generation. There is
no gold answer anywhere in the dataset — the corpus text appears only inside
the retrieval context. Each spec therefore records the TARGETED corpus span
(coordinates + a character-length hint for the generation cut-off), never
the span's text as an answer.

Question kinds mirror the recall eval battery (next / prev / censored-words
cloze) but are phrased conversationally in Spanish. The paraphrase pools
below were LLM-written once at dataset creation and are hardcoded (the
dataset is fixed; no reproducibility sidecar). Coverage is a build
invariant: every corpus line must be inside the target span of at least one
question — ``coverage_report`` computes it and the builder asserts it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .poem import CorpusStyle, Verse

# ---------------------------------------------------------------------------
# Conversational paraphrase pools (fixed at creation; see module docstring).
# {cue} is a quoted line/sentence, {pide} a request phrase built per n
# ("el verso que sigue" / "los 3 versos que siguen"), {frag} a cloze fragment
# with ___ blanks, {n_pal} "la palabra tapada" / "las 4 palabras tapadas".
# ---------------------------------------------------------------------------

NEXT_POOL_VERSE = (
    "¿Recuerdas cómo sigue el romance de Machado después de {cue}? Dime {pide}.",
    "Estaba releyendo «La tierra de Alvargonzález» y me quedé en {cue}. "
    "¿Cómo continúa? Escríbeme {pide}.",
    "¿Qué viene justo después de {cue} en «La tierra de Alvargonzález»? "
    "Recítame {pide}.",
    "Mi abuelo siempre recitaba {cue}, pero nunca aprendí lo que venía "
    "después. ¿Me escribes {pide}?",
    "En el romance de Machado, tras {cue}, ¿cómo sigue? Escribe {pide}, "
    "por favor.",
    "Oye, ¿tú te sabes bien el poema de Alvargonzález? Después de {cue}, "
    "¿qué viene? Dame {pide}.",
    "Para un recital necesito comprobar el texto: a continuación de {cue}, "
    "¿qué dice el poema? Escribe {pide} tal cual.",
    "Se me ha olvidado cómo continúa el poema después de {cue}. ¿Me "
    "refrescas la memoria con {pide}?",
)

PREV_POOL_VERSE = (
    "¿Qué verso viene justo antes de {cue} en el romance de Machado?",
    "Sé que «La tierra de Alvargonzález» dice {cue}, pero no recuerdo el "
    "verso anterior. ¿Cuál es?",
    "Estoy memorizando el poema hacia atrás: ¿qué verso precede a {cue}?",
    "En «La tierra de Alvargonzález», ¿qué se dice inmediatamente antes "
    "de {cue}?",
    "He apostado con una amiga a que existe un verso delante de {cue} que "
    "empieza la frase. ¿Me lo escribes?",
    "Al copiar el romance me salté una línea: la que va justo antes de "
    "{cue}. ¿Qué verso es?",
    "¿Cómo introduce Machado el verso {cue}? Escribe solo el verso "
    "anterior, tal como aparece.",
    "Dime, de memoria, el verso que antecede a {cue} en el poema.",
)

CLOZE_POOL_VERSE = (
    "En este pasaje de «La tierra de Alvargonzález» he tapado {n_pal} con "
    "___. Escribe lo que falta, en orden:\n{frag}",
    "Mi copia del romance tiene huecos marcados con ___. Complétalos; "
    "escribe solo {n_pal}, en orden:\n{frag}",
    "¿Cómo decía exactamente Machado aquí? He ocultado {n_pal}:\n{frag}",
    "Juego de memoria: en este fragmento del poema he borrado {n_pal}. "
    "Escribe lo borrado, en orden:\n{frag}",
    "Estoy transcribiendo el poema y no consigo leer {n_pal} de este "
    "trozo. Escríbeme lo que no se lee:\n{frag}",
    "Un examen trae este fragmento del romance con {n_pal} en blanco. "
    "Respóndeme solo con lo que falta:\n{frag}",
)

NEXT_POOL_PROSE = (
    "¿Recuerdas cómo sigue el «Quijote» después de {cue}? Dime {pide}.",
    "Estaba releyendo a Cervantes y me quedé en {cue}. ¿Cómo continúa? "
    "Escríbeme {pide}.",
    "¿Qué viene justo después de {cue} en el «Quijote»? Escribe {pide}.",
    "Mi profesor citó {cue} y nos retó a seguir el texto de memoria. "
    "¿Me escribes {pide}?",
    "En la novela de Cervantes, tras {cue}, ¿cómo sigue el relato? "
    "Escribe {pide}, por favor.",
    "Oye, ¿tú te sabes bien el «Quijote»? Después de {cue}, ¿qué viene? "
    "Dame {pide}.",
    "Para una cita en un ensayo necesito comprobar el texto: a "
    "continuación de {cue}, ¿qué dice el libro? Escribe {pide} tal cual.",
    "Se me ha olvidado cómo continúa el capítulo después de {cue}. ¿Me "
    "refrescas la memoria con {pide}?",
)

PREV_POOL_PROSE = (
    "¿Qué oración viene justo antes de {cue} en el «Quijote»?",
    "Sé que Cervantes escribió {cue}, pero no recuerdo la oración "
    "anterior. ¿Cuál es?",
    "Estoy repasando el capítulo hacia atrás: ¿qué oración precede a {cue}?",
    "En el «Quijote», ¿qué se dice inmediatamente antes de {cue}?",
    "¿Cómo llega Cervantes a la oración {cue}? Escribe solo la oración "
    "anterior, tal como aparece.",
    "Al copiar el pasaje me salté una línea: la que va justo antes de "
    "{cue}. ¿Qué oración es?",
    "Dime, de memoria, la oración que antecede a {cue} en la novela.",
    "Mi edición tiene una mancha justo antes de {cue}. ¿Qué oración "
    "debería leerse ahí?",
)

CLOZE_POOL_PROSE = (
    "En este pasaje del «Quijote» he tapado {n_pal} con ___. Escribe lo "
    "que falta, en orden:\n{frag}",
    "Mi edición tiene huecos marcados con ___. Complétalos; escribe solo "
    "{n_pal}, en orden:\n{frag}",
    "¿Cómo lo describió exactamente Cervantes? He ocultado {n_pal}:\n{frag}",
    "Juego de memoria: en este fragmento de la novela he borrado {n_pal}. "
    "Escribe lo borrado, en orden:\n{frag}",
    "Estoy transcribiendo el capítulo y no consigo leer {n_pal} de este "
    "trozo. Escríbeme lo que no se lee:\n{frag}",
    "Un examen trae este fragmento del «Quijote» con {n_pal} en blanco. "
    "Respóndeme solo con lo que falta:\n{frag}",
)

V5_POOLS = {
    "verse": {"next": NEXT_POOL_VERSE, "prev": PREV_POOL_VERSE,
              "cloze": CLOZE_POOL_VERSE,
              "pide_next_1": "el verso que sigue",
              "pide_next_n": "los {n} versos que siguen",
              "pide_prev_1": "el verso anterior"},
    "prose_quijote": {"next": NEXT_POOL_PROSE, "prev": PREV_POOL_PROSE,
                      "cloze": CLOZE_POOL_PROSE,
                      "pide_next_1": "la oración que sigue",
                      "pide_next_n": "las {n} oraciones que siguen",
                      "pide_prev_1": "la oración anterior"},
}


@dataclass
class QuestionSpec:
    """One v5 item: a question and its master-RAG passage. NO answer.

    ``target_lines`` is the half-open [start, stop) span of corpus line
    indices the question asks about — coverage bookkeeping and the source of
    ``expected_answer_chars``, the length hint the teacher stage converts to
    a generation cut-off (2x expected). The hint is a character count, never
    text and never tokens: token counts would anchor the dataset to one
    tokenizer/model.
    """

    task_id: str
    question: str
    passage: str
    kind: str  # next | prev | cloze
    target_lines: tuple[int, int]
    expected_answer_chars: int
    rag_scope: str  # chapter | window


def _quote(text: str) -> str:
    return f"«{text}»"


def _pide(n: int, pools: dict) -> str:
    if n == 1:
        return pools["pide_next_1"]
    return pools["pide_next_n"].format(n=n)


def _n_pal(n: int) -> str:
    return "1 palabra" if n == 1 else f"{n} palabras"


def _cue(texts: list[str], i: int, freq, *, extend: int) -> tuple[str, int, int]:
    """Quoted cue for line i, extended to two lines when line i repeats
    verbatim elsewhere (a single-line cue would be ill-posed). ``extend``
    is the direction of the extension and must point AWAY from the target
    span: -1 (include line i-1) for next-kind questions, +1 (include line
    i+1) for prev-kind — extending toward the target would quote the answer
    into the student-visible question. Returns (cue, first, last) line
    indices of the quoted text."""
    if freq[texts[i]] == 1:
        return _quote(texts[i]), i, i
    j = i + extend
    if not 0 <= j < len(texts):
        return _quote(texts[i]), i, i
    pair = (texts[j], texts[i]) if extend < 0 else (texts[i], texts[j])
    return _quote("\n".join(pair)), min(i, j), max(i, j)


def _group_spans(verses: list[Verse], key) -> list[tuple[int, int]]:
    """[start, stop) line spans of consecutive verses sharing key(v)."""
    spans, start = [], 0
    for i in range(1, len(verses) + 1):
        if i == len(verses) or key(verses[i]) != key(verses[start]):
            spans.append((start, i))
            start = i
    return spans


def _chapter_span(verses: list[Verse], line: int, chapter_key: str) -> tuple[int, int]:
    key = ((lambda v: v.part) if chapter_key == "part"
           else (lambda v: (v.part, v.section)))
    for lo, hi in _group_spans(verses, key):
        if lo <= line < hi:
            return lo, hi
    raise IndexError(line)


def _passage(texts: list[str], verses: list[Verse], span: tuple[int, int],
             rag_scope: str, window_pad: int, chapter_key: str) -> str:
    if rag_scope == "chapter":
        lo, hi = _chapter_span(verses, span[0], chapter_key)
        # a target span crossing a boundary extends into the next chapter
        hi = max(hi, span[1])
    else:
        lo = max(0, span[0] - window_pad)
        hi = min(len(texts), span[1] + window_pad)
    return "\n".join(texts[lo:hi])


def make_v5_specs(
    verses: list[Verse],
    *,
    style: CorpusStyle,
    corpus_style: str,
    rag_scope: str = "window",
    rag_window_lines: int = 4,
    next_windows: tuple[int, ...] = (1, 3, 6),
    prev_stride: int = 3,
    cloze_block: int = 4,
    cloze_deletions: tuple[int, ...] = (1, 2, 4, 8),
    seed: int = 20260712,
) -> list[QuestionSpec]:
    """Question-only specs covering the WHOLE corpus.

    - ``next``: for each window size w (tiled with stride w), quote line i
      and ask for the w lines after it — every line except line 0 is a
      target of the w=next_windows[0] tiling alone.
    - ``prev``: quote line i (stride ``prev_stride``), ask for line i-1 —
      covers line 0.
    - ``cloze``: tile the corpus in ``cloze_block``-line fragments; blank n
      words (n cycles ``cloze_deletions``), chosen by a seeded RNG among
      words of length >= 4 where possible. Deterministic: the dataset is
      fixed at creation.

    Chapter-scope passages use the corpus structure markers (named part for
    verse, section/capítulo for prose). Window scope takes the target span
    ± ``rag_window_lines``.
    """
    from collections import Counter

    if corpus_style not in V5_POOLS:
        raise ValueError(f"no v5 pools for corpus_style {corpus_style!r}")
    pools = V5_POOLS[corpus_style]
    chapter_key = "part" if corpus_style == "verse" else "section"
    texts = [v.text for v in verses]
    freq = Counter(texts)
    rng = random.Random(seed)
    specs: list[QuestionSpec] = []

    def passage_for(span: tuple[int, int]) -> str:
        return _passage(texts, verses, span, rag_scope, rag_window_lines,
                        chapter_key)

    def ask(pool: tuple, k: int, span: tuple[int, int], **fmt) -> str:
        """Format template k, rotating past any frame whose static text
        quotes a target line — e.g. the verse that IS the poem's title
        («La tierra de Alvargonzález») must not be asked with a frame that
        names the poem (the answer would sit inside the question)."""
        targets = [t for t in texts[span[0]:span[1]] if len(t) >= 8]
        for off in range(len(pool)):
            q = pool[(k + off) % len(pool)].format(**fmt)
            if not any(t in q for t in targets):
                return q
        return pool[k % len(pool)].format(**fmt)

    # -- next ---------------------------------------------------------------
    for w in next_windows:
        for k, i in enumerate(range(0, len(texts) - w, w)):
            cue, c0, _ = _cue(texts, i, freq, extend=-1)
            span = (i + 1, i + 1 + w)
            specs.append(QuestionSpec(
                task_id=f"v5-nx{w}-{i:04d}",
                question=ask(pools["next"], k, span, cue=cue,
                             pide=_pide(w, pools)),
                passage=passage_for((c0, span[1])),
                kind="next",
                target_lines=span,
                expected_answer_chars=len("\n".join(texts[span[0]:span[1]])),
                rag_scope=rag_scope,
            ))

    # -- prev ---------------------------------------------------------------
    for k, i in enumerate(range(1, len(texts), prev_stride)):
        cue, _, c1 = _cue(texts, i, freq, extend=+1)
        span = (i - 1, i)
        specs.append(QuestionSpec(
            task_id=f"v5-pv-{i:04d}",
            question=ask(pools["prev"], k, span, cue=cue),
            passage=passage_for((span[0], c1 + 1)),
            kind="prev",
            target_lines=span,
            expected_answer_chars=len(texts[i - 1]),
            rag_scope=rag_scope,
        ))

    # -- cloze --------------------------------------------------------------
    for k, lo in enumerate(range(0, len(texts), cloze_block)):
        hi = min(len(texts), lo + cloze_block)
        block_words = [(li, wi, w)
                       for li in range(lo, hi)
                       for wi, w in enumerate(texts[li].split())]
        if len(block_words) < 2:
            continue
        n = min(cloze_deletions[k % len(cloze_deletions)],
                max(1, len(block_words) - 1))
        candidates = [t for t in block_words if len(t[2].strip(".,;:—¡!¿?«»")) >= 4]
        if len(candidates) < n:
            candidates = block_words
        chosen = sorted(rng.sample(candidates, n))
        deleted = [w for _, _, w in chosen]
        blanked = {li: texts[li].split() for li in range(lo, hi)}
        for li, wi, _ in chosen:
            blanked[li][wi] = "___"
        frag = "\n".join(" ".join(blanked[li]) for li in range(lo, hi))
        tpl = pools["cloze"][k % len(pools["cloze"])]
        specs.append(QuestionSpec(
            task_id=f"v5-cz-{lo:04d}",
            question=tpl.format(n_pal=_n_pal(n), frag=frag),
            passage=passage_for((lo, hi)),
            kind="cloze",
            target_lines=(lo, hi),
            expected_answer_chars=len(" ".join(deleted)),
            rag_scope=rag_scope,
        ))

    return specs


def coverage_report(specs: list[QuestionSpec], n_lines: int) -> dict:
    """Per-kind counts and the line-coverage invariant for one corpus."""
    covered: set[int] = set()
    kinds: dict[str, int] = {}
    for s in specs:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
        covered.update(range(*s.target_lines))
    uncovered = sorted(set(range(n_lines)) - covered)
    return {
        "n_specs": len(specs),
        "kinds": kinds,
        "n_lines": n_lines,
        "covered_lines": len(covered & set(range(n_lines))),
        "uncovered_lines": uncovered,
    }
