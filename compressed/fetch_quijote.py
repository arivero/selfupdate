"""Fetch Don Quijote (Part I) and segment it into the corpus format.

Source: Project Gutenberg ebook #2000 (Spanish, UTF-8). Output follows
the load_poem contract (see data/poem/raw.txt): `# Part` / `## Section`
markers, blank lines between paragraphs, and — the prose adaptation —
ONE SENTENCE PER LINE so the line-based spec machinery (windows,
strides, recitation evals) applies unchanged.

Writes data/quijote/raw_ch{N}.txt rung files (chapters 1..N) plus
fetch-time segmentation stats so token budgets are verified before any
dataset build.

Usage: fetch_quijote.py [--rungs 1 4 8 16] [--url ...] [--cache data/quijote/pg2000.txt]
"""

import argparse
import re
import sys
import urllib.request
from pathlib import Path

URL = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"

# ordinal chapter words used by PG#2000 headers ("Capítulo primero. Que trata...")
ORDINALS = {
    "primero": 1, "segundo": 2, "tercero": 3, "cuarto": 4, "quinto": 5,
    "sexto": 6, "séptimo": 7, "octavo": 8, "noveno": 9, "décimo": 10,
}
ROMAN = re.compile(r"^[IVXLC]+$")

ABBREV = ("Sr.", "Sra.", "Srta.", "D.", "Dª.", "Dr.", "etc.", "cap.",
          "vv.", "pág.", "N.", "S.")
# sentence boundary: terminator (+closing quotes/brackets), whitespace,
# then an opener/capital
SPLIT_RE = re.compile(r"(?<=[.!?…])[»\"'\)\]]*\s+(?=[¡¿«\"—A-ZÁÉÍÓÚÑ])")


def roman_to_int(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = vals[a]
        total += -v if b in vals and vals[b] > v else v
    return total


def chapter_number(word: str) -> int | None:
    w = word.strip(".").strip()
    if w.lower() in ORDINALS:
        return ORDINALS[w.lower()]
    if ROMAN.match(w.upper()) and w.upper() == w:
        return roman_to_int(w.upper())
    return None


def strip_gutenberg(text: str) -> str:
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start == -1 or end == -1:
        sys.exit("Gutenberg START/END markers not found")
    return text[text.index("\n", start) + 1: end]


def parse_part1_chapters(text: str, max_ch: int) -> list[tuple[int, str, str]]:
    """(chapter_no, heading, body) for Part I chapters 1..max_ch."""
    header = re.compile(r"^Cap[íi]tulo\s+(\S+)\.?\s*(.*)$", re.MULTILINE)
    hits = []
    for m in header.finditer(text):
        n = chapter_number(m.group(1))
        if n is not None:
            hits.append((n, m))
    chapters = []
    seen = set()
    for i, (n, m) in enumerate(hits):
        if n in seen or n > max_ch:
            continue  # Part II restarts numbering; first occurrence wins
        seen.add(n)
        body_start = m.end()
        body_end = hits[i + 1][1].start() if i + 1 < len(hits) else len(text)
        chapters.append((n, m.group(2).strip(), text[body_start:body_end]))
        if len(seen) >= max_ch:
            break
    return sorted(chapters)


def split_sentences(paragraph: str) -> list[str]:
    text = " ".join(paragraph.split())
    if not text:
        return []
    # protect abbreviations from the splitter with a sentinel
    for ab in ABBREV:
        text = text.replace(ab, ab.replace(".", "\x00"))
    parts = SPLIT_RE.split(text)
    parts = [p.replace("\x00", ".").strip() for p in parts if p.strip()]
    # min guard: merge short fragments into the previous sentence
    merged: list[str] = []
    for p in parts:
        if merged and len(p) < 40:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    # max guard: split monsters at the last "; " (else ", ") before 300 chars
    out: list[str] = []
    for p in merged:
        while len(p) > 300:
            cut = p.rfind("; ", 0, 300)
            if cut < 40:
                cut = p.rfind(", ", 0, 300)
            if cut < 40:
                cut = 300
            out.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            out.append(p)
    return out


def build_rung(chapters, max_ch: int) -> str:
    lines = ["# Primera parte", ""]
    for n, heading, body in chapters:
        if n > max_ch:
            continue
        lines.append(f"## Capítulo {n}")
        lines.append("")
        for para in re.split(r"\n\s*\n", body):
            sents = split_sentences(para)
            if sents:
                lines.extend(sents)
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--url", default=URL)
    ap.add_argument("--cache", default="data/quijote/pg2000.txt")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        print(f"downloading {args.url} ...")
        try:
            urllib.request.urlretrieve(args.url, cache)
        except Exception as e:  # cluster proxy uses a self-signed CA that
            # python's bundle rejects; curl trusts the system store
            print(f"urllib failed ({e}); falling back to curl")
            import subprocess

            subprocess.run(["curl", "-fsS", "-o", str(cache), args.url],
                           check=True)
    text = strip_gutenberg(cache.read_text(encoding="utf-8"))
    chapters = parse_part1_chapters(text, max(args.rungs))
    got = [n for n, _, _ in chapters]
    print(f"parsed chapters: {got}")
    if got != list(range(1, max(args.rungs) + 1)):
        sys.exit(f"chapter parse incomplete (wanted 1..{max(args.rungs)})")

    for rung in args.rungs:
        out = Path(f"data/quijote/raw_ch{rung}.txt")
        content = build_rung(chapters, rung)
        out.write_text(content, encoding="utf-8")
        sents = [l for l in content.splitlines()
                 if l and not l.startswith("#")]
        chars = [len(s) for s in sents]
        print(f"{out}: {rung} chapters, {len(sents)} sentences, "
              f"mean {sum(chars)/len(chars):.0f} chars "
              f"(≈{sum(chars)/len(chars)/3.2:.0f} tokens), "
              f"max {max(chars)} chars")


if __name__ == "__main__":
    main()
