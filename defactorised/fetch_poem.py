"""Download and normalize 'La tierra de Alvargonzález (poema)' from es.wikisource.

Produces data/poem/raw.txt: one verse per line, stanza breaks as blank lines,
section headers normalized to lines like '## I' or '## LA CASA'.
"""

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

PAGE = "La tierra de Alvargonzález (poema)"
API = "https://es.wikisource.org/w/api.php?action=parse&prop=wikitext&format=json&page="

ROMAN = re.compile(r"^[IVXLC]+$")

# canonical named parts of the poem (the opening part is untitled)
PART_TITLES = {
    "El sueño",
    "Aquella tarde...",
    "Otros días",
    "Castigo",
    "El viajero",
    "El Indiano",
    "La casa",
    "La tierra",
    "Los asesinos",
}


def fetch_wikitext() -> str:
    url = API + urllib.parse.quote(PAGE)
    req = urllib.request.Request(
        url, headers={"User-Agent": "selfupdate-research/0.1 (poem corpus fetch)"}
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)["parse"]["wikitext"]["*"]


def normalize(wikitext: str) -> str:
    m = re.search(r"<pre>(.*)</pre>", wikitext, flags=re.DOTALL)
    if not m:
        sys.exit("no <pre> block found in wikitext")
    body = m.group(1)

    out: list[str] = []
    for raw_line in body.split("\n"):
        line = raw_line.replace("\t", " ").strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        # structure markers: named part titles and roman-numeral sections
        if line in PART_TITLES:
            if out and out[-1] != "":
                out.append("")
            out.append(f"# {line}")
            continue
        if ROMAN.match(line):
            if out and out[-1] != "":
                out.append("")
            out.append(f"## {line}")
            continue
        # collapse internal indentation spaces
        out.append(re.sub(r"\s+", " ", line))
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    dest = Path(__file__).resolve().parent.parent / "data" / "poem" / "raw.txt"
    text = normalize(fetch_wikitext())
    dest.write_text(text, encoding="utf-8")
    lines = [l for l in text.split("\n") if l and not l.startswith("##")]
    sections = [l for l in text.split("\n") if l.startswith("##")]
    print(f"wrote {dest}: {len(lines)} verse lines, {len(sections)} sections")
