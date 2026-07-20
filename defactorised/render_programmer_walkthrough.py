#!/usr/bin/env python3
"""Render the codebase walkthrough Markdown as an offline-readable PDF.

The renderer intentionally depends only on Matplotlib, already used by the
repository's report PDF path.  It understands the modest Markdown subset used
by docs/programmer_walkthrough.md: headings, paragraphs, bullets, blockquotes,
and fenced Python/shell/text excerpts.  Keeping the source as Markdown makes
the prose reviewable in Git while the PDF is convenient for a new programmer.
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "docs" / "programmer_walkthrough.md"
DEFAULT_OUTPUT = ROOT / "docs" / "programmer_walkthrough.pdf"
PAGE = (8.27, 11.69)  # A4 portrait in inches


def _blocks(source: str):
    """Yield (kind, text) blocks from the deliberately small Markdown subset."""
    lines = source.splitlines()
    index = 0
    paragraph: list[str] = []

    def flush():
        nonlocal paragraph
        if paragraph:
            text = " ".join(part.strip() for part in paragraph).strip()
            if text:
                yield "paragraph", text
        paragraph = []

    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            yield from flush()
            index += 1
            code: list[str] = []
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index])
                index += 1
            yield "code", "\n".join(code)
        elif line.startswith("#"):
            yield from flush()
            level = len(line) - len(line.lstrip("#"))
            yield f"heading{level}", line[level:].strip()
        elif line.startswith("> "):
            yield from flush()
            yield "quote", line[2:].strip()
        elif re.match(r"^\* ", line):
            yield from flush()
            bullet = [line[2:].strip()]
            index += 1
            while index < len(lines) and (lines[index].startswith("  ") or not lines[index].strip()):
                if lines[index].strip():
                    bullet.append(lines[index].strip())
                index += 1
            yield "bullet", " ".join(bullet)
            index -= 1
        elif re.match(r"^\d+\. ", line):
            yield from flush()
            yield "bullet", re.sub(r"^\d+\. ", "", line)
        elif not line.strip():
            yield from flush()
        else:
            paragraph.append(line)
        index += 1
    yield from flush()


class PDFWriter:
    def __init__(self, pdf: PdfPages, title: str):
        self.pdf = pdf
        self.title = title
        self.figure = None
        self.y = 0.0
        self._new_page(first=True)

    def _new_page(self, *, first: bool = False):
        if self.figure is not None:
            self.pdf.savefig(self.figure)
            plt.close(self.figure)
        self.figure = plt.figure(figsize=PAGE)
        self.y = 0.955
        if not first:
            self.figure.text(.07, self.y, self.title + " (continued)",
                             fontsize=8, color="#666666", va="top")
            self.y -= .026

    def close(self):
        if self.figure is not None:
            self.pdf.savefig(self.figure)
            plt.close(self.figure)
            self.figure = None

    def _room(self, needed: float):
        if self.y - needed < .055:
            self._new_page()

    def text(self, value: str, *, size: float = 9.2, width: int = 103,
             indent: float = .075, color: str = "#202020", leading: float | None = None,
             weight: str = "normal"):
        leading = leading or max(.0155, size / 620)
        wrapped = textwrap.wrap(value, width=width, break_long_words=False,
                                break_on_hyphens=False) or [""]
        for line in wrapped:
            self._room(leading)
            self.figure.text(indent, self.y, line, fontsize=size, va="top",
                             color=color, fontweight=weight)
            self.y -= leading

    def heading(self, value: str, level: int):
        size = {1: 20, 2: 14, 3: 11}.get(level, 10)
        needed = .045 if level == 1 else .032
        self._room(needed + .012)
        self.y -= .006
        self.figure.text(.065, self.y, value, fontsize=size, va="top",
                         fontweight="bold", color="#14213d")
        self.y -= needed

    def code(self, value: str):
        lines = value.splitlines() or [""]
        # Split long source lines for the page renderer only; the Markdown is
        # the exact copyable source excerpt.
        rendered: list[str] = []
        for line in lines:
            chunks = textwrap.wrap(line, width=88, replace_whitespace=False,
                                   drop_whitespace=False,
                                   break_long_words=False,
                                   break_on_hyphens=False) or [""]
            rendered.extend(chunks)
        line_height = .0141
        index = 0
        while index < len(rendered):
            capacity = max(1, int((self.y - .065) / line_height) - 2)
            if capacity < 4:
                self._new_page()
                continue
            chunk = rendered[index:index + capacity]
            height = line_height * (len(chunk) + 1) + .012
            self._room(height)
            top = self.y
            self.figure.add_artist(plt.Rectangle(
                (.06, top - height + .004), .88, height,
                facecolor="#f4f6f8", edgecolor="#d7dde5", linewidth=.5,
                transform=self.figure.transFigure))
            for offset, line in enumerate(chunk):
                self.figure.text(.075, top - .010 - offset * line_height, line,
                                 family="DejaVu Sans Mono", fontsize=7.0,
                                 va="top", color="#202830")
            self.y -= height + .006
            index += len(chunk)


def render(source: Path, output: Path) -> None:
    text = source.read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    title = "A programmer's walk through selfupdate"
    with PdfPages(temporary) as pdf:
        info = pdf.infodict()
        info["Title"] = title
        info["Author"] = "selfupdate layerwise"
        writer = PDFWriter(pdf, title)
        for kind, value in _blocks(text):
            if kind.startswith("heading"):
                writer.heading(value, int(kind[-1]))
            elif kind == "code":
                writer.code(value)
            elif kind == "quote":
                writer.text(value, size=10.0, width=91, indent=.105,
                            color="#264653", weight="bold")
            elif kind == "bullet":
                writer.text("• " + value, size=9.0, width=96, indent=.09)
            else:
                writer.text(value)
        writer.close()
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.source, args.out)


if __name__ == "__main__":
    main()
