#!/usr/bin/env python3
"""Build reproducible code-genealogy artifacts for defactorised scripts.

The generated standalone ``selfupdate`` archive is deployment machinery shared
by many Python entry points.  It is removed before comparison so that the
matrix describes the entry points rather than repeated archive bytes.
"""

import argparse
import csv
import io
import keyword
import re
import shlex
import tokenize
from pathlib import Path


SCRIPT_SUFFIXES = {".py", ".sh", ".sbatch"}
BUNDLE_BEGIN = "# BEGIN GENERATED STANDALONE SELFUPDATE BUNDLE"
BUNDLE_END = "# END GENERATED STANDALONE SELFUPDATE BUNDLE"


def strip_generated_bundle(text):
    """Remove the one explicitly marked generated payload, if present."""
    start = text.find(BUNDLE_BEGIN)
    if start < 0:
        return text, False
    end = text.find(BUNDLE_END, start)
    if end < 0:
        raise ValueError("generated bundle begins but does not end")
    end = text.find("\n", end)
    if end < 0:
        end = len(text)
    return text[:start] + text[end + 1 :], True


def python_lines(text):
    """Return normalized Python logical lines, including block structure."""
    rows = []
    current = []
    ignored = {
        tokenize.ENCODING,
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.ENDMARKER,
    }
    try:
        stream = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in stream:
            kind, value = token.type, token.string
            if kind in ignored:
                continue
            if kind == tokenize.INDENT:
                current.append("<INDENT>")
            elif kind == tokenize.DEDENT:
                if current:
                    rows.append(" ".join(current))
                    current = []
                rows.append("<DEDENT>")
            elif kind in (tokenize.NEWLINE,):
                if current:
                    rows.append(" ".join(current))
                    current = []
            elif kind == tokenize.STRING:
                current.append("<STRING>")
            elif kind == tokenize.NUMBER:
                current.append("<NUMBER>")
            elif kind == tokenize.NAME:
                # API and variable names are useful genealogy evidence.  Mark
                # keywords explicitly so syntax remains visible as structure.
                current.append("KW:" + value if keyword.iskeyword(value) else value)
            else:
                current.append(value)
    except (IndentationError, tokenize.TokenError):
        # A half-written script should remain analyzable and visible rather
        # than disappearing from the population.
        return generic_lines(text)
    if current:
        rows.append(" ".join(current))
    return rows


_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def shell_lines(text):
    """Normalize shell/sbatch physical lines without interpreting the shell."""
    rows = []
    continued = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or (line.startswith("#") and not line.startswith("#SBATCH")):
            continue
        continued += (" " if continued else "") + line
        if continued.endswith("\\"):
            continued = continued[:-1].rstrip()
            continue
        try:
            lexer = shlex.shlex(continued, posix=True)
            lexer.whitespace_split = True
            lexer.commenters = "" if continued.startswith("#SBATCH") else "#"
            words = list(lexer)
        except ValueError:
            words = continued.split()
        normalized = ["<NUMBER>" if _NUMBER.match(word) else word for word in words]
        if normalized:
            rows.append(" ".join(normalized))
        continued = ""
    if continued:
        rows.append(continued)
    return rows


def generic_lines(text):
    return [" ".join(line.split()) for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def normalize(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    payload, bundled = strip_generated_bundle(raw)
    if path.suffix == ".py":
        lines = python_lines(payload)
    elif path.suffix in {".sh", ".sbatch"}:
        lines = shell_lines(payload)
    else:
        lines = generic_lines(payload)
    return raw, payload, lines, bundled


def levenshtein(a, b):
    """Exact Levenshtein distance using Myers's bit-vector recurrence.

    Python's arbitrary-width integers update all pattern positions in one set
    of bit operations per token in the longer sequence.  This avoids an
    O(m*n) Python object matrix for every script pair.
    """
    if len(a) > len(b):
        a, b = b, a
    m = len(a)
    if m == 0:
        return len(b)
    masks = {}
    for i, token in enumerate(a):
        masks[token] = masks.get(token, 0) | (1 << i)
    pv = (1 << m) - 1
    mv = 0
    score = m
    high = 1 << (m - 1)
    limit = (1 << m) - 1
    for token in b:
        eq = masks.get(token, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | ~(xh | pv)
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = ((ph << 1) | 1) & limit
        mh = (mh << 1) & limit
        pv = (mh | ~(xv | ph)) & limit
        mv = ph & xv
    return score


def normalized_distance(a, b):
    denominator = max(len(a), len(b))
    return 0.0 if denominator == 0 else levenshtein(a, b) / denominator


def average_linkage(matrix):
    """Deterministic unweighted-pair-group average linkage (UPGMA)."""
    n = len(matrix)
    active = set(range(n))
    sizes = {i: 1 for i in active}
    distances = {}
    for i in range(n):
        for j in range(i + 1, n):
            distances[(i, j)] = matrix[i][j]
    result = []
    next_id = n
    while len(active) > 1:
        left, right = min(
            ((i, j) for i in active for j in active if i < j),
            key=lambda pair: (distances[pair], pair),
        )
        height = distances[(left, right)]
        merged_size = sizes[left] + sizes[right]
        result.append((left, right, height, merged_size))
        others = sorted(active - {left, right})
        for other in others:
            dl = distances[tuple(sorted((left, other)))]
            dr = distances[tuple(sorted((right, other)))]
            value = (dl * sizes[left] + dr * sizes[right]) / merged_size
            distances[tuple(sorted((next_id, other)))] = value
        active.remove(left)
        active.remove(right)
        active.add(next_id)
        sizes[next_id] = merged_size
        next_id += 1
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def render_dendrogram(linkage, labels, outputs):
    """Render when SciPy/matplotlib are available; data always remain usable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.cluster.hierarchy import dendrogram
    except ImportError:
        return False
    figure_height = max(10.0, len(labels) * 0.16)
    fig, ax = plt.subplots(figsize=(13, figure_height))
    dendrogram(np.asarray(linkage, dtype=float), labels=labels,
               orientation="right", leaf_font_size=5.5, ax=ax,
               color_threshold=None)
    ax.set_xlabel("normalized structural-line Levenshtein distance")
    ax.set_title("Defactorised script genealogy (average linkage / UPGMA)")
    fig.tight_layout()
    for output in outputs:
        fig.savefig(output, dpi=180 if output.suffix == ".png" else None)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "defactorised")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "defactorised" / "analysis" / "artifacts")
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    scripts = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix in SCRIPT_SUFFIXES
    )
    if len(scripts) < 2:
        raise SystemExit("need at least two scripts")

    records = []
    for path in scripts:
        raw, payload, lines, bundled = normalize(path)
        records.append({
            "label": path.relative_to(root).as_posix(),
            "lines": lines,
            "raw_bytes": len(raw.encode("utf-8")),
            "payload_bytes": len(payload.encode("utf-8")),
            "bundle_removed": bundled,
        })

    count = len(records)
    matrix = [[0.0] * count for _ in range(count)]
    condensed = []
    for i in range(count):
        for j in range(i + 1, count):
            value = normalized_distance(records[i]["lines"], records[j]["lines"])
            matrix[i][j] = matrix[j][i] = value
            condensed.append((records[i]["label"], records[j]["label"], value))

    linkage = average_linkage(matrix)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [record["label"] for record in records]
    write_csv(out_dir / "edit_distance_matrix.csv",
              [["script"] + labels] +
              [[labels[i]] + ["{:.9f}".format(value) for value in matrix[i]]
               for i in range(count)])
    write_csv(out_dir / "pairwise_distances.csv",
              [["script_a", "script_b", "distance"]] +
              [[a, b, "{:.9f}".format(d)] for a, b, d in condensed])
    write_csv(out_dir / "linkage_average.csv",
              [["left_cluster", "right_cluster", "distance", "member_count"]] +
              [[a, b, "{:.9f}".format(d), size]
               for a, b, d, size in linkage])
    write_csv(out_dir / "script_population.csv",
              [["script", "kind", "raw_bytes", "analyzed_bytes",
                "normalized_lines", "bundle_removed"]] +
              [[record["label"], Path(record["label"]).suffix[1:],
                record["raw_bytes"], record["payload_bytes"],
                len(record["lines"]), str(record["bundle_removed"]).lower()]
               for record in records])
    rendered = render_dendrogram(
        linkage, labels,
        [out_dir / "dendrogram.png", out_dir / "dendrogram.svg"],
    )
    print("analyzed {} scripts; dendrogram rendered: {}".format(count, rendered))


if __name__ == "__main__":
    main()
