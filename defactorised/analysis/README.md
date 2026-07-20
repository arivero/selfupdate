# Code-genealogy analysis

This directory treats the defactorised entry points as a population of code
specimens and measures their structural proximity. It is intended to support
genealogy diagrams and selection of compact, representative demo scripts; it
does not claim that textual similarity establishes shared authorship or equal
runtime behavior.

## Population and reproducibility

The population is every `.py`, `.sh`, and `.sbatch` file recursively beneath
`defactorised/`, including the analysis generators themselves. Generated
artifacts have other suffixes and therefore cannot recursively enter the
population. Paths are sorted lexicographically, which fixes matrix and leaf
identifiers. Run:

```bash
/opt/ohpc/pub/apps/anaconda/anaconda-2025/bin/python3 \
  defactorised/analysis/code_genealogy.py
```

The analyzer itself requires only the Python standard library. The shown
cluster Python also has SciPy and matplotlib, so it renders the dendrogram; on
a minimal interpreter it still emits the complete matrix and linkage table.
No model, GPU, training worker, repository import, or network access is used.

## Normalization

For Python, the region delimited by `BEGIN/END GENERATED STANDALONE
SELFUPDATE BUNDLE` is removed before tokenization. This is generated packaging
infrastructure repeated across entry points and would otherwise dominate the
distance. Python comments and formatting-only newlines are discarded, string
and numeric literals are mapped to type markers, keywords and indentation are
marked, and identifiers/operators are retained. The resulting units are
normalized logical lines.

For shell and Slurm scripts, comments and blank lines are discarded except
semantic `#SBATCH` directives, backslash continuations are joined, shell words
are split without evaluating the shell, and standalone numeric words are
mapped to a type marker. Command names, option names, paths, and variable
expressions remain evidence.

## Distance and clustering

For normalized line sequences *A* and *B*, the matrix entry is

```text
d(A, B) = Levenshtein(A, B) / max(length(A), length(B)).
```

Insertions, deletions, and substitutions each cost one. Thus the matrix is
symmetric, its diagonal is zero, and values lie in `[0, 1]`. The exact edit
distance is computed with Myers's bit-vector recurrence: arbitrary-width
Python integers represent all positions of the shorter sequence. This avoids
naive character-level dynamic-programming tables for the embedded 400 KB
bundles while preserving exact unit-cost Levenshtein distance on the stated
analysis units.

Clustering uses deterministic average linkage (UPGMA). Cluster-to-cluster
distance is the size-weighted mean of predecessor distances. Ties are resolved
by numeric cluster identifiers. Initial identifiers `0..N-1` correspond to
the rows of `script_population.csv`; each linkage row creates the next
identifier beginning at `N`.

## Artifacts

- `artifacts/edit_distance_matrix.csv`: labeled symmetric square matrix.
- `artifacts/pairwise_distances.csv`: long-form upper triangle.
- `artifacts/linkage_average.csv`: SciPy-compatible four-column linkage data
  with an explanatory header.
- `artifacts/script_population.csv`: population, sizes, normalized line counts,
  and whether an embedded bundle was removed.
- `artifacts/dendrogram.png` and `artifacts/dendrogram.svg`: rendered
  right-oriented average-linkage trees when SciPy and matplotlib are present.

## Limitations

This is syntactic genealogy. Renaming an API changes evidence, while two
scripts with similar control skeletons can cluster despite different effects.
Literal normalization deliberately hides configuration-value differences.
Shell parsing is lexical rather than a full Bash AST. Distances across
languages are mathematically defined but usually less scientifically useful
than within-language subtrees. UPGMA imposes a hierarchy even when the data do
not arise from a tree, and its branch heights are dissimilarities rather than
evolutionary time. Runtime equivalence, safety, and scientific equivalence
must be assessed separately.
