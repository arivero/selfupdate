"""Pin imports to THIS tree's src/. The venv may be a symlink shared with a
sibling checkout (see AGENTS.md), and its editable install would otherwise
shadow this tree — tests must exercise the code they sit next to, exactly
like the scripts/ sys.path convention."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
