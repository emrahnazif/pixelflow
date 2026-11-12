"""Print one version's section of CHANGELOG.md as GitHub release notes.

Usage: python scripts/release_notes.py 0.7.0 > release-notes.md

Exits with status 1 when the changelog has no section for that version, so the
publish workflow can refuse a release nobody wrote notes for.

The changelog is hard-wrapped for reading in an editor. GitHub renders a release
body the way it renders a comment - every newline becomes a line break - so the
same text shown there comes out ragged, each paragraph broken at column 90. The
lines of a paragraph are therefore joined back together before printing.
"""

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"

# A line that begins a new block and so is never a continuation of the one above.
_BLOCK_START = re.compile(r"\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\|)")


def section(changelog: str, version: str) -> str:
    """The lines between `## [version]` and the next `## [` heading."""
    kept, inside = [], False
    for line in changelog.splitlines():
        if line.startswith("## ["):
            inside = line.startswith(f"## [{version}]")
            continue
        if inside:
            kept.append(line)
    return "\n".join(kept).strip("\n")


def unwrap(text: str) -> str:
    """Join the hard-wrapped lines of each paragraph and list item into one line."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            out.append(line)
        elif fenced or not line.strip() or _BLOCK_START.match(line):
            out.append(line)
        elif out and out[-1].strip() and not out[-1].lstrip().startswith(("```", "#", "|")):
            out[-1] += " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    version = sys.argv[1].removeprefix("v")
    notes = section(CHANGELOG.read_text(encoding="utf-8"), version)
    if not notes.strip():
        print(f"CHANGELOG.md has no section for {version}", file=sys.stderr)
        return 1
    print(unwrap(notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
