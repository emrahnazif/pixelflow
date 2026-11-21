"""Write docs/llms.txt, the index of the docs pages, from the pages' own headers.

Usage: python scripts/llms_txt.py

llms.txt ships inside the package (site-packages/pixelflow/docs/llms.txt), so a coding
agent can find the docs of the installed version without the website. It lists every
page with its title and `summary`, in the order of the website's menu. It is generated so
it cannot drift from the pages: tests/unit/test_packaging.py fails when it is out of date.
"""

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
OUTPUT = DOCS / "llms.txt"

HEADER = re.compile(r"\A# (?P<title>.+)\n\n<!--\n(?P<fields>.*?)\n-->\n", re.S)

INTRO = """\
These are the docs of the pixelflow version installed next to this file. Paths below are
relative to this file. Links inside the pages are written from the repository root:
/docs/tracking/zones.md is tracking/zones.md here. The same pages, with images, are at
https://emrah.fr/pixelflow/."""


def read_header(page: Path) -> tuple[str, dict[str, str]]:
    """The page's `# ` title and the `key: value` fields of its header comment."""
    match = HEADER.match(page.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"{page}: no '# Title' followed by a <!-- --> header comment")
    fields = dict(line.split(": ", 1) for line in match["fields"].splitlines() if ": " in line)
    return match["title"], fields


def menu(docs: Path = DOCS) -> list[tuple[str | None, list[Path]]]:
    """Pages grouped as the website's menu shows them: index.md, the other top-level
    pages, then each folder as a group, everything sorted by file name."""
    top = sorted(p for p in docs.glob("*.md") if p.name != "index.md")
    groups = [(None, [docs / "index.md", *top])]
    for folder in sorted(p for p in docs.iterdir() if p.is_dir() and p.name != "assets"):
        pages = sorted(folder.glob("*.md"))
        if pages:
            groups.append((folder.name.replace("-", " ").capitalize(), pages))
    return groups


def render(docs: Path = DOCS) -> str:
    """The content of llms.txt for the pages in *docs*."""
    title, fields = read_header(docs / "index.md")
    lines = [f"# {title}", "", f"> {fields['summary']}", "", INTRO]
    for group, pages in menu(docs):
        lines.append("")
        if group:
            lines += [f"## {group}", ""]
        for page in pages:
            page_title, page_fields = read_header(page)
            path = page.relative_to(docs).as_posix()
            lines.append(f"- [{page_title}]({path}): {page_fields['summary']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT.write_text(render(), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
