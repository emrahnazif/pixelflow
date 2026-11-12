"""
Unit tests for scripts/release_notes.py, which the publish workflow runs to turn a
changelog section into the body of a GitHub release.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pixelflow as pf

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release_notes.py"

spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
release_notes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_notes)

CHANGELOG = """# Changelog

## [1.2.0] - 2030-01-02

A paragraph that was
wrapped by hand.

### Added

- **A thing.** Its description runs
  over two lines.

  A second paragraph of the same
  item.
- Another thing.

```python
kept = [
    "as written",
]
```

## [1.1.0] - 2030-01-01

Older notes.
"""


class TestReleaseNotes:
    """Test changelog extraction and unwrapping."""

    def test_section_stops_at_the_next_version(self):
        notes = release_notes.section(CHANGELOG, "1.2.0")

        assert notes.startswith("A paragraph that was")
        assert "Older notes" not in notes and "## [" not in notes
        assert release_notes.section(CHANGELOG, "1.1.0") == "Older notes."

    def test_a_version_is_not_matched_by_its_prefix(self):
        assert release_notes.section(CHANGELOG, "1.2") == ""
        assert release_notes.section(CHANGELOG, "9.9.9") == ""

    def test_unwrap_joins_paragraphs_and_keeps_structure(self):
        assert release_notes.unwrap(release_notes.section(CHANGELOG, "1.2.0")).splitlines() == [
            "A paragraph that was wrapped by hand.",
            "",
            "### Added",
            "",
            "- **A thing.** Its description runs over two lines.",
            "",
            "  A second paragraph of the same item.",
            "- Another thing.",
            "",
            "```python",
            "kept = [",
            '    "as written",',
            "]",
            "```",
        ]

    def test_the_current_version_has_notes(self):
        """The publish workflow refuses a tag whose version has no changelog section."""
        result = subprocess.run([sys.executable, str(SCRIPT), f"v{pf.__version__}"],
                                capture_output=True, text=True)

        assert result.returncode == 0
        assert result.stdout.strip()

    def test_a_missing_version_fails(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "9.9.9"], capture_output=True, text=True)

        assert result.returncode == 1
        assert "no section for 9.9.9" in result.stderr
