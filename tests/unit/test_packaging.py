"""
Guards for what the wheel ships: every code package, and an up-to-date docs/llms.txt.

pyproject.toml lists the packages by hand (the docs are mapped in from outside
pixelflow/, which automatic discovery cannot do), and docs/llms.txt repeats each page's
summary. Both would drift silently without these tests.
"""

import importlib.util
from pathlib import Path

import pytest
from setuptools import find_packages

ROOT = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location("llms_txt", ROOT / "scripts" / "llms_txt.py")
llms_txt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llms_txt)


def test_every_code_package_is_listed_in_pyproject():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    listed = set(config["tool"]["setuptools"]["packages"])
    found = set(find_packages(ROOT, include=["pixelflow", "pixelflow.*"]))
    assert listed - {"pixelflow.docs"} == found, (
        "pyproject.toml [tool.setuptools] packages must list exactly the packages under pixelflow/"
    )
    assert config["tool"]["setuptools"]["package-dir"]["pixelflow.docs"] == "docs"


def test_llms_txt_is_up_to_date():
    assert (ROOT / "docs" / "llms.txt").read_text(encoding="utf-8") == llms_txt.render(), (
        "docs/llms.txt is out of date: run python scripts/llms_txt.py"
    )


def test_llms_txt_lists_every_page_with_its_summary():
    text = llms_txt.render()
    pages = sorted((ROOT / "docs").rglob("*.md"))
    assert pages
    for page in pages:
        _, fields = llms_txt.read_header(page)
        path = page.relative_to(ROOT / "docs").as_posix()
        assert f"]({path}): {fields['summary']}\n" in text, path
