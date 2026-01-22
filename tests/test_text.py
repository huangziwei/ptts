from pathlib import Path

from ptts import text


def test_guess_title_from_path_heading(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("\n\n# Hello World\nBody", encoding="utf-8")
    assert text.guess_title_from_path(path) == "Hello World"


def test_guess_title_from_path_fallback(tmp_path: Path) -> None:
    path = tmp_path / "01-some-book.txt"
    path.write_text("No heading here", encoding="utf-8")
    assert text.guess_title_from_path(path) == "some book"
