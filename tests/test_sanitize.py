import json
from pathlib import Path

from ptts import sanitize


def test_normalize_text_unwraps_lines() -> None:
    text = "This is a line-\nbreak.\nNext line\n\nNew para."
    assert (
        sanitize.normalize_text(text)
        == "This is a line-break. Next line\n\nNew para."
    )


def test_normalize_text_unwraps_bracketed_dropcap() -> None:
    text = "[T]o see what is in front of one’s nose needs a constant struggle."
    assert (
        sanitize.normalize_text(text)
        == "To see what is in front of one's nose needs a constant struggle."
    )


def test_normalize_text_preserves_attribution_lines() -> None:
    text = (
        "Quote line one.\n"
        "—Author Name\n"
        "Another quote line.\n"
        "—Second Author\n"
    )
    assert (
        sanitize.normalize_text(text)
        == "Quote line one.\n—Author Name\nAnother quote line.\n—Second Author"
    )


def test_apply_remove_patterns_citation() -> None:
    text = "Some text (Smith, 2010, p. 5) continues."
    patterns = sanitize.compile_patterns(sanitize.DEFAULT_RULES["remove_patterns"])
    cleaned, _stats = sanitize.apply_remove_patterns(text, patterns)
    assert "(Smith, 2010, p. 5)" not in cleaned


def test_sanitize_book_adds_title_chapter(tmp_path: Path) -> None:
    book_dir = tmp_path / "book"
    raw_dir = book_dir / "raw" / "chapters"
    raw_dir.mkdir(parents=True)

    (raw_dir / "0001-title.txt").write_text("Title page", encoding="utf-8")
    (raw_dir / "0002-dedication.txt").write_text(
        "For someone", encoding="utf-8"
    )
    (raw_dir / "0003-ch1.txt").write_text("Chapter text", encoding="utf-8")

    toc = {
        "source_epub": "book.epub",
        "metadata": {
            "title": "Sample: Subtitle",
            "authors": ["Author One", "Author Two"],
            "year": "2024",
        },
        "chapters": [
            {
                "index": 1,
                "title": "Title",
                "path": "raw/chapters/0001-title.txt",
            },
            {
                "index": 2,
                "title": "Dedication",
                "path": "raw/chapters/0002-dedication.txt",
            },
            {
                "index": 3,
                "title": "Chapter 1",
                "path": "raw/chapters/0003-ch1.txt",
            },
        ],
    }
    (book_dir / "toc.json").write_text(
        json.dumps(toc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    written = sanitize.sanitize_book(book_dir, overwrite=True, base_dir=book_dir)
    clean_dir = book_dir / "clean" / "chapters"
    title_path = clean_dir / "0000-title.txt"

    assert written == 3
    assert title_path.exists()
    assert (clean_dir / "0001-title.txt").exists() is False

    title_text = title_path.read_text(encoding="utf-8").strip()
    assert (
        title_text
        == "Sample: Subtitle\n\n2024\n\nby Author One, Author Two"
    )

    clean_toc = json.loads((book_dir / "clean" / "toc.json").read_text())
    assert len(clean_toc["chapters"]) == 3
    assert clean_toc["chapters"][0]["kind"] == "title"


def test_sanitize_book_skips_title_for_txt(tmp_path: Path) -> None:
    book_dir = tmp_path / "book"
    raw_dir = book_dir / "raw" / "chapters"
    raw_dir.mkdir(parents=True)

    (raw_dir / "0001-preface.txt").write_text("Preface text", encoding="utf-8")
    (raw_dir / "0002-main.txt").write_text("Main text", encoding="utf-8")

    toc = {
        "source_epub": "notes.txt",
        "metadata": {
            "title": "Notes",
            "authors": ["Someone"],
            "year": "2024",
        },
        "chapters": [
            {
                "index": 1,
                "title": "Preface",
                "path": "raw/chapters/0001-preface.txt",
            },
            {
                "index": 2,
                "title": "Main",
                "path": "raw/chapters/0002-main.txt",
            },
        ],
    }
    (book_dir / "toc.json").write_text(
        json.dumps(toc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    written = sanitize.sanitize_book(book_dir, overwrite=True, base_dir=book_dir)
    clean_dir = book_dir / "clean" / "chapters"

    assert written == 2
    assert (clean_dir / "0000-title.txt").exists() is False

    clean_toc = json.loads((book_dir / "clean" / "toc.json").read_text())
    assert len(clean_toc["chapters"]) == 2
    assert all(entry.get("kind") != "title" for entry in clean_toc["chapters"])


def test_normalize_small_caps_opening_words() -> None:
    text = "IN MY FIRST year I learned a lot.\n\nSecond paragraph."
    expected = "In my first year I learned a lot.\n\nSecond paragraph."
    assert sanitize.normalize_small_caps(text) == expected


def test_normalize_small_caps_uses_case_map_for_names() -> None:
    text = (
        "EDGAR ALLAN POE AS A young man wrote a letter.\n\n"
        "Edgar Allan Poe is mentioned later."
    )
    expected = (
        "Edgar Allan Poe as a young man wrote a letter.\n\n"
        "Edgar Allan Poe is mentioned later."
    )
    assert sanitize.normalize_small_caps(text) == expected


def test_normalize_small_caps_preserves_acronyms() -> None:
    text = "USA TODAY REPORTS that this is fine. Today reports continue."
    expected = "USA Today reports that this is fine. Today reports continue."
    assert sanitize.normalize_small_caps(text) == expected


def test_normalize_small_caps_preserves_short_acronyms() -> None:
    text = "THE SQL QUERY is fast."
    expected = "The SQL query is fast."
    assert sanitize.normalize_small_caps(text) == expected
