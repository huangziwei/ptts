from pathlib import Path

import pytest

from ptts import epub as epub_util

EPUB_DIR = Path(__file__).resolve().parent / "epub"
CHANDLER_EPUB = EPUB_DIR / "Collected Stories of Raymond Chandler - Raymond Chandler.epub"
NEWS_EPUB = EPUB_DIR / "The News_ A User's Manual - Alain de Botton.epub"


def _load_book(path: Path) -> epub_util.epub.EpubBook:
    if not path.exists():
        pytest.skip(f"Missing test epub: {path}")
    return epub_util.read_epub(path)


def _split_series_counts(entries: list[epub_util.TocEntry]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for entry in entries:
        base_href = epub_util.normalize_href(entry.href)
        key = epub_util._split_series_key(base_href)
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _group_entries_by_key(
    entries: list[epub_util.TocEntry],
) -> dict[tuple[str, str], list[tuple[str, epub_util.TocEntry]]]:
    grouped: dict[tuple[str, str], list[tuple[str, epub_util.TocEntry]]] = {}
    for entry in entries:
        base_href = epub_util.normalize_href(entry.href)
        key = epub_util._split_series_key(base_href)
        if key:
            grouped.setdefault(key, []).append((base_href, entry))
    return grouped


def _find_unique_snippet(text: str, other: str, length: int = 40) -> str | None:
    if not text:
        return None
    limit = max(0, len(text) - length + 1)
    for idx in range(0, limit):
        snippet = text[idx : idx + length]
        if snippet.strip() and snippet not in other:
            return snippet
    return None


def test_html_to_text_dropcap() -> None:
    html = b"<html><body><p><span class='dropcap'>T</span>his book.</p></body></html>"
    assert epub_util.html_to_text(html) == "This book."


def test_html_to_text_removes_footnotes() -> None:
    html = (
        b"<html><body>"
        b"<p>Sentence<sup><a epub:type='noteref'>1</a></sup>.</p>"
        b"<aside epub:type='footnote'>Footnote text.</aside>"
        b"<p class='footnote'>Another footnote.</p>"
        b"</body></html>"
    )
    assert epub_util.html_to_text(html) == "Sentence."


def test_html_to_text_normalizes_modifier_apostrophe() -> None:
    html = "<html><body><p>It\u02bcs fine.</p></body></html>".encode("utf-8")
    assert epub_util.html_to_text(html) == "It's fine."


def test_html_to_text_normalizes_curly_apostrophe() -> None:
    html = "<html><body><p>It\u2019s fine.</p></body></html>".encode("utf-8")
    assert epub_util.html_to_text(html) == "It's fine."


def test_extract_chapters_preserves_multi_toc_split_series() -> None:
    book = _load_book(CHANDLER_EPUB)
    entries = epub_util.build_toc_entries(book)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    chapter_by_source = {chapter.source: chapter for chapter in chapters}
    counts = _split_series_counts(entries)
    grouped = _group_entries_by_key(entries)

    assert any(count > 1 for count in counts.values())

    for key, group in grouped.items():
        if counts.get(key, 0) <= 1:
            continue
        for base_href, _entry in group:
            item = book.get_item_with_href(base_href) or book.get_item_with_id(base_href)
            assert item is not None
            expected = epub_util.html_to_text(item.get_content())
            chapter = chapter_by_source.get(base_href)
            assert chapter is not None
            assert chapter.text == expected


def test_extract_chapters_merges_single_toc_split_series() -> None:
    book = _load_book(NEWS_EPUB)
    entries = epub_util.build_toc_entries(book)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    chapter_by_source = {chapter.source: chapter for chapter in chapters}
    toc_counts = _split_series_counts(entries)
    grouped = _group_entries_by_key(entries)

    spine_items = epub_util._build_spine_items(book)
    spine_series: dict[tuple[str, str], list[tuple[str, object]]] = {}
    for href, item in spine_items:
        key = epub_util._split_series_key(href)
        if key:
            spine_series.setdefault(key, []).append((href, item))

    candidates = [
        key
        for key, items in spine_series.items()
        if len(items) > 1 and toc_counts.get(key, 0) == 1
    ]
    assert candidates

    for key in candidates:
        group = grouped.get(key)
        if not group:
            continue
        base_href, _entry = group[0]
        chapter = chapter_by_source.get(base_href)
        assert chapter is not None

        first_href, first_item = spine_series[key][0]
        first_text = epub_util.html_to_text(first_item.get_content())
        assert len(chapter.text) > len(first_text)

        second_item = spine_series[key][1][1]
        second_text = epub_util.html_to_text(second_item.get_content())
        snippet = _find_unique_snippet(second_text, first_text)
        if snippet:
            assert snippet in chapter.text
