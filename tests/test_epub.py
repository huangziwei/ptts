import re
from pathlib import Path

import pytest

from ptts import epub as epub_util

EPUB_DIR = Path(__file__).resolve().parent / "epub"
CHANDLER_EPUB = EPUB_DIR / "Collected Stories of Raymond Chandler - Raymond Chandler.epub"
NEWS_EPUB = EPUB_DIR / "The News_ A User's Manual - Alain de Botton.epub"


class _FakeDocumentItem:
    def __init__(self, name: str, title: str, html: bytes) -> None:
        self.file_name = name
        self.title = title
        self._content = html

    def get_name(self) -> str:
        return self.file_name

    def get_title(self) -> str:
        return self.title

    def get_type(self) -> int:
        return epub_util.ITEM_DOCUMENT

    def get_content(self) -> bytes:
        return self._content


class _FakeBook:
    def __init__(self, items: list[_FakeDocumentItem]) -> None:
        self._by_href = {item.get_name(): item for item in items}

    def get_item_with_href(self, href: str):
        return self._by_href.get(href)

    def get_item_with_id(self, _item_id: str):
        return None


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


def test_html_to_text_keeps_chapter_ids_when_backrefs_include_numeric_crossrefs() -> None:
    html = (
        b"<html><body>"
        b"<div id='ch01'>"
        b"<h1>Chapter One</h1>"
        b"<p>Body text<a id='fn1r' href='notes.xhtml#fn1'>1</a>.</p>"
        b"</div>"
        b"</body></html>"
    )
    # Regression guard: chapter IDs must not be removed even if mistakenly listed
    # as backrefs, while real note markers should still be stripped.
    text = epub_util.html_to_text(
        html,
        footnote_index={"note_ids": {"fn1"}, "backref_ids": {"ch01", "fn1"}},
    )
    assert "Chapter One" in text
    assert "Body text." in text


def test_html_to_text_normalizes_modifier_apostrophe() -> None:
    html = "<html><body><p>It\u02bcs fine.</p></body></html>".encode("utf-8")
    assert epub_util.html_to_text(html) == "It's fine."


def test_html_to_text_normalizes_curly_apostrophe() -> None:
    html = "<html><body><p>It\u2019s fine.</p></body></html>".encode("utf-8")
    assert epub_util.html_to_text(html) == "It's fine."


def test_html_to_text_preserves_heading_break_strength() -> None:
    html = (
        b"<html><body>"
        b"<h1>Main Title</h1>"
        b"<p>First paragraph.</p>"
        b"<h2>2.</h2>"
        b"<p>Second paragraph.</p>"
        b"</body></html>"
    )
    text = epub_util.html_to_text(html)
    assert "Main Title\n\n\n\n\nFirst paragraph." in text
    assert "2.\n\n\nSecond paragraph." in text


def test_html_to_text_infers_structural_heading_from_semantic_class() -> None:
    html = (
        b"<html><body>"
        b"<p class='chapter-title'>Chapter One</p>"
        b"<p>Body paragraph.</p>"
        b"</body></html>"
    )
    text = epub_util.html_to_text(html)
    assert "Chapter One\n\n\n\n\nBody paragraph." in text


def test_html_to_text_infers_structural_heading_from_semantic_role() -> None:
    html = (
        b"<html><body>"
        b"<p>Opening paragraph.</p>"
        b"<p role='doc-subtitle'>Subheading</p>"
        b"<p>Body paragraph.</p>"
        b"</body></html>"
    )
    text = epub_util.html_to_text(html)
    assert "Opening paragraph.\n\nSubheading\n\n\nBody paragraph." in text


def test_html_to_text_preserves_heading_breaks_inside_wrapper_div() -> None:
    html = (
        b"<html><body>"
        b"<div>"
        b"<h1>Wrapped Heading</h1>"
        b"<p>Body paragraph.</p>"
        b"</div>"
        b"</body></html>"
    )
    text = epub_util.html_to_text(html)
    assert "Wrapped Heading\n\n\n\n\nBody paragraph." in text


def test_html_to_text_infers_structural_heading_from_compound_title_classes() -> None:
    html = (
        b"<html><body>"
        b"<p class='chaptertitle'>1</p>"
        b"<p class='chaptersubtitle'>Introduction</p>"
        b"<p class='paraaftertitle'>Body paragraph.</p>"
        b"</body></html>"
    )
    text = epub_util.html_to_text(html)
    assert "1\n\n\nIntroduction\n\n\nBody paragraph." in text


def test_chapters_from_entries_infers_title_from_text_when_metadata_is_filename() -> None:
    href = "CR!chunk_split_002.html"
    item = _FakeDocumentItem(
        name=href,
        title=href,
        html=(
            b"<html><body>"
            b"<p>Title fallback should use the first few words from text.</p>"
            b"</body></html>"
        ),
    )
    chapters = epub_util._chapters_from_entries(
        _FakeBook([item]),
        [epub_util.TocEntry(title=href, href=href)],
    )
    assert chapters[0].title == "Title fallback should use the first few words from text"


def test_chapters_from_entries_infers_title_from_heading_when_metadata_is_filename() -> None:
    href = "chapter-001.xhtml"
    item = _FakeDocumentItem(
        name=href,
        title=href,
        html=(
            b"<html><body>"
            b"<h1>A Real Chapter Title</h1>"
            b"<p>Body text.</p>"
            b"</body></html>"
        ),
    )
    chapters = epub_util._chapters_from_entries(
        _FakeBook([item]),
        [epub_util.TocEntry(title=href, href=href)],
    )
    assert chapters[0].title == "A Real Chapter Title"


def test_chapters_from_entries_keeps_meaningful_entry_title() -> None:
    href = "chapter-001.xhtml"
    item = _FakeDocumentItem(
        name=href,
        title=href,
        html=b"<html><body><p>Body text.</p></body></html>",
    )
    chapters = epub_util._chapters_from_entries(
        _FakeBook([item]),
        [epub_util.TocEntry(title="Chapter 1", href=href)],
    )
    assert chapters[0].title == "Chapter 1"


def test_chapters_from_entries_appends_ellipsis_when_text_fallback_is_truncated() -> None:
    href = "chapter-002.xhtml"
    item = _FakeDocumentItem(
        name=href,
        title=href,
        html=(
            b"<html><body>"
            b"<p>One two three four five six seven eight nine ten eleven twelve.</p>"
            b"</body></html>"
        ),
    )
    chapters = epub_util._chapters_from_entries(
        _FakeBook([item]),
        [epub_util.TocEntry(title=href, href=href)],
    )
    assert chapters[0].title == "One two three four five six seven eight nine ten..."


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


def test_extract_chapters_preserves_section_breaks_in_news_epub() -> None:
    book = _load_book(NEWS_EPUB)
    chapters = epub_util.extract_chapters(book, prefer_toc=True)
    assert any(re.search(r"\n{3,}", chapter.text) for chapter in chapters)
