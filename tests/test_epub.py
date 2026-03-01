import re
from pathlib import Path

import pytest

from ptts import epub as epub_util

EPUB_DIR = Path(__file__).resolve().parent / "epub"
CHANDLER_EPUB = EPUB_DIR / "Collected Stories of Raymond Chandler - Raymond Chandler.epub"
NEWS_EPUB = EPUB_DIR / "The News_ A User's Manual - Alain de Botton.epub"


class _FakeDocumentItem:
    def __init__(self, name: str, title: str, html: bytes, item_id: str = "") -> None:
        self.file_name = name
        self.title = title
        self._content = html
        self.id = item_id or name

    def get_name(self) -> str:
        return self.file_name

    def get_title(self) -> str:
        return self.title

    def get_id(self) -> str:
        return self.id

    def get_type(self) -> int:
        return epub_util.ITEM_DOCUMENT

    def get_content(self) -> bytes:
        return self._content


class _FakeBook:
    def __init__(
        self,
        items: list[_FakeDocumentItem],
        spine: list[str] | None = None,
    ) -> None:
        self._items = list(items)
        self._by_href = {item.get_name(): item for item in self._items}
        self._by_id = {item.get_id(): item for item in self._items}
        if spine is not None:
            self.spine = [(sid, "yes") for sid in spine]
        else:
            self.spine = [(item.get_id(), "yes") for item in self._items]

    def get_item_with_href(self, href: str):
        return self._by_href.get(href)

    def get_item_with_id(self, item_id: str):
        return self._by_id.get(item_id)

    def get_items_of_type(self, item_type: int):
        if item_type != epub_util.ITEM_DOCUMENT:
            return []
        return list(self._items)


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


def test_collect_footnote_index_detects_notes_entries_with_anchor_ids() -> None:
    chapter_html = (
        b"<html><body>"
        b"<p>Body text<a href='notes.xhtml#anchor-note-1' class='class_nounder'>1</a>.</p>"
        b"</body></html>"
    )
    notes_html = (
        b"<html><body>"
        b"<p class='class_h5'>NOTES</p>"
        b"<p id='anchor-note-1' class='class_notes'>"
        b"<a href='chapter.xhtml#anchor-callout-1'>1.</a> Citation text."
        b"</p>"
        b"</body></html>"
    )
    book = _FakeBook(
        [
            _FakeDocumentItem("chapter.xhtml", "Chapter", chapter_html),
            _FakeDocumentItem("notes.xhtml", "Notes", notes_html),
        ]
    )
    footnote_index = epub_util._collect_footnote_index(book)

    assert "anchor-note-1" in footnote_index["note_ids"]
    assert (
        epub_util.html_to_text(
            chapter_html,
            footnote_index=footnote_index,
            source_href="chapter.xhtml",
        )
        == "Body text."
    )


def test_html_to_text_normalizes_modifier_apostrophe() -> None:
    html = "<html><body><p>It\u02bcs fine.</p></body></html>".encode("utf-8")
    assert epub_util.html_to_text(html) == "It's fine."


def test_html_to_text_normalizes_curly_apostrophe() -> None:
    html = "<html><body><p>It\u2019s fine.</p></body></html>".encode("utf-8")
    assert epub_util.html_to_text(html) == "It's fine."


def test_html_to_text_collapses_source_soft_wraps_inside_paragraph() -> None:
    html = (
        b"<html><body>"
        b"<p>First line of text\n"
        b" wrapped in source.\n"
        b" Still same paragraph.</p>"
        b"</body></html>"
    )
    assert (
        epub_util.html_to_text(html)
        == "First line of text wrapped in source. Still same paragraph."
    )


def test_html_to_text_preserves_explicit_br_linebreaks() -> None:
    html = b"<html><body><p>Line one<br/>Line two</p></body></html>"
    assert epub_util.html_to_text(html) == "Line one\nLine two"


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
            # Chapter contains at least the primary file's text; the fallback
            # merge may append orphaned spine items that follow it.
            assert chapter.text.startswith(expected)


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


def test_chapters_from_toc_entries_fallback_merges_multi_file_chapters() -> None:
    """When all split files share one prefix (count > 1), the fallback merge
    should capture spine items between consecutive TOC entries."""
    items = [
        _FakeDocumentItem(
            "book_split_000.html", "", b"<html><body><h1>Chapter One</h1></body></html>", "s0"
        ),
        _FakeDocumentItem(
            "book_split_001.html", "", b"<html><body><p>Chapter one body content.</p></body></html>", "s1"
        ),
        _FakeDocumentItem(
            "book_split_002.html", "", b"<html><body><p>More chapter one content.</p></body></html>", "s2"
        ),
        _FakeDocumentItem(
            "book_split_003.html", "", b"<html><body><h1>Chapter Two</h1></body></html>", "s3"
        ),
        _FakeDocumentItem(
            "book_split_004.html", "", b"<html><body><p>Chapter two body content.</p></body></html>", "s4"
        ),
    ]
    book = _FakeBook(items)
    entries = [
        epub_util.TocEntry(title="Chapter One", href="book_split_000.html"),
        epub_util.TocEntry(title="Chapter Two", href="book_split_003.html"),
    ]
    chapters = epub_util._chapters_from_toc_entries(book, entries)
    assert len(chapters) == 2
    assert "Chapter one body content" in chapters[0].text
    assert "More chapter one content" in chapters[0].text
    assert "Chapter two body content" in chapters[1].text


def test_chapters_from_toc_entries_last_entry_merges_to_end_of_spine() -> None:
    """The last TOC entry should merge all remaining spine items to end."""
    items = [
        _FakeDocumentItem(
            "book_split_000.html", "", b"<html><body><h1>Chapter One</h1></body></html>", "s0"
        ),
        _FakeDocumentItem(
            "book_split_001.html", "", b"<html><body><h1>Chapter Two</h1></body></html>", "s1"
        ),
        _FakeDocumentItem(
            "book_split_002.html", "", b"<html><body><p>Chapter two part one.</p></body></html>", "s2"
        ),
        _FakeDocumentItem(
            "book_split_003.html", "", b"<html><body><p>Chapter two part two.</p></body></html>", "s3"
        ),
    ]
    book = _FakeBook(items)
    entries = [
        epub_util.TocEntry(title="Chapter One", href="book_split_000.html"),
        epub_util.TocEntry(title="Chapter Two", href="book_split_001.html"),
    ]
    chapters = epub_util._chapters_from_toc_entries(book, entries)
    assert len(chapters) == 2
    assert "Chapter two part one" in chapters[1].text
    assert "Chapter two part two" in chapters[1].text


def test_chapters_from_toc_entries_split_series_merge_regression() -> None:
    """Single TOC entry pointing to a split series with count==1 should still
    merge all files in that series via the existing logic."""
    items = [
        _FakeDocumentItem(
            "intro.html", "", b"<html><body><p>Introduction.</p></body></html>", "intro"
        ),
        _FakeDocumentItem(
            "ch_split_000.html", "", b"<html><body><h1>Chapter</h1></body></html>", "s0"
        ),
        _FakeDocumentItem(
            "ch_split_001.html", "", b"<html><body><p>Chapter body.</p></body></html>", "s1"
        ),
        _FakeDocumentItem(
            "ch_split_002.html", "", b"<html><body><p>More chapter body.</p></body></html>", "s2"
        ),
    ]
    book = _FakeBook(items)
    entries = [
        epub_util.TocEntry(title="Introduction", href="intro.html"),
        epub_util.TocEntry(title="Chapter", href="ch_split_000.html"),
    ]
    chapters = epub_util._chapters_from_toc_entries(book, entries)
    assert len(chapters) == 2
    assert "Chapter body" in chapters[1].text
    assert "More chapter body" in chapters[1].text


def test_chapters_from_toc_entries_single_file_chapters_unaffected() -> None:
    """Single-file chapters (no split files) should remain unchanged."""
    items = [
        _FakeDocumentItem(
            "ch1.html", "", b"<html><body><p>Chapter one.</p></body></html>", "c1"
        ),
        _FakeDocumentItem(
            "ch2.html", "", b"<html><body><p>Chapter two.</p></body></html>", "c2"
        ),
    ]
    book = _FakeBook(items)
    entries = [
        epub_util.TocEntry(title="Chapter 1", href="ch1.html"),
        epub_util.TocEntry(title="Chapter 2", href="ch2.html"),
    ]
    chapters = epub_util._chapters_from_toc_entries(book, entries)
    assert len(chapters) == 2
    assert chapters[0].text == "Chapter one."
    assert chapters[1].text == "Chapter two."


def test_ingestion_report_detects_orphaned_items() -> None:
    """ingestion_report should detect spine items not captured in chapters."""
    items = [
        _FakeDocumentItem(
            "ch1.html", "", b"<html><body><p>Chapter one.</p></body></html>", "c1"
        ),
        _FakeDocumentItem(
            "ch2.html", "", b"<html><body><p>Orphaned content here.</p></body></html>", "c2"
        ),
    ]
    book = _FakeBook(items)
    chapters = [
        epub_util.Chapter(title="Ch1", href="ch1.html", source="ch1.html", text="Chapter one."),
    ]
    report = epub_util.ingestion_report(book, chapters)
    assert len(report["orphaned_items"]) == 1
    assert report["orphaned_items"][0]["href"] == "ch2.html"
    assert report["orphaned_chars"] > 0


def test_ingestion_report_no_orphans_after_merge() -> None:
    """After a successful merge, ingestion_report should show no orphans."""
    items = [
        _FakeDocumentItem(
            "book_split_000.html", "", b"<html><body><h1>Chapter One</h1></body></html>", "s0"
        ),
        _FakeDocumentItem(
            "book_split_001.html", "", b"<html><body><p>Chapter one body.</p></body></html>", "s1"
        ),
        _FakeDocumentItem(
            "book_split_002.html", "", b"<html><body><h1>Chapter Two</h1></body></html>", "s2"
        ),
        _FakeDocumentItem(
            "book_split_003.html", "", b"<html><body><p>Chapter two body.</p></body></html>", "s3"
        ),
    ]
    book = _FakeBook(items)
    entries = [
        epub_util.TocEntry(title="Chapter One", href="book_split_000.html"),
        epub_util.TocEntry(title="Chapter Two", href="book_split_002.html"),
    ]
    chapters = epub_util._chapters_from_toc_entries(book, entries)
    report = epub_util.ingestion_report(book, chapters)
    assert len(report["orphaned_items"]) == 0
    assert report["orphaned_chars"] == 0
