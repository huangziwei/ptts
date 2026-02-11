from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from urllib.parse import unquote

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, ITEM_IMAGE, epub

_PARAGRAPH_BREAK = "\n\n"
_SECTION_BREAK = "\n\n\n"
_TITLE_BREAK = "\n\n\n\n\n"
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_NUMERIC_HEADING_RE = re.compile(r"^[\W_]*(?:\d+|[ivxlcdm]+)[\W_]*$", re.IGNORECASE)


@dataclass(frozen=True)
class TocEntry:
    title: str
    href: str


@dataclass(frozen=True)
class Chapter:
    title: str
    href: str
    source: str
    text: str


def read_epub(path: Path) -> epub.EpubBook:
    return epub.read_epub(str(path))


def _first_dc_meta(book: epub.EpubBook, name: str) -> str:
    items = book.get_metadata("DC", name)
    if not items:
        return ""
    value, _attrs = items[0]
    return value or ""


def _all_dc_meta(book: epub.EpubBook, name: str) -> List[str]:
    items = book.get_metadata("DC", name)
    values: List[str] = []
    for value, _attrs in items:
        if value:
            values.append(value)
    return values


def _item_name(item: object) -> str:
    get_name = getattr(item, "get_name", None)
    if callable(get_name):
        return get_name() or ""
    return getattr(item, "file_name", "") or ""


def _item_title(item: object) -> str:
    title = getattr(item, "title", "")
    if title:
        return title
    get_title = getattr(item, "get_title", None)
    if callable(get_title):
        value = get_title()
        if value:
            return value
    return ""


def _item_id(item: object) -> str:
    get_id = getattr(item, "get_id", None)
    if callable(get_id):
        value = get_id()
        if value:
            return value
    return getattr(item, "id", "") or ""


def _find_cover_item(book: epub.EpubBook) -> object | None:
    cover_meta = book.get_metadata("OPF", "cover")
    if cover_meta:
        _value, attrs = cover_meta[0]
        cover_id = attrs.get("content") if attrs else None
        if cover_id:
            item = book.get_item_with_id(cover_id)
            if item:
                return item

    opf_meta = book.get_metadata("OPF", "meta")
    for _value, attrs in opf_meta:
        if not attrs:
            continue
        if str(attrs.get("name") or "").lower() != "cover":
            continue
        cover_id = attrs.get("content")
        if cover_id:
            item = book.get_item_with_id(cover_id)
            if item:
                return item

    for item in book.get_items():
        props = getattr(item, "properties", []) or []
        if isinstance(props, str):
            props = [props]
        if any("cover-image" in prop for prop in props):
            return item

    for item in book.get_items_of_type(ITEM_IMAGE):
        name = _item_name(item).lower()
        item_id = _item_id(item).lower()
        if "cover" in name or "cover" in item_id:
            return item

    best_item = None
    best_size = 0
    for item in book.get_items_of_type(ITEM_IMAGE):
        try:
            data = item.get_content()
        except Exception:
            continue
        if not data:
            continue
        size = len(data)
        if size > best_size:
            best_size = size
            best_item = item

    if best_item:
        return best_item

    return None


def extract_metadata(book: epub.EpubBook) -> dict:
    title = _first_dc_meta(book, "title")
    authors = _all_dc_meta(book, "creator")
    language = _first_dc_meta(book, "language")
    dates = _all_dc_meta(book, "date")
    year = ""
    for value in dates:
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            year = match.group(0)
            break

    cover_info = None
    cover_item = _find_cover_item(book)
    if cover_item:
        cover_info = {
            "id": _item_id(cover_item),
            "href": _item_name(cover_item),
            "media_type": getattr(cover_item, "media_type", "") or "",
        }

    return {
        "title": title,
        "authors": authors,
        "language": language,
        "dates": dates,
        "year": year,
        "cover": cover_info,
    }


def extract_cover_image(book: epub.EpubBook) -> dict | None:
    cover_item = _find_cover_item(book)
    if not cover_item:
        return None
    data = cover_item.get_content()
    if not data:
        return None
    return {
        "data": data,
        "id": _item_id(cover_item),
        "href": _item_name(cover_item),
        "media_type": getattr(cover_item, "media_type", "") or "",
    }


def normalize_href(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    href = href.split("#", 1)[0]
    # Some EPUBs percent-encode filenames in TOC entries.
    return unquote(href)


def flatten_toc(toc: Iterable) -> List[TocEntry]:
    entries: List[TocEntry] = []

    def walk(nodes: Iterable) -> None:
        for node in nodes:
            if isinstance(node, epub.Link):
                if node.href:
                    entries.append(TocEntry(title=node.title or "", href=node.href))
            elif isinstance(node, epub.Section):
                if node.href:
                    entries.append(TocEntry(title=node.title or "", href=node.href))
                subitems = getattr(node, "subitems", None)
                if subitems:
                    walk(subitems)
            elif isinstance(node, (list, tuple)):
                walk(node)

    walk(toc)
    return entries


def build_toc_entries(book: epub.EpubBook) -> List[TocEntry]:
    toc = book.toc or []
    entries = flatten_toc(toc)
    return [e for e in entries if e.href]


def build_spine_entries(book: epub.EpubBook) -> List[TocEntry]:
    entries: List[TocEntry] = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if not item or item.get_type() != ITEM_DOCUMENT:
            continue
        title = _item_title(item) or _item_name(item) or ""
        entries.append(TocEntry(title=title, href=_item_name(item)))
    return entries


def _build_spine_items(book: epub.EpubBook) -> List[tuple[str, object]]:
    items: List[tuple[str, object]] = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if not item or item.get_type() != ITEM_DOCUMENT:
            continue
        name = _item_name(item)
        if not name:
            continue
        items.append((normalize_href(name), item))
    return items


def _split_series_key(href: str) -> tuple[str, str] | None:
    match = re.match(r"^(?P<prefix>.+?)(?:_split_|-split-)\d+(?P<ext>\.[^./]+)$", href)
    if not match:
        return None
    return (match.group("prefix"), match.group("ext"))


def _join_item_text(
    items: Iterable[object], footnote_index: dict[str, set[str]] | None = None
) -> str:
    parts: List[str] = []
    for item in items:
        text = html_to_text(
            item.get_content(),
            footnote_index=footnote_index,
            source_href=_item_name(item),
        )
        if text:
            parts.append(text)
    if not parts:
        return ""
    return normalize_text("\n\n".join(parts))


_NOTE_MARKER_RE = re.compile(r"^\d{1,4}[a-z]?[.)]?$", re.IGNORECASE)
_NOTE_ID_RE = re.compile(
    r"^(?:fn|footnote|endnote|note|noteref|fnref|en)[a-z0-9._:-]*$",
    re.IGNORECASE,
)


def _normalize_id(value: str) -> str:
    return (value or "").strip().lower()


def _href_fragment(href: str) -> str:
    if not href:
        return ""
    parts = href.split("#", 1)
    if len(parts) < 2:
        return ""
    return unquote(parts[1])


def _looks_like_note_marker(text: str) -> bool:
    if not text:
        return False
    return bool(_NOTE_MARKER_RE.match(text.strip()))


def _looks_like_note_id(value: str) -> bool:
    cleaned = _normalize_id(value)
    if not cleaned:
        return False
    if _NOTE_ID_RE.fullmatch(cleaned):
        return True
    # Some EPUBs use compact IDs such as n12/en9 for notes.
    return bool(re.fullmatch(r"(?:n|en)\d+[a-z]?", cleaned))


def _normalize_break_runs(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        count = len(match.group(0))
        if count >= len(_TITLE_BREAK):
            return _TITLE_BREAK
        if count >= len(_SECTION_BREAK):
            return _SECTION_BREAK
        return match.group(0)

    return re.sub(r"\n{3,}", repl, text)


def _is_title_heading(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return False
    return not bool(_NUMERIC_HEADING_RE.fullmatch(cleaned))


def _break_after_block(text: str, tag_name: str, is_first_block: bool) -> str:
    if tag_name in _HEADING_TAGS:
        if is_first_block and _is_title_heading(text):
            return _TITLE_BREAK
        return _SECTION_BREAK
    return _PARAGRAPH_BREAK


def _collect_footnote_index(
    book: epub.EpubBook,
) -> dict[str, set[str]]:
    note_ids: set[str] = set()
    backref_candidates: set[tuple[str, str]] = set()
    note_sources: set[str] = set()

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        content = item.get_content()
        if not content:
            continue
        source_href = normalize_href(_item_name(item))
        head = content.lstrip()[:512].lower()
        parser = (
            "lxml-xml"
            if (head.startswith(b"<?xml") or b"xmlns=" in head)
            else "lxml"
        )
        soup = BeautifulSoup(content, parser)
        is_note_source = False
        for tag in soup.find_all(attrs={"epub:type": "footnote"}):
            note_id = _normalize_id(str(tag.get("id") or ""))
            if note_id:
                note_ids.add(note_id)
                is_note_source = True
        for tag in soup.find_all(attrs={"role": "doc-footnote"}):
            note_id = _normalize_id(str(tag.get("id") or ""))
            if note_id:
                note_ids.add(note_id)
                is_note_source = True
        for tag in soup.find_all(["p", "section", "div", "aside", "ol", "ul", "li", "td"]):
            attrs = getattr(tag, "attrs", None)
            if attrs is None:
                continue
            classes = attrs.get("class", [])
            if isinstance(classes, str):
                classes = [classes]
            class_text = " ".join(classes).lower()
            id_text = str(attrs.get("id") or "")
            if "footnote" in class_text or "endnote" in class_text:
                note_id = _normalize_id(id_text)
                if note_id:
                    note_ids.add(note_id)
                is_note_source = True
            if id_text.lower().startswith(("fn", "footnote", "endnote")):
                note_id = _normalize_id(id_text)
                if note_id:
                    note_ids.add(note_id)
                    is_note_source = True

        for anchor in soup.find_all("a"):
            text = anchor.get_text(strip=True)
            if not _looks_like_note_marker(text):
                continue
            href = str(anchor.get("href") or "")
            fragment = _normalize_id(_href_fragment(href))
            if fragment:
                backref_candidates.add((normalize_href(href), fragment))
            parent = anchor.parent
            if parent and getattr(parent, "attrs", None):
                parent_id = _normalize_id(str(parent.get("id") or ""))
                if parent_id:
                    parent_text = parent.get_text(strip=True)
                    if parent_text == text:
                        note_ids.add(parent_id)

        if is_note_source and source_href:
            note_sources.add(source_href)

    backref_ids = {
        fragment
        for base_href, fragment in backref_candidates
        if fragment in note_ids
        or _looks_like_note_id(fragment)
        or (base_href and base_href in note_sources)
    }

    return {"note_ids": note_ids, "backref_ids": backref_ids}


def html_to_text(
    html: bytes,
    footnote_index: dict[str, set[str]] | None = None,
    source_href: str = "",
) -> str:
    head = html.lstrip()[:512].lower()
    parser = "lxml-xml" if (head.startswith(b"<?xml") or b"xmlns=" in head) else "lxml"
    soup = BeautifulSoup(html, parser)
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()
    for tag in soup.find_all("sup"):
        tag.decompose()
    for tag in soup.find_all(attrs={"epub:type": "noteref"}):
        tag.decompose()
    for tag in soup.find_all(attrs={"epub:type": "footnote"}):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": "doc-noteref"}):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": "doc-footnote"}):
        tag.decompose()
    if footnote_index:
        note_ids = footnote_index.get("note_ids", set())
        backref_ids = footnote_index.get("backref_ids", set())
        if note_ids or backref_ids:
            for tag in soup.find_all(attrs={"id": True}):
                attrs = getattr(tag, "attrs", None)
                if not attrs:
                    continue
                tag_id = _normalize_id(str(attrs.get("id") or ""))
                if (
                    tag_id
                    and tag_id in backref_ids
                    and (_looks_like_note_id(tag_id) or getattr(tag, "name", "") in {"a", "sup"})
                ):
                    tag.decompose()
            for anchor in soup.find_all("a"):
                text = anchor.get_text(strip=True)
                if not _looks_like_note_marker(text):
                    continue
                fragment = _normalize_id(
                    _href_fragment(str(anchor.get("href") or ""))
                )
                if fragment and (fragment in note_ids or fragment in backref_ids):
                    anchor.decompose()
    for tag in soup.find_all(["p", "section", "div", "aside", "ol", "ul", "li"]):
        attrs = getattr(tag, "attrs", None)
        if attrs is None:
            continue
        classes = attrs.get("class", [])
        if isinstance(classes, str):
            classes = [classes]
        class_text = " ".join(classes).lower()
        id_text = str(attrs.get("id") or "").lower()
        if (
            "footnote" in class_text
            or "endnote" in class_text
            or "copyright" in class_text
            or "credit" in class_text
        ):
            tag.decompose()
            continue
        if id_text.startswith(("fn", "footnote", "endnote")):
            tag.decompose()
    root = soup.body if soup.body else soup

    block_tags = {"p", "div", "li", "blockquote", "pre", *_HEADING_TAGS}

    blocks: List[tuple[str, str]] = []
    for elem in root.find_all(block_tags):
        if any(getattr(parent, "name", None) in block_tags for parent in elem.parents):
            continue
        for br in elem.find_all("br"):
            br.replace_with("\n")
        text = elem.get_text(separator="", strip=False)
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = _normalize_break_runs(text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = text.strip()
        if text:
            blocks.append((text, (elem.name or "").lower()))

    if blocks:
        parts: List[str] = []
        for idx, (block_text, tag_name) in enumerate(blocks):
            parts.append(block_text)
            if idx + 1 < len(blocks):
                parts.append(
                    _break_after_block(
                        block_text, tag_name, is_first_block=idx == 0
                    )
                )
        return normalize_text("".join(parts))

    text = root.get_text(separator="", strip=False)
    return normalize_text(text)


def normalize_text(text: str) -> str:
    text = (
        text.replace("\u02bc", "'")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u201e", '"')
        .replace("\u201f", '"')
        .replace("\u00ab", '"')
        .replace("\u00bb", '"')
    )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = _normalize_break_runs(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\.[ \t]*\.[ \t]*\.", "...", text)
    return text.strip()


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:60] or "chapter"


def _chapters_from_entries(
    book: epub.EpubBook,
    entries: Iterable[TocEntry],
    footnote_index: dict[str, set[str]] | None = None,
) -> List[Chapter]:
    seen: set[str] = set()
    chapters: List[Chapter] = []

    for entry in entries:
        base_href = normalize_href(entry.href)
        if not base_href or base_href in seen:
            continue
        seen.add(base_href)

        item = book.get_item_with_href(base_href)
        if not item:
            item = book.get_item_with_id(base_href)
        if not item or item.get_type() != ITEM_DOCUMENT:
            continue

        text = html_to_text(
            item.get_content(),
            footnote_index=footnote_index,
            source_href=_item_name(item),
        )
        if not text:
            continue

        title = entry.title or _item_title(item) or Path(base_href).stem
        chapters.append(
            Chapter(title=title, href=entry.href, source=base_href, text=text)
        )

    return chapters


def _chapters_from_toc_entries(
    book: epub.EpubBook,
    entries: Iterable[TocEntry],
    footnote_index: dict[str, set[str]] | None = None,
) -> List[Chapter]:
    spine_items = _build_spine_items(book)
    spine_index = {href: idx for idx, (href, _item) in enumerate(spine_items)}
    chapters: List[Chapter] = []
    seen: set[str] = set()
    split_series_counts: dict[tuple[str, str], int] = {}
    for entry in entries:
        base_href = normalize_href(entry.href)
        if not base_href:
            continue
        key = _split_series_key(base_href)
        if key:
            split_series_counts[key] = split_series_counts.get(key, 0) + 1

    for entry in entries:
        base_href = normalize_href(entry.href)
        if not base_href or base_href in seen:
            continue
        seen.add(base_href)

        start_idx = spine_index.get(base_href)
        merged_items: List[object] = []
        if start_idx is not None:
            key = _split_series_key(base_href)
            prev_key = (
                _split_series_key(spine_items[start_idx - 1][0])
                if start_idx > 0
                else None
            )
            if (
                key
                and key != prev_key
                and split_series_counts.get(key, 0) == 1
            ):
                idx = start_idx
                while idx < len(spine_items):
                    href, item = spine_items[idx]
                    if _split_series_key(href) != key:
                        break
                    merged_items.append(item)
                    idx += 1

        if merged_items:
            text = _join_item_text(merged_items, footnote_index=footnote_index)
            item_for_title = merged_items[0]
        else:
            item_for_title = None
            if start_idx is not None:
                item_for_title = spine_items[start_idx][1]
            if not item_for_title:
                item_for_title = book.get_item_with_href(base_href)
            if not item_for_title:
                item_for_title = book.get_item_with_id(base_href)
            if not item_for_title or item_for_title.get_type() != ITEM_DOCUMENT:
                continue
            text = html_to_text(
                item_for_title.get_content(),
                footnote_index=footnote_index,
                source_href=_item_name(item_for_title),
            )

        if not text:
            continue

        title = entry.title or _item_title(item_for_title) or Path(base_href).stem
        chapters.append(
            Chapter(title=title, href=entry.href, source=base_href, text=text)
        )

    return chapters


def extract_chapters(book: epub.EpubBook, prefer_toc: bool = True) -> List[Chapter]:
    footnote_index = _collect_footnote_index(book)
    entries = build_toc_entries(book) if prefer_toc else []
    chapters = (
        _chapters_from_toc_entries(book, entries, footnote_index)
        if entries
        else []
    )
    if entries and not chapters:
        chapters = _chapters_from_entries(book, entries, footnote_index)
    if not chapters:
        chapters = _chapters_from_entries(
            book, build_spine_entries(book), footnote_index
        )
    return chapters
