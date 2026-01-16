from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub


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


def extract_metadata(book: epub.EpubBook) -> dict:
    title = _first_dc_meta(book, "title")
    authors = _all_dc_meta(book, "creator")
    language = _first_dc_meta(book, "language")

    cover_info = None
    cover_meta = book.get_metadata("OPF", "cover")
    if cover_meta:
        _value, attrs = cover_meta[0]
        cover_id = attrs.get("content") if attrs else None
        if cover_id:
            item = book.get_item_with_id(cover_id)
            if item:
                cover_info = {
                    "id": cover_id,
                    "href": _item_name(item),
                    "media_type": item.media_type,
                }

    return {
        "title": title,
        "authors": authors,
        "language": language,
        "cover": cover_info,
    }


def normalize_href(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    return href.split("#", 1)[0]


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


def html_to_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()
    root = soup.body if soup.body else soup
    text = root.get_text(separator="\n")
    return normalize_text(text)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:60] or "chapter"


def extract_chapters(book: epub.EpubBook, prefer_toc: bool = True) -> List[Chapter]:
    entries = build_toc_entries(book) if prefer_toc else []
    if not entries:
        entries = build_spine_entries(book)

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

        text = html_to_text(item.get_content())
        if not text:
            continue

        title = entry.title or _item_title(item) or Path(base_href).stem
        chapters.append(
            Chapter(title=title, href=entry.href, source=base_href, text=text)
        )

    return chapters
