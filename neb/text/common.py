"""Language-agnostic text helpers shared by all normalizers.

This module holds:
  * punctuation / break constants used by both chunking (in neb.tts) and
    the TTS-time normalizers;
  * bracket / quote stripping;
  * Pali / Sanskrit transliteration (ASCII fallback used regardless of the
    book's language);
  * reading-override parsing, loading, merging, and application.

None of the logic here is English- or German-specific; per-language lexical
normalizers live in sibling modules and receive text that has already passed
through these pre-processing steps.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Punctuation sets and pause-break tuning
# ---------------------------------------------------------------------------

_CLAUSE_PUNCT = {",", ";", ":"}
_SENT_PUNCT = {".", "!", "?"}
_CLOSING_PUNCT = "\"')]}" + "”’"

_SECTION_BREAK_NEWLINES = 3
_TITLE_BREAK_NEWLINES = 5
_SECTION_BREAK_PAD_MULTIPLIER = 3
_TITLE_BREAK_PAD_MULTIPLIER = 5
_CHAPTER_BREAK_PAD_MULTIPLIER = _TITLE_BREAK_PAD_MULTIPLIER


# ---------------------------------------------------------------------------
# Quote + bracket stripping
# ---------------------------------------------------------------------------

_DOUBLE_QUOTE_CHARS = {'"', "“", "”", "«", "»", "„", "‟", "❝", "❞"}
_SINGLE_QUOTE_CHARS = {"'", "‘", "’", "‚", "‛"}
_LEADING_ELISIONS = {
    "tis",
    "twas",
    "twere",
    "twill",
    "til",
    "em",
    "cause",
    "bout",
    "round",
}

_BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}
_CLOSE_BRACKETS = set(_BRACKET_PAIRS.values())
_CLOSE_TO_OPEN = {v: k for k, v in _BRACKET_PAIRS.items()}


def _strip_brackets(text: str) -> str:
    if not text:
        return text
    stripped = text.strip()
    for open_b, close_b in _BRACKET_PAIRS.items():
        if stripped.startswith(open_b) and stripped.endswith(close_b):
            inner = stripped[1:-1]
            depth = 0
            matched = True
            for ch in inner:
                if ch == open_b:
                    depth += 1
                elif ch == close_b:
                    if depth == 0:
                        matched = False
                        break
                    depth -= 1
            if matched:
                stripped = inner
    for close_b in _CLOSE_BRACKETS:
        open_b = _CLOSE_TO_OPEN[close_b]
        if close_b in stripped and open_b not in stripped:
            stripped = stripped.replace(close_b, "")
    stripped = re.sub(r"([.!?,;:])[\)\]\}]", r"\1", stripped)
    return stripped


def _strip_double_quotes(text: str) -> str:
    if not text:
        return text
    return "".join(ch for ch in text if ch not in _DOUBLE_QUOTE_CHARS)


def _strip_single_quotes(text: str) -> str:
    if not text:
        return text
    out: List[str] = []
    for idx, ch in enumerate(text):
        if ch not in _SINGLE_QUOTE_CHARS:
            out.append(ch)
            continue
        prev = text[idx - 1] if idx > 0 else ""
        next_ch = text[idx + 1] if idx + 1 < len(text) else ""
        if prev and next_ch and prev.isalnum() and next_ch.isalnum():
            out.append(ch)
            continue
        if (not prev or not prev.isalnum()) and next_ch and next_ch.isalpha():
            end = idx + 1
            while end < len(text) and text[end].isalpha():
                end += 1
            word = text[idx + 1 : end].lower()
            if word in _LEADING_ELISIONS:
                out.append(ch)
                continue
        continue
    return "".join(out)


# ---------------------------------------------------------------------------
# Pali / Sanskrit transliteration
# ---------------------------------------------------------------------------

_PALI_SANSKRIT_ASCII_MAP = {
    "ā": "aa",
    "Ā": "Aa",
    "ī": "ii",
    "Ī": "Ii",
    "ū": "uu",
    "Ū": "Uu",
    "ṛ": "ri",
    "Ṛ": "Ri",
    "ṝ": "rii",
    "Ṝ": "Rii",
    "ḷ": "l",
    "Ḷ": "L",
    "ḹ": "lii",
    "Ḹ": "Lii",
    "ṃ": "m",
    "Ṃ": "M",
    "ṁ": "m",
    "Ṁ": "M",
    "ṅ": "ng",
    "Ṅ": "Ng",
    "ñ": "ny",
    "Ñ": "Ny",
    "ṭ": "t",
    "Ṭ": "T",
    "ḍ": "d",
    "Ḍ": "D",
    "ṇ": "n",
    "Ṇ": "N",
    "ś": "sh",
    "Ś": "Sh",
    "ṣ": "sh",
    "Ṣ": "Sh",
    "ḥ": "h",
    "Ḥ": "H",
}
_MACRON_VOWELS = set("aAiIuUeEoO")
_COMBINING_MACRON = "̄"


def _double_vowel(base: str) -> str:
    if base.isupper():
        return base + base.lower()
    return base + base


def _normalize_combining_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    out: List[str] = []
    i = 0
    while i < len(decomposed):
        ch = decomposed[i]
        if unicodedata.combining(ch):
            i += 1
            continue
        j = i + 1
        marks: List[str] = []
        while j < len(decomposed) and unicodedata.combining(decomposed[j]):
            marks.append(decomposed[j])
            j += 1
        if marks and _COMBINING_MACRON in marks and ch in _MACRON_VOWELS:
            out.append(_double_vowel(ch))
        else:
            out.append(ch)
        i = j
    return "".join(out)


def _transliterate_pali_sanskrit(text: str) -> str:
    if not text or text.isascii():
        return text
    for src, dst in _PALI_SANSKRIT_ASCII_MAP.items():
        if src in text:
            text = text.replace(src, dst)
    return _normalize_combining_diacritics(text)


# ---------------------------------------------------------------------------
# Linebreak → pause mapping (used by every language)
# ---------------------------------------------------------------------------

def _normalize_linebreak_pauses(text: str) -> str:
    if "\n" not in text:
        return text

    boundary_punct = _SENT_PUNCT | _CLAUSE_PUNCT | set(_CLOSING_PUNCT)

    def replace(match: re.Match[str]) -> str:
        start = match.start()
        end = match.end()
        prev_char = text[start - 1] if start > 0 else ""
        next_char = text[end] if end < len(text) else ""
        if not prev_char or not next_char:
            return " "

        newline_count = match.group(0).count("\n")
        if newline_count >= _SECTION_BREAK_NEWLINES:
            if prev_char in _SENT_PUNCT:
                return " "
            return ". "

        if prev_char in boundary_punct or next_char in boundary_punct:
            return " "
        return ", "

    return re.sub(r"[ \t]*\n+[ \t]*", replace, text)


# ---------------------------------------------------------------------------
# Reading overrides
# ---------------------------------------------------------------------------

READING_OVERRIDES_FILENAME = "reading-overrides.json"
_READING_MODES = {"all", "first", "word", "word_first"}
_READING_MODE_ALIASES = {
    "": "word",
    "all": "all",
    "first": "first",
    "word": "word",
    "word_first": "word_first",
    "once": "first",
    "substring": "all",
    "substring_first": "first",
}


def _normalize_reading_mode(value: object, *, default: str) -> str:
    cleaned = str(value or "").strip().lower()
    if not cleaned:
        return default
    mode = _READING_MODE_ALIASES.get(cleaned)
    if mode is None or mode not in _READING_MODES:
        raise ValueError(
            "Reading override mode must be one of: "
            "all, first, word, word_first."
        )
    return mode


def _normalize_reading_override_entry(raw: object) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None

    reading = str(
        raw.get("reading")
        or raw.get("replacement")
        or raw.get("to")
        or raw.get("value")
        or ""
    ).strip()
    if not reading:
        return None

    pattern = str(raw.get("pattern") or "").strip()
    base = str(raw.get("base") or raw.get("from") or raw.get("key") or "").strip()
    is_regex = bool(raw.get("regex"))
    case_sensitive = bool(raw.get("case_sensitive"))
    mode_raw = raw.get("mode")

    if pattern or (is_regex and base):
        if not pattern:
            pattern = base
        mode = _normalize_reading_mode(mode_raw, default="all")
        return {
            "pattern": pattern,
            "reading": reading,
            "mode": mode,
            "case_sensitive": case_sensitive,
        }

    if not base:
        return None

    mode = _normalize_reading_mode(mode_raw, default="word")
    return {
        "base": base,
        "reading": reading,
        "mode": mode,
        "case_sensitive": case_sensitive,
    }


def _parse_reading_entry_line(line: str) -> Optional[Dict[str, Any]]:
    raw = str(line or "").strip()
    if not raw or raw.startswith("#"):
        return None
    if "＝" in raw:
        base, reading = raw.split("＝", 1)
    elif "=" in raw:
        base, reading = raw.split("=", 1)
    else:
        return None
    return _normalize_reading_override_entry(
        {"base": base.strip(), "reading": reading.strip()}
    )


def _parse_reading_entries(raw: object) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        list_like = raw.get("replacements")
        if list_like is None:
            list_like = raw.get("entries")
        if list_like is None:
            list_like = [
                {"base": key, "reading": value}
                for key, value in raw.items()
                if isinstance(value, str)
            ]
    elif isinstance(raw, list):
        list_like = raw
    else:
        list_like = []

    entries: List[Dict[str, Any]] = []
    for item in list_like:
        entry: Optional[Dict[str, Any]] = None
        if isinstance(item, dict):
            entry = _normalize_reading_override_entry(item)
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            entry = _normalize_reading_override_entry(
                {"base": item[0], "reading": item[1]}
            )
        elif isinstance(item, str):
            entry = _parse_reading_entry_line(item)
        if entry:
            entries.append(entry)
    return entries


def _split_reading_overrides_data(
    data: object,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    global_entries: List[Dict[str, Any]] = []
    chapters: Dict[str, List[Dict[str, Any]]] = {}
    chapters_raw: object = {}

    if isinstance(data, list):
        global_entries = _parse_reading_entries(data)
    elif isinstance(data, dict):
        has_scoped_keys = any(
            key in data
            for key in ("global", "default", "*", "chapters", "replacements", "entries")
        )
        if "global" in data:
            global_entries = _parse_reading_entries(data.get("global"))
        elif "default" in data:
            global_entries = _parse_reading_entries(data.get("default"))
        elif "*" in data:
            global_entries = _parse_reading_entries(data.get("*"))
        elif "replacements" in data or "entries" in data:
            global_entries = _parse_reading_entries(data)
        elif not has_scoped_keys:
            global_entries = _parse_reading_entries(data)

        if "chapters" in data:
            chapters_raw = data.get("chapters") or {}

    if isinstance(chapters_raw, dict):
        for chapter_id, raw_entries in chapters_raw.items():
            chapter_entries = _parse_reading_entries(raw_entries)
            if chapter_entries:
                chapters[str(chapter_id)] = chapter_entries

    return global_entries, chapters


def _reading_overrides_path(book_dir: Path) -> Path:
    return book_dir / READING_OVERRIDES_FILENAME


def _load_reading_overrides(
    book_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    path = _reading_overrides_path(book_dir)
    if not path.exists():
        return [], {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    global_entries, chapter_entries = _split_reading_overrides_data(data)
    return global_entries, chapter_entries


def _merge_reading_overrides(
    global_overrides: Sequence[Dict[str, Any]],
    chapter_overrides: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not global_overrides and not chapter_overrides:
        return []

    merged: Dict[str, Dict[str, Any]] = {}

    def key_for(entry: Dict[str, Any]) -> str:
        pattern = str(entry.get("pattern") or "").strip()
        if pattern:
            case_key = "cs1" if bool(entry.get("case_sensitive")) else "cs0"
            mode = str(entry.get("mode") or "all")
            return f"re:{pattern}:{mode}:{case_key}"
        base = str(entry.get("base") or "").strip()
        case_sensitive = bool(entry.get("case_sensitive"))
        mode = str(entry.get("mode") or "word")
        if not case_sensitive:
            base = base.lower()
        case_key = "cs1" if case_sensitive else "cs0"
        return f"lit:{base}:{mode}:{case_key}"

    def add_items(items: Sequence[Dict[str, Any]]) -> None:
        for item in items:
            entry = _normalize_reading_override_entry(item)
            if not entry:
                continue
            merged[key_for(entry)] = entry

    add_items(global_overrides)
    add_items(chapter_overrides)
    return list(merged.values())


def _literal_override_pattern(base: str, mode: str) -> str:
    escaped = re.escape(base)
    if mode not in {"word", "word_first"}:
        return escaped
    return rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"


def apply_reading_overrides(text: str, overrides: Sequence[Dict[str, Any]]) -> str:
    if not text or not overrides:
        return text

    literals: List[Dict[str, Any]] = []
    regex_entries: List[Dict[str, Any]] = []
    for item in overrides:
        entry = _normalize_reading_override_entry(item)
        if not entry:
            continue
        if entry.get("pattern"):
            regex_entries.append(entry)
        else:
            literals.append(entry)

    out = text
    for item in sorted(literals, key=lambda e: len(str(e.get("base") or "")), reverse=True):
        base = str(item.get("base") or "")
        reading = str(item.get("reading") or "")
        mode = str(item.get("mode") or "word")
        if not base or not reading:
            continue
        pattern = _literal_override_pattern(base, mode)
        flags = 0 if bool(item.get("case_sensitive")) else re.IGNORECASE
        count = 1 if mode in {"first", "word_first"} else 0
        out = re.sub(pattern, lambda _m, value=reading: value, out, count=count, flags=flags)

    for item in regex_entries:
        pattern = str(item.get("pattern") or "")
        reading = str(item.get("reading") or "")
        mode = str(item.get("mode") or "all")
        if not pattern or not reading:
            continue
        flags = 0 if bool(item.get("case_sensitive")) else re.IGNORECASE
        count = 1 if mode in {"first", "word_first"} else 0
        try:
            out = re.sub(pattern, reading, out, count=count, flags=flags)
        except re.error:
            continue

    return out
