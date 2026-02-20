from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
import logging
import re
import shutil
import sys
import threading
import time
import unicodedata
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .text import read_clean_text
from .voice import DEFAULT_VOICE, resolve_voice_prompt

try:
    import torch
    from pocket_tts import TTSModel
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    TTSModel = None


_TTS_WARNING_CONTEXT = contextvars.ContextVar("_TTS_WARNING_CONTEXT", default=None)
_TTS_WARNING_FILTER_INSTALLED = False
_TTS_WARNING_CONTEXT_LOCK = threading.Lock()
_TTS_WARNING_CONTEXT_STACK: List[dict[str, Any]] = []


@contextmanager
def _tts_warning_context(
    chapter_id: str,
    chunk_idx: int,
    chunk_total: int,
    sub_idx: Optional[int] = None,
    sub_total: Optional[int] = None,
) -> Iterator[None]:
    ctx = {
        "chapter_id": chapter_id,
        "chunk_idx": chunk_idx,
        "chunk_total": chunk_total,
        "sub_idx": sub_idx,
        "sub_total": sub_total,
    }
    token = _TTS_WARNING_CONTEXT.set(ctx)
    with _TTS_WARNING_CONTEXT_LOCK:
        _TTS_WARNING_CONTEXT_STACK.append(ctx)
    try:
        yield
    finally:
        with _TTS_WARNING_CONTEXT_LOCK:
            for idx in range(len(_TTS_WARNING_CONTEXT_STACK) - 1, -1, -1):
                if _TTS_WARNING_CONTEXT_STACK[idx] is ctx:
                    _TTS_WARNING_CONTEXT_STACK.pop(idx)
                    break
        _TTS_WARNING_CONTEXT.reset(token)


def _active_tts_warning_context() -> Optional[dict]:
    ctx = _TTS_WARNING_CONTEXT.get()
    if ctx:
        return ctx
    with _TTS_WARNING_CONTEXT_LOCK:
        if _TTS_WARNING_CONTEXT_STACK:
            return _TTS_WARNING_CONTEXT_STACK[-1]
    return None


class _TTSWarningContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.WARNING:
            return True
        if record.name != "pocket_tts.models.tts_model":
            return True
        msg = record.getMessage()
        if "Maximum generation length reached without EOS" not in msg:
            return True
        ctx = _active_tts_warning_context()
        if not ctx:
            return True
        details = (
            f" [chapter={ctx['chapter_id']} chunk={ctx['chunk_idx']}/{ctx['chunk_total']}"
        )
        if ctx.get("sub_idx") is not None and ctx.get("sub_total") is not None:
            details += f" sub={ctx['sub_idx']}/{ctx['sub_total']}"
        details += "]"
        record.msg = f"{msg}{details}"
        record.args = ()
        return True


def _install_tts_warning_filter() -> None:
    global _TTS_WARNING_FILTER_INSTALLED
    if _TTS_WARNING_FILTER_INSTALLED:
        return
    logging.getLogger("pocket_tts.models.tts_model").addFilter(_TTSWarningContextFilter())
    _TTS_WARNING_FILTER_INSTALLED = True


_SENT_SPLIT_RE = re.compile(
    r"(?<=[.!?][\"')\]\}\u201d\u2019»])\s+|(?<=[.!?])\s+"
)
_ABBREV_DOT_RE = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Fig|Figs)\.", re.IGNORECASE)
_ABBREV_SENT_RE = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Fig|Figs)\.$", re.IGNORECASE)
_SINGLE_INITIAL_RE = re.compile(r"\b[A-Z]\.$")
_NAME_INITIAL_RE = re.compile(r"\b([A-Z][a-z]+)\s+[A-Z]\.$")
_MULTI_INITIAL_RE = re.compile(r"(?:\b[A-Z]\.\s*){2,}$")
_VOL_NO_ABBREV_RE = re.compile(r"\b(?:vol|no|nos)\.$", re.IGNORECASE)
_VOL_NO_FOLLOW_RE = re.compile(
    r"""^[\"'(\[]*(?:\d|[IVXLCDM]+\b|[A-Za-z](?:\d+|[.-]\d+)?\b)""",
    re.IGNORECASE,
)
_ABBREV_WHITELIST = {
    "a.a.",
    "a.e.",
    "a.k.a.",
    "a.m.",
    "b.f.",
    "b.a.",
    "b.f.a.",
    "b.s.",
    "b.s.n.",
    "c.g.",
    "c.e.o.",
    "c.f.o.",
    "c.i.a.",
    "c.o.o.",
    "c.p.a.",
    "c.s.",
    "cf.",
    "d.c.",
    "d.d.s.",
    "d.o.",
    "d.v.m.",
    "d.h.",
    "e.e.",
    "e.g.",
    "e.m.",
    "et al.",
    "etc.",
    "f.b.i.",
    "g.k.",
    "h.g.",
    "h.p.",
    "i.e.",
    "j.d.",
    "j.b.",
    "j.f.",
    "j.g.",
    "j.k.",
    "j.m.",
    "j.r.r.",
    "l.p.n.",
    "l.m.",
    "m.a.",
    "m.b.a.",
    "m.d.",
    "m.f.a.",
    "m.p.h.",
    "m.r.",
    "m.s.",
    "m.s.w.",
    "p.e.",
    "p.g.",
    "p.m.",
    "ph.d.",
    "r.n.",
    "r.l.",
    "t.e.",
    "t.s.",
    "v.s.",
    "u.k.",
    "u.n.",
    "u.s.",
    "u.s.a.",
    "viz.",
    "vs.",
    "w.b.",
    "w.e.b.",
    "w.h.",
}
_DOT_SPACE_DOT_RE = re.compile(r"(?<=\.)\s+(?=[A-Za-z]\.)")
_LAST_DOT_TOKEN_RE = re.compile(r"([A-Za-z][A-Za-z'-]*\.)\s*$")
_NEXT_DOT_TOKEN_RE = re.compile(r"([A-Za-z][A-Za-z'-]*\.)")
_ELLIPSIS_RE = re.compile(r"(\.\.\.|…)\s*$")
_ABBREV_EXPANSIONS = {
    "prof": "professor",
    "fig": "figure",
    "figs": "figures",
    "i.e": "that is",
    "e.g": "for example",
    "etc": "et cetera",
    "vs": "versus",
    "viz": "namely",
}
_ABBREV_EXPANSION_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, _ABBREV_EXPANSIONS)) + r")\.",
    re.IGNORECASE,
)
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
_SENTENCE_STARTERS = {
    "the",
    "a",
    "an",
    "and",
    "but",
    "or",
    "so",
    "yet",
    "for",
    "nor",
    "in",
    "on",
    "at",
    "by",
    "to",
    "from",
    "with",
    "without",
    "as",
    "if",
    "when",
    "while",
    "after",
    "before",
    "because",
    "since",
    "however",
    "therefore",
    "thus",
    "then",
    "this",
    "that",
    "these",
    "those",
    "i",
    "we",
    "you",
    "he",
    "she",
    "it",
    "they",
    "there",
}
_INITIAL_STOPWORDS = {
    "chapter",
    "section",
    "figure",
    "fig",
    "table",
    "appendix",
    "part",
    "volume",
    "vol",
    "no",
    "nos",
    "item",
    "book",
    "act",
}
_CLAUSE_PUNCT = {",", ";", ":"}
_SENT_PUNCT = {".", "!", "?"}
_CLOSING_PUNCT = "\"')]}"+ "\u201d\u2019"
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_CANONICAL_RE = re.compile(
    r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)
_ROMAN_HEADING_RE = re.compile(
    r"\b(?P<label>(?:chapter|book|part|volume|vol|section|act|appendix)\.?)"
    r"\s+(?P<num>[IVXLCDM]+)\b",
    re.IGNORECASE,
)
_ROMAN_LEADING_TITLE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<num>[IVXLCDM]+)(?P<trail>[^A-Za-z0-9\n]*)[ \t]*\n"
    r"(?=[ \t]*[A-Z])",
    re.IGNORECASE,
)
_ROMAN_STANDALONE_RE = re.compile(r"^(?P<num>[IVXLCDM]+)(?P<trail>[^A-Za-z0-9]*)$", re.IGNORECASE)
_ROMAN_I_DETERMINERS = {
    "a",
    "an",
    "another",
    "any",
    "each",
    "every",
    "his",
    "her",
    "its",
    "my",
    "no",
    "our",
    "some",
    "that",
    "the",
    "their",
    "this",
    "your",
}
_ROMAN_HEADING_TRAIL_PUNCT = (
    _SENT_PUNCT
    | _CLAUSE_PUNCT
    | set(_CLOSING_PUNCT)
    | {"-", "\u2013", "\u2014"}
)
_SECTION_BREAK_NEWLINES = 3
_TITLE_BREAK_NEWLINES = 5
_SECTION_BREAK_PAD_MULTIPLIER = 3
_TITLE_BREAK_PAD_MULTIPLIER = 5
_CHAPTER_BREAK_PAD_MULTIPLIER = _TITLE_BREAK_PAD_MULTIPLIER
_WORD_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_WORD_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_NUMBER_LABEL_RE = re.compile(
    r"\b(?P<label>(?:fig(?:ure)?|table|chapter|section|part|vol(?:ume)?|no|appendix|eq|equation))\.?"
    r"\s+(?P<num>\d+(?:\.\d+)+)\b",
    re.IGNORECASE,
)
_GROUPED_INT_COMMA_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_GROUPED_INT_DOT_RE = re.compile(r"\b\d{1,3}(?:\.\d{3})+\b")
_DECIMAL_RE = re.compile(r"\b\d+\.\d+\b")
_PLAIN_INT_RE = re.compile(r"\b\d+\b")
_SIGNED_INT_RE = re.compile(r"(?<!\w)(?P<sign>[+-])(?P<num>\d+)\b")
_YEAR_RANGE_RE = re.compile(
    r"\b(?P<start>1\d{3}|20\d{2})\s*[–-]\s*(?P<end>\d{2,4})\b(?!\s*[–-]\s*\d)"
)
_YEAR_RE = re.compile(r"\b(?P<year>1\d{3}|20\d{2})\b")
_ORDINAL_RE = re.compile(r"\b(?P<num>\d+)(?P<suffix>st|nd|rd|th)\b", re.IGNORECASE)
_RESIDUAL_DIGITS_RE = re.compile(r"\d+")
_ROMAN_DECIMAL_RE = re.compile(r"\b(?P<roman>[IVXLCDM]+)\.(?P<num>\d+(?:\.\d+)*)\b")
_CURRENCY_SYMBOL_UNITS = {
    "$": ("dollar", "dollars"),
    "€": ("euro", "euros"),
    "£": ("pound", "pounds"),
    "¥": ("yen", "yen"),
    "₹": ("rupee", "rupees"),
    "₽": ("ruble", "rubles"),
    "₩": ("won", "won"),
    "₪": ("shekel", "shekels"),
    "₫": ("dong", "dong"),
    "₴": ("hryvnia", "hryvnias"),
    "₦": ("naira", "naira"),
    "฿": ("baht", "baht"),
    "₺": ("lira", "lira"),
    "₱": ("peso", "pesos"),
}
_CURRENCY_SYMBOL_CLASS = "".join(re.escape(symbol) for symbol in _CURRENCY_SYMBOL_UNITS)
_CURRENCY_PREFIX_RE = re.compile(
    rf"(?<!\w)(?P<sym>[{_CURRENCY_SYMBOL_CLASS}])\s*"
    r"(?P<amount>[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))"
)
_CURRENCY_SUFFIX_RE = re.compile(
    r"(?P<amount>[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*"
    rf"(?P<sym>[{_CURRENCY_SYMBOL_CLASS}])(?!\w)"
)
_ERA_DOTTED_REPLACEMENTS = (
    (re.compile(r"\bB\s*\.\s*C\s*\.\s*E\s*\.?", re.IGNORECASE), "B-C-E"),
    (re.compile(r"\bA\s*\.\s*D\s*\.?", re.IGNORECASE), "A-D"),
    (re.compile(r"\bC\s*\.\s*E\s*\.?", re.IGNORECASE), "C-E"),
    (re.compile(r"\bB\s*\.\s*C\s*\.(?!\s*E\s*\.?)", re.IGNORECASE), "B-C"),
)
_ERA_PLAIN_WITH_YEAR_RE = re.compile(
    r"(?P<year>\b\d{1,4}(?:\s*[–-]\s*\d{1,4})?)\s+(?P<era>BCE|CE|BC|AD)\b"
)
_ERA_PLAIN_BEFORE_YEAR_RE = re.compile(
    r"\b(?P<era>BCE|CE|BC|AD)\s+(?P<year>\d{1,4}(?:\s*[–-]\s*\d{1,4})?)\b"
)
_SCALE_WORDS = (
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
)
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


@dataclass
class ChapterInput:
    index: int
    id: str
    title: str
    text: str
    path: Optional[str] = None


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:60] or "chapter"


def chapter_id_from_path(index: int, title: str, rel_path: Optional[str]) -> str:
    if rel_path:
        stem = Path(rel_path).stem
        if stem:
            return stem
    return f"{index:04d}-{slugify(title or 'chapter')}"


# ----------------------------
# Text chunking
# ----------------------------

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _trim_span(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def split_paragraph_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"\n\s*\n+", text):
        end = match.start()
        span = _trim_span(text, start, end)
        if span:
            spans.append(span)
        start = match.end()
    span = _trim_span(text, start, len(text))
    if span:
        spans.append(span)
    return spans


def split_sentence_spans(paragraph: str, offset: int) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    for match in _SENT_SPLIT_RE.finditer(paragraph):
        end = match.start()
        if _should_skip_sentence_split(paragraph, end, match.end()):
            continue
        span = _trim_span(paragraph, start, end)
        if span:
            spans.append((offset + span[0], offset + span[1]))
        start = match.end()
    span = _trim_span(paragraph, start, len(paragraph))
    if span:
        spans.append((offset + span[0], offset + span[1]))
    return spans


def _coerce_span_pairs(spans: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        try:
            start = int(span[0])
            end = int(span[1])
        except (TypeError, ValueError):
            continue
        if start < 0 or end < start:
            continue
        pairs.append((start, end))
    return pairs


def _pause_multiplier_from_gap(gap: str) -> int:
    if not gap:
        return 1
    max_run = 0
    for match in re.finditer(r"\n+", gap):
        max_run = max(max_run, len(match.group(0)))
    if max_run >= _TITLE_BREAK_NEWLINES:
        return _TITLE_BREAK_PAD_MULTIPLIER
    if max_run >= _SECTION_BREAK_NEWLINES:
        return _SECTION_BREAK_PAD_MULTIPLIER
    return 1


def _gap_has_symbolic_separator(gap: str) -> bool:
    for line in gap.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(ch.isalnum() for ch in stripped):
            continue
        return True
    return False


def compute_chunk_pause_multipliers(
    text: str, spans: Sequence[Tuple[int, int]]
) -> List[int]:
    if not spans:
        return []
    multipliers = [1] * len(spans)
    for idx in range(len(spans) - 1):
        end = int(spans[idx][1])
        next_start = int(spans[idx + 1][0])
        if next_start < end:
            continue
        gap = text[end:next_start]
        pause = _pause_multiplier_from_gap(gap)
        if "\n" in gap and _gap_has_symbolic_separator(gap):
            pause = max(pause, _SECTION_BREAK_PAD_MULTIPLIER)
        multipliers[idx] = pause
    return multipliers


def _normalize_pause_multipliers(
    pause_multipliers: object, chunk_count: int, fallback: Optional[Sequence[int]] = None
) -> List[int]:
    if chunk_count <= 0:
        return []
    normalized = [1] * chunk_count
    if isinstance(fallback, Sequence):
        for idx in range(min(chunk_count, len(fallback))):
            try:
                parsed = int(fallback[idx])
            except (TypeError, ValueError):
                parsed = 1
            normalized[idx] = parsed if parsed > 0 else 1
    if isinstance(pause_multipliers, list) and len(pause_multipliers) == chunk_count:
        for idx, value in enumerate(pause_multipliers):
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                normalized[idx] = parsed
    return normalized


def _apply_chapter_boundary_pause_multipliers(manifest_chapters: Sequence[dict]) -> None:
    for idx, entry in enumerate(manifest_chapters):
        if idx >= len(manifest_chapters) - 1:
            break
        if not isinstance(entry, dict):
            continue
        chunks = entry.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            continue
        normalized = _normalize_pause_multipliers(
            entry.get("pause_multipliers"), len(chunks)
        )
        normalized[-1] = max(normalized[-1], _CHAPTER_BREAK_PAD_MULTIPLIER)
        entry["pause_multipliers"] = normalized


def _next_word(text: str, start: int) -> str:
    match = re.search(r"[A-Za-z][A-Za-z'’\-]*", text[start:])
    if not match:
        return ""
    return match.group(0)


def _ends_with_whitelisted_abbrev(text: str) -> bool:
    text = text.lower()
    for abbr in _ABBREV_WHITELIST:
        if text.endswith(abbr):
            return True
    return False


def _ends_with_etc(tail: str) -> bool:
    stripped = tail.rstrip(_CLOSING_PUNCT + "»")
    return stripped.lower().endswith("etc.")


def _ends_with_ellipsis(tail: str) -> bool:
    stripped = tail.rstrip(_CLOSING_PUNCT + "»")
    return bool(_ELLIPSIS_RE.search(stripped))


def _is_whitelisted_abbrev_boundary(tail: str, paragraph: str, next_pos: int) -> bool:
    if _ends_with_whitelisted_abbrev(tail):
        return True
    if _DOT_SPACE_DOT_RE.search(tail):
        joined = _DOT_SPACE_DOT_RE.sub("", tail)
        if _ends_with_whitelisted_abbrev(joined):
            return True
    last_token = _LAST_DOT_TOKEN_RE.search(tail)
    if not last_token:
        return False
    next_token = _NEXT_DOT_TOKEN_RE.match(paragraph[next_pos:])
    if not next_token:
        return False
    combined = (last_token.group(1) + next_token.group(1)).lower()
    for abbr in _ABBREV_WHITELIST:
        if abbr.startswith(combined):
            return True
    return False


def _should_skip_sentence_split(paragraph: str, end: int, next_pos: int) -> bool:
    tail = paragraph[:end]
    next_word = _next_word(paragraph, next_pos)
    next_lower = next_word.lower()

    if _ends_with_ellipsis(tail):
        if next_word and next_word[0].islower():
            return True

    if _VOL_NO_ABBREV_RE.search(tail):
        if _VOL_NO_FOLLOW_RE.match(paragraph[next_pos:]):
            return True

    if _is_whitelisted_abbrev_boundary(tail, paragraph, next_pos):
        if _ends_with_etc(tail) and next_word and next_word[0].isupper():
            if next_lower in _SENTENCE_STARTERS:
                return False
        return True

    if _ABBREV_SENT_RE.search(tail):
        if next_lower and next_lower in _SENTENCE_STARTERS:
            return False
        return True

    if _MULTI_INITIAL_RE.search(tail):
        if next_word and next_word[0].islower():
            return True
        if next_lower and next_lower in _SENTENCE_STARTERS:
            return False
        return True

    if not _SINGLE_INITIAL_RE.search(tail):
        return False

    if next_word and next_word[0].islower():
        return True

    if next_lower and next_lower in _SENTENCE_STARTERS:
        return False

    name_match = _NAME_INITIAL_RE.search(tail)
    if name_match and next_word:
        prev_word = name_match.group(1).lower()
        if prev_word not in _INITIAL_STOPWORDS:
            return True

    if len(next_word) == 1:
        return True

    return False


def _ends_with_clause_punct(token: str) -> bool:
    if not token:
        return False
    stripped = token.rstrip(_CLOSING_PUNCT)
    if not stripped:
        return False
    return stripped[-1] in _CLAUSE_PUNCT


def _ends_with_sentence_punct(text: str) -> bool:
    if not text:
        return False
    stripped = text.rstrip(_CLOSING_PUNCT + "»")
    if not stripped:
        return False
    return stripped[-1] in _SENT_PUNCT


def _span_has_speakable_text(text: str, start: int, end: int) -> bool:
    for ch in text[start:end]:
        if ch.isalnum():
            return True
    return False


def split_span_by_words(
    text: str, start: int, end: int, max_chars: int
) -> List[Tuple[int, int]]:
    segment = text[start:end]
    words = list(re.finditer(r"\S+", segment))
    if not words:
        return []
    spans: List[Tuple[int, int]] = []
    idx = 0
    chunk_start = start + words[0].start()
    chunk_end = start + words[0].end()
    last_fit_idx = 0
    last_punct_idx = 0 if _ends_with_clause_punct(words[0].group()) else None
    idx = 1
    while idx < len(words):
        word = words[idx]
        word_start = start + word.start()
        word_end = start + word.end()
        if word_end - chunk_start > max_chars and chunk_end > chunk_start:
            split_idx = last_punct_idx if last_punct_idx is not None else last_fit_idx
            split_end = start + words[split_idx].end()
            if split_end > chunk_start:
                spans.append((chunk_start, split_end))
                idx = split_idx + 1
                if idx >= len(words):
                    return spans
                chunk_start = start + words[idx].start()
                chunk_end = start + words[idx].end()
                last_fit_idx = idx
                last_punct_idx = idx if _ends_with_clause_punct(words[idx].group()) else None
                idx += 1
                continue
        chunk_end = word_end
        last_fit_idx = idx
        if _ends_with_clause_punct(word.group()):
            last_punct_idx = idx
        idx += 1
    spans.append((chunk_start, chunk_end))
    return spans


def make_chunk_spans(
    text: str, max_chars: int, chunk_mode: str = "sentence"
) -> List[Tuple[int, int]]:
    if chunk_mode != "sentence":
        chunk_mode = "sentence"

    spans: List[Tuple[int, int]] = []
    for para_start, para_end in split_paragraph_spans(text):
        paragraph = text[para_start:para_end]
        sentence_spans = split_sentence_spans(paragraph, para_start)
        for sent_start, sent_end in sentence_spans:
            if sent_end - sent_start > max_chars:
                spans.extend(split_span_by_words(text, sent_start, sent_end, max_chars))
            else:
                spans.append((sent_start, sent_end))
    return [span for span in spans if _span_has_speakable_text(text, *span)]


def make_chunks(text: str, max_chars: int, chunk_mode: str = "sentence") -> List[str]:
    spans = make_chunk_spans(text, max_chars=max_chars, chunk_mode=chunk_mode)
    return [text[start:end] for start, end in spans]


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
_COMBINING_MACRON = "\u0304"


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


def _expand_abbreviations(text: str) -> str:
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        expansion = _ABBREV_EXPANSIONS.get(token.lower())
        if not expansion:
            return match.group(0)
        if token.isupper():
            return expansion.upper()
        if token[0].isupper():
            return expansion.capitalize()
        return expansion

    return _ABBREV_EXPANSION_RE.sub(replace, text)


def normalize_abbreviations(text: str) -> str:
    text = _expand_abbreviations(text)
    return _ABBREV_DOT_RE.sub(r"\1", text)


def _roman_to_int(value: str) -> Optional[int]:
    roman = value.upper()
    if not roman or not _ROMAN_CANONICAL_RE.fullmatch(roman):
        return None
    total = 0
    prev = 0
    for ch in reversed(roman):
        number = _ROMAN_VALUES.get(ch)
        if number is None:
            return None
        if number < prev:
            total -= number
        else:
            total += number
            prev = number
    return total or None


def _int_to_words(value: int) -> str:
    if value < 0:
        return f"minus {_int_to_words(abs(value))}"
    if value < 20:
        return _WORD_ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        if ones == 0:
            return _WORD_TENS[tens]
        return f"{_WORD_TENS[tens]} {_WORD_ONES[ones]}"
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        if rest == 0:
            return f"{_WORD_ONES[hundreds]} hundred"
        return f"{_WORD_ONES[hundreds]} hundred {_int_to_words(rest)}"
    for scale, label in _SCALE_WORDS:
        if value >= scale:
            major, rest = divmod(value, scale)
            if rest == 0:
                return f"{_int_to_words(major)} {label}"
            return f"{_int_to_words(major)} {label} {_int_to_words(rest)}"
    return str(value)


def _digits_to_words(value: str) -> str:
    parts: List[str] = []
    for ch in value:
        if ch.isdigit():
            parts.append(_WORD_ONES[int(ch)])
        else:
            parts.append(ch)
    return " ".join(parts)


def _int_to_ordinal_words(value: int) -> str:
    if value < 0:
        return f"minus {_int_to_ordinal_words(abs(value))}"
    if value == 0:
        return "zeroth"

    cardinal = _int_to_words(value)
    parts = cardinal.split()
    if not parts:
        return cardinal

    last = parts[-1]
    irregular = {
        "one": "first",
        "two": "second",
        "three": "third",
        "five": "fifth",
        "eight": "eighth",
        "nine": "ninth",
        "twelve": "twelfth",
    }
    if last in irregular:
        parts[-1] = irregular[last]
    elif last.endswith("y"):
        parts[-1] = f"{last[:-1]}ieth"
    elif last.endswith("e"):
        parts[-1] = f"{last}th"
    else:
        parts[-1] = f"{last}th"
    return " ".join(parts)


def _year_to_words(value: int) -> str:
    if value < 1000 or value > 2099:
        return _int_to_words(value)
    if value < 2000:
        century = value // 100
        suffix = value % 100
        prefix = _int_to_words(century)
        if suffix == 0:
            return f"{prefix} hundred"
        if suffix < 10:
            return f"{prefix} oh {_int_to_words(suffix)}"
        return f"{prefix} {_int_to_words(suffix)}"
    if value == 2000:
        return "two thousand"
    suffix = value - 2000
    if suffix < 10:
        return f"two thousand {_int_to_words(suffix)}"
    return f"twenty {_int_to_words(suffix)}"


def _expand_year_range_end(start: int, end_raw: str) -> Optional[int]:
    token = end_raw.strip()
    if not token.isdigit():
        return None
    if len(token) == 2:
        year = (start // 100) * 100 + int(token)
        if year < start:
            year += 100
        return year
    if len(token) == 4:
        return int(token)
    return None


def _normalize_label_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        label = match.group("label")
        num = match.group("num")
        parts = []
        for piece in num.split("."):
            try:
                value = int(piece)
            except ValueError:
                parts.append(piece)
                continue
            parts.append(_int_to_words(value))
        return f"{label} {' point '.join(parts)}"

    return _NUMBER_LABEL_RE.sub(replace, text)


def _normalize_grouped_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        stripped = token.replace(",", "").replace(".", "")
        try:
            value = int(stripped)
        except ValueError:
            return token
        return _int_to_words(value)

    text = _GROUPED_INT_COMMA_RE.sub(replace, text)
    text = _GROUPED_INT_DOT_RE.sub(replace, text)
    return text


def _normalize_decimal_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        left, right = token.split(".", 1)
        try:
            left_value = int(left)
        except ValueError:
            return token
        left_words = _int_to_words(left_value)
        right_words = _digits_to_words(right)
        return f"{left_words} point {right_words}"

    return _DECIMAL_RE.sub(replace, text)


def _normalize_plain_large_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if len(token) < 7:
            return token
        if len(token) > 1 and token.startswith("0"):
            return token
        try:
            value = int(token)
        except ValueError:
            return token
        return _int_to_words(value)

    return _PLAIN_INT_RE.sub(replace, text)


def _normalize_signed_integers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        sign = match.group("sign")
        token = match.group("num")
        try:
            value = int(token)
        except ValueError:
            return match.group(0)
        words = _int_to_words(value)
        if sign == "-":
            return f"minus {words}"
        return f"plus {words}"

    return _SIGNED_INT_RE.sub(replace, text)


def _normalize_year_ranges(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        try:
            start = int(match.group("start"))
        except ValueError:
            return match.group(0)
        end = _expand_year_range_end(start, match.group("end"))
        if end is None or not (1000 <= end <= 2099):
            return match.group(0)
        return f"{_year_to_words(start)} to {_year_to_words(end)}"

    return _YEAR_RANGE_RE.sub(replace, text)


def _normalize_plain_years(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        try:
            year = int(match.group("year"))
        except ValueError:
            return match.group(0)
        return _year_to_words(year)

    return _YEAR_RE.sub(replace, text)


def _normalize_ordinals(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group("num")
        try:
            value = int(token)
        except ValueError:
            return match.group(0)
        return _int_to_ordinal_words(value)

    return _ORDINAL_RE.sub(replace, text)


def _number_run_to_words(token: str) -> str:
    if not token:
        return token
    if len(token) > 1 and token.startswith("0"):
        return _digits_to_words(token)
    try:
        value = int(token)
    except ValueError:
        return _digits_to_words(token)
    return _int_to_words(value)


def _normalize_residual_digits(text: str) -> str:
    if not text:
        return text
    parts: List[str] = []
    cursor = 0
    text_len = len(text)
    for match in _RESIDUAL_DIGITS_RE.finditer(text):
        start, end = match.span()
        if cursor < start:
            parts.append(text[cursor:start])
        left = text[start - 1] if start > 0 else ""
        right = text[end] if end < text_len else ""
        replacement = _number_run_to_words(match.group(0))
        if left and left.isalpha():
            if not parts or not parts[-1].endswith(" "):
                parts.append(" ")
        parts.append(replacement)
        if right and right.isalpha():
            parts.append(" ")
        cursor = end
    if cursor < text_len:
        parts.append(text[cursor:])
    return "".join(parts)


def _normalize_roman_decimal_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        roman = match.group("roman")
        number = _roman_to_int(roman)
        if number is None:
            return match.group(0)
        parts = [_int_to_words(number)]
        for piece in match.group("num").split("."):
            try:
                value = int(piece)
            except ValueError:
                parts.append(piece)
                continue
            parts.append(_int_to_words(value))
        return " point ".join(parts)

    return _ROMAN_DECIMAL_RE.sub(replace, text)


def _is_singular_currency_amount(amount: str) -> bool:
    value = amount.replace(",", "").lstrip("+-")
    if not value:
        return False
    if "." not in value:
        return value == "1"
    left, right = value.split(".", 1)
    if not left:
        left = "0"
    return left == "1" and (not right or all(ch == "0" for ch in right))


def _normalize_currency_amount(amount: str) -> str:
    value = amount.replace(",", "")
    sign = ""
    if value.startswith(("+", "-")):
        sign, value = value[0], value[1:]
    if "." not in value:
        return sign + value
    left, right = value.split(".", 1)
    if right and all(ch == "0" for ch in right):
        return sign + left
    return sign + value


def _normalize_currency_symbols(text: str) -> str:
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        symbol = match.group("sym")
        amount = match.group("amount")
        unit = _CURRENCY_SYMBOL_UNITS.get(symbol)
        if not unit:
            return match.group(0)
        normalized_amount = _normalize_currency_amount(amount)
        singular, plural = unit
        noun = singular if _is_singular_currency_amount(amount) else plural
        return f"{normalized_amount} {noun}"

    text = _CURRENCY_PREFIX_RE.sub(replace, text)
    return _CURRENCY_SUFFIX_RE.sub(replace, text)


def _normalize_era_abbreviations(text: str) -> str:
    if not text:
        return text

    for pattern, replacement in _ERA_DOTTED_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    def hyphenate_era_letters(era: str) -> str:
        return "-".join(ch for ch in era if ch.isalpha())

    def replace_plain_with_year(match: re.Match[str]) -> str:
        year = match.group("year")
        era = match.group("era")
        return f"{year} {hyphenate_era_letters(era)}"

    text = _ERA_PLAIN_WITH_YEAR_RE.sub(replace_plain_with_year, text)

    def replace_plain_before_year(match: re.Match[str]) -> str:
        era = match.group("era")
        year = match.group("year")
        return f"{hyphenate_era_letters(era)} {year}"

    return _ERA_PLAIN_BEFORE_YEAR_RE.sub(replace_plain_before_year, text)


def normalize_numbers_for_tts(text: str) -> str:
    text = _normalize_roman_decimal_numbers(text)
    text = _normalize_label_numbers(text)
    text = _normalize_grouped_numbers(text)
    text = _normalize_decimal_numbers(text)
    text = _normalize_plain_large_numbers(text)
    text = _normalize_signed_integers(text)
    text = _normalize_year_ranges(text)
    text = _normalize_plain_years(text)
    text = _normalize_ordinals(text)
    text = _normalize_residual_digits(text)
    return text


def split_tts_text_for_synthesis(text: str, max_chars: int) -> List[str]:
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    spans = make_chunk_spans(text, max_chars=max_chars, chunk_mode="sentence")
    if not spans:
        return [text]
    return [text[start:end] for start, end in spans]


def _normalize_roman_numerals(text: str) -> str:
    def prev_word(start: int) -> Optional[str]:
        match = re.search(r"([A-Za-z]+)\s*$", text[:start])
        if not match:
            return None
        return match.group(1)

    def next_word(start: int) -> Optional[str]:
        match = re.search(r"\b([A-Za-z]+)", text[start:])
        if not match:
            return None
        return match.group(1)

    def next_non_space(start: int) -> str:
        match = re.search(r"\S", text[start:])
        if not match:
            return ""
        return match.group(0)

    def should_convert_roman_i(match: re.Match[str]) -> bool:
        label = match.group("label")
        if label and label[0].isupper():
            return True
        prev = prev_word(match.start())
        if prev and prev.lower() in _ROMAN_I_DETERMINERS:
            return False
        next_char = next_non_space(match.end())
        if not next_char:
            return True
        if next_char in _ROMAN_HEADING_TRAIL_PUNCT:
            return True
        next_token = next_word(match.end())
        if next_token and next_token[0].isupper():
            return True
        return False

    def replace_heading(match: re.Match[str]) -> str:
        roman = match.group("num")
        number = _roman_to_int(match.group("num"))
        if number is None:
            return match.group(0)
        if roman.upper() == "I" and not should_convert_roman_i(match):
            return match.group(0)
        return f"{match.group('label')} {_int_to_words(number)}"

    def replace_leading_title(match: re.Match[str]) -> str:
        number = _roman_to_int(match.group("num"))
        if number is None:
            return match.group(0)
        trail = match.group("trail") or ""
        return f"{match.group('indent')}{_int_to_words(number)}{trail}\n"

    text = _ROMAN_LEADING_TITLE_RE.sub(replace_leading_title, text, count=1)
    text = _ROMAN_HEADING_RE.sub(replace_heading, text)
    stripped = text.strip()
    match = _ROMAN_STANDALONE_RE.fullmatch(stripped)
    if not match:
        return text
    number = _roman_to_int(match.group("num"))
    if number is None:
        return text
    suffix = match.group("trail") or ""
    return f"{_int_to_words(number)}{suffix}"


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
    # Word boundary based on ASCII word chars; this keeps punctuation-delimited tokens replaceable.
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


def prepare_tts_text(
    text: str,
    reading_overrides: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    text = _strip_double_quotes(text)
    text = _strip_single_quotes(text)
    text = apply_reading_overrides(text, reading_overrides or [])
    text = _transliterate_pali_sanskrit(text)
    # Apply twice so users can match either original spellings (with diacritics)
    # or transliterated forms used by Pocket-TTS.
    text = apply_reading_overrides(text, reading_overrides or [])
    text = normalize_abbreviations(text)
    text = _normalize_era_abbreviations(text)
    text = _normalize_roman_numerals(text)
    text = _normalize_currency_symbols(text)
    text = normalize_numbers_for_tts(text)
    text = _normalize_linebreak_pauses(text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def load_text_chapters(text_path: Path) -> List[ChapterInput]:
    text = read_clean_text(text_path)
    title = text_path.stem or "text"
    chapter_id = chapter_id_from_path(1, title, None)
    return [
        ChapterInput(index=1, id=chapter_id, title=title, text=text, path=str(text_path))
    ]


def load_book_chapters(book_dir: Path) -> List[ChapterInput]:
    toc_path = book_dir / "clean" / "toc.json"
    if not toc_path.exists():
        raise FileNotFoundError(f"Missing clean/toc.json at {toc_path}")

    toc = json.loads(toc_path.read_text(encoding="utf-8"))
    entries = toc.get("chapters", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("clean/toc.json contains no chapters.")

    chapters: List[ChapterInput] = []
    for fallback_idx, entry in enumerate(entries, start=1):
        rel = entry.get("path")
        if not rel:
            continue
        path = book_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing chapter file: {path}")

        text = read_clean_text(path)
        if not text.strip():
            continue

        index = int(entry.get("index") or fallback_idx)
        title = str(entry.get("title") or f"Chapter {index}")
        chapter_id = chapter_id_from_path(index, title, rel)

        chapters.append(
            ChapterInput(
                index=index,
                id=chapter_id,
                title=title,
                text=text,
                path=rel,
            )
        )

    if not chapters:
        raise ValueError("No chapter text found in clean/chapters.")

    return chapters


def write_combined_input(chapters: Sequence[ChapterInput], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(ch.text.strip() for ch in chapters if ch.text.strip()).strip()
    path = out_dir / "input.txt"
    path.write_text(text + "\n", encoding="utf-8")
    return path


# ----------------------------
# Manifest + outputs
# ----------------------------

def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_status(out_dir: Path, stage: str, detail: Optional[str] = None) -> None:
    payload = {"stage": stage, "updated_unix": int(time.time())}
    if detail:
        payload["detail"] = detail
    atomic_write_json(out_dir / "status.json", payload)


def write_chunk_files(
    chunks: Sequence[str], chunk_dir: Path, overwrite: bool = False
) -> List[Path]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in chunk_dir.glob("*.txt"):
            path.unlink()

    paths: List[Path] = []
    for idx, chunk in enumerate(chunks, start=1):
        path = chunk_dir / f"{idx:06d}.txt"
        if overwrite or not path.exists():
            path.write_text(chunk.rstrip() + "\n", encoding="utf-8")
        paths.append(path)

    if overwrite:
        for path in chunk_dir.glob("*.txt"):
            stem = path.stem
            if stem.isdigit() and int(stem) > len(chunks):
                path.unlink()

    return paths


# ----------------------------
# WAV IO utilities
# ----------------------------

def _require_tts() -> None:
    if torch is None or TTSModel is None:
        raise RuntimeError(
            "Pocket-TTS dependencies are missing. Install torch and pocket-tts "
            "or run with uv: `uv run --with pocket-tts`."
        )


def tensor_to_int16(audio: "torch.Tensor") -> "torch.Tensor":
    """
    Pocket-TTS README says returned audio is PCM data in a 1D torch tensor.
    Make this robust to float or int tensors.
    """
    _require_tts()
    a = audio.detach().cpu().flatten().contiguous()

    if a.dtype in (torch.float16, torch.float32, torch.float64):
        # Heuristic: if values look like [-1, 1], scale to int16.
        max_abs = float(a.abs().max().item()) if a.numel() else 0.0
        if max_abs <= 1.5:
            a = torch.clamp(a, -1.0, 1.0)
            a = torch.round(a * 32767.0).to(torch.int16)
        else:
            a = torch.round(a).to(torch.int16)
    elif a.dtype != torch.int16:
        a = a.to(torch.int16)

    return a


def write_wav_mono_16k_or_24k(
    path: Path, samples_i16: "torch.Tensor", sample_rate: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    arr = samples_i16.numpy()  # requires numpy via torch; typical torch installs include it
    data = arr.tobytes()

    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(data)

    tmp.replace(path)


def wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    if rate <= 0:
        return 0
    return int(round(frames * 1000.0 / rate))


def is_valid_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getnframes() > 0
    except Exception:
        return False


def build_concat_file(segment_paths: List[Path], concat_path: Path, base_dir: Path) -> None:
    lines = []
    for p in segment_paths:
        rel = p.relative_to(base_dir).as_posix()
        # ffmpeg concat demuxer format
        lines.append(f"file '{rel}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_chapters_ffmeta(
    chapters: Sequence[Tuple[str, int]], ffmeta_path: Path
) -> None:
    """
    Generates a simple FFMETADATA1 file with one chapter per chapter.
    ffmpeg can import chapters using -map_chapters.
    """
    out = [";FFMETADATA1"]
    t = 0
    for title, d in chapters:
        start = t
        end = t + max(int(d), 1)
        out.append("")
        out.append("[CHAPTER]")
        out.append("TIMEBASE=1/1000")
        out.append(f"START={start}")
        out.append(f"END={end}")
        out.append(f"title={title}")
        t = end
    ffmeta_path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ----------------------------
# Synthesis
# ----------------------------

def prune_chapter_dirs(root: Path, keep: set[str]) -> None:
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child)


def prepare_manifest(
    chapters: Sequence[ChapterInput],
    out_dir: Path,
    voice: str,
    max_chars: int,
    pad_ms: int,
    chunk_mode: str,
    rechunk: bool,
) -> Tuple[Dict[str, Any], List[List[str]], int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    chunk_root = out_dir / "chunks"

    chapter_ids = {c.id for c in chapters}
    if rechunk:
        prune_chapter_dirs(chunk_root, chapter_ids)

    if manifest_path.exists() and not rechunk:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_chapters = manifest.get("chapters", [])
        if not isinstance(manifest_chapters, list) or not manifest_chapters:
            raise ValueError("manifest.json contains no chapters.")
        existing_mode = manifest.get("chunk_mode", "sentence")
        if existing_mode == "packed" and chunk_mode == "sentence":
            existing_mode = "sentence"
        if existing_mode != chunk_mode:
            raise ValueError(
                "manifest.json chunk_mode differs from requested. "
                "Run with --rechunk to regenerate manifest."
            )
        if len(manifest_chapters) != len(chapters):
            raise ValueError(
                "manifest.json chapters differ from current input. "
                "Run with --rechunk to regenerate manifest."
            )

        chapter_chunks: List[List[str]] = []
        for ch_input, ch_manifest in zip(chapters, manifest_chapters):
            if ch_manifest.get("id") != ch_input.id:
                raise ValueError(
                    "manifest.json chapter order or ids differ. "
                    "Run with --rechunk to regenerate manifest."
                )
            ch_manifest["index"] = ch_input.index
            ch_manifest["title"] = ch_input.title
            ch_manifest["path"] = ch_input.path
            text_hash = sha256_str(ch_input.text)
            if ch_manifest.get("text_sha256") != text_hash:
                raise ValueError(
                    "manifest.json exists but chapter text hash differs. "
                    "Run with --rechunk to regenerate manifest."
                )
            chunks = ch_manifest.get("chunks", [])
            if not chunks:
                raise ValueError("manifest.json contains no chunks.")
            chunk_spans = ch_manifest.get("chunk_spans", [])
            if not isinstance(chunk_spans, list) or len(chunk_spans) != len(chunks):
                raise ValueError(
                    "manifest.json missing chunk spans. "
                    "Run with --rechunk to regenerate manifest."
                )
            span_pairs = _coerce_span_pairs(chunk_spans)
            if len(span_pairs) != len(chunks):
                raise ValueError(
                    "manifest.json contains invalid chunk spans. "
                    "Run with --rechunk to regenerate manifest."
                )
            expected_pause = compute_chunk_pause_multipliers(
                ch_input.text, span_pairs
            )
            ch_manifest["pause_multipliers"] = _normalize_pause_multipliers(
                ch_manifest.get("pause_multipliers"),
                len(chunks),
                fallback=expected_pause,
            )
            chapter_chunks.append(chunks)
        _apply_chapter_boundary_pause_multipliers(manifest_chapters)
        pad_ms = int(manifest.get("pad_ms", pad_ms))
    else:
        chapter_chunks = []
        manifest_chapters = []
        for ch in chapters:
            spans = make_chunk_spans(
                ch.text, max_chars=max_chars, chunk_mode=chunk_mode
            )
            chunks = [ch.text[start:end] for start, end in spans]
            span_list = [[start, end] for start, end in spans]
            pause_multipliers = compute_chunk_pause_multipliers(ch.text, spans)
            if not chunks:
                raise ValueError(f"No chunks generated for chapter: {ch.id}")
            chapter_chunks.append(chunks)
            manifest_chapters.append(
                {
                    "index": ch.index,
                    "id": ch.id,
                    "title": ch.title,
                    "path": ch.path,
                    "text_sha256": sha256_str(ch.text),
                    "chunks": chunks,
                    "chunk_spans": span_list,
                    "pause_multipliers": pause_multipliers,
                    "durations_ms": [None] * len(chunks),
                }
            )
        _apply_chapter_boundary_pause_multipliers(manifest_chapters)

        manifest = {
            "created_unix": int(time.time()),
            "voice": voice,
            "max_chars": int(max_chars),
            "pad_ms": int(pad_ms),
            "chunk_mode": chunk_mode,
            "chapters": manifest_chapters,
        }
        atomic_write_json(manifest_path, manifest)

    manifest["voice"] = voice
    manifest["max_chars"] = int(max_chars)
    manifest["pad_ms"] = int(manifest.get("pad_ms", pad_ms))
    manifest["chunk_mode"] = chunk_mode

    for ch_entry, chunks in zip(manifest["chapters"], chapter_chunks):
        if "durations_ms" not in ch_entry or len(ch_entry["durations_ms"]) != len(chunks):
            ch_entry["durations_ms"] = [None] * len(chunks)

    for ch_entry, chunks in zip(manifest["chapters"], chapter_chunks):
        chunk_dir = chunk_root / ch_entry["id"]
        write_chunk_files(chunks, chunk_dir, overwrite=rechunk)

    atomic_write_json(manifest_path, manifest)

    return manifest, chapter_chunks, int(manifest["pad_ms"])


def _normalize_voice_id(value: Optional[str], default_voice: str) -> str:
    if value is None:
        return default_voice
    cleaned = str(value).strip()
    if not cleaned:
        return default_voice
    if cleaned.lower() == "default":
        return default_voice
    return cleaned


def _load_voice_map(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Voice map not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Voice map must be a JSON object: {path}")
    chapters = data.get("chapters", {})
    if not isinstance(chapters, dict):
        chapters = {}
    return {
        "default": data.get("default"),
        "chapters": chapters,
    }


def chunk_book(
    book_dir: Path,
    out_dir: Optional[Path] = None,
    voice: Optional[str] = None,
    max_chars: int = 400,
    pad_ms: int = 300,
    chunk_mode: str = "sentence",
    rechunk: bool = True,
) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = book_dir / "tts"
    if voice is None:
        voice = DEFAULT_VOICE
    else:
        voice = voice.strip()
        if not voice or voice.lower() == "default":
            voice = DEFAULT_VOICE

    chapters = load_book_chapters(book_dir)
    manifest, _chapter_chunks, _pad_ms = prepare_manifest(
        chapters=chapters,
        out_dir=out_dir,
        voice=voice,
        max_chars=max_chars,
        pad_ms=pad_ms,
        chunk_mode=chunk_mode,
        rechunk=rechunk,
    )
    return manifest


def synthesize(
    chapters: Sequence[ChapterInput],
    voice: Optional[str],
    out_dir: Path,
    max_chars: int = 400,
    pad_ms: int = 300,
    chunk_mode: str = "sentence",
    rechunk: bool = False,
    wipe_segments: Optional[bool] = None,
    only_chapter_ids: Optional[set[str]] = None,
    voice_map_path: Optional[Path] = None,
    reading_overrides_dir: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> int:
    _require_tts()

    if base_dir is None:
        base_dir = Path.cwd()
    if wipe_segments is None:
        wipe_segments = rechunk
    if voice is None:
        voice = DEFAULT_VOICE
    else:
        voice = voice.strip()
        if not voice or voice.lower() == "default":
            voice = DEFAULT_VOICE

    try:
        voice_map = _load_voice_map(voice_map_path)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    try:
        global_reading_overrides: List[Dict[str, Any]] = []
        chapter_reading_overrides: Dict[str, List[Dict[str, Any]]] = {}
        if reading_overrides_dir is not None:
            global_reading_overrides, chapter_reading_overrides = _load_reading_overrides(
                reading_overrides_dir
            )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    default_voice = voice
    if voice_map:
        default_voice = _normalize_voice_id(voice_map.get("default"), default_voice)

    out_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = out_dir / "segments"
    manifest_path = out_dir / "manifest.json"
    concat_path = out_dir / "concat.txt"
    chapters_path = out_dir / "chapters.ffmeta"

    try:
        manifest, chapter_chunks, pad_ms = prepare_manifest(
            chapters=chapters,
            out_dir=out_dir,
            voice=default_voice,
            max_chars=max_chars,
            pad_ms=pad_ms,
            chunk_mode=chunk_mode,
            rechunk=rechunk,
        )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    if wipe_segments and seg_dir.exists():
        shutil.rmtree(seg_dir)

    chapter_voice_map: Dict[str, str] = {}
    voice_overrides: Dict[str, str] = {}
    if voice_map:
        raw_overrides = voice_map.get("chapters", {})
        for entry in manifest.get("chapters", []):
            chapter_id = entry.get("id") or "chapter"
            raw_value = raw_overrides.get(chapter_id) if isinstance(raw_overrides, dict) else None
            selected = _normalize_voice_id(raw_value, default_voice)
            chapter_voice_map[chapter_id] = selected
            entry["voice"] = selected
            if selected != default_voice:
                voice_overrides[chapter_id] = selected
        manifest["voice_overrides"] = voice_overrides
        manifest["voice"] = default_voice
        atomic_write_json(manifest_path, manifest)
    else:
        for entry in manifest.get("chapters", []):
            chapter_id = entry.get("id") or "chapter"
            chapter_voice_map[chapter_id] = default_voice

    chapter_reading_map: Dict[str, List[Dict[str, Any]]] = {}
    for entry in manifest.get("chapters", []):
        chapter_id = entry.get("id") or "chapter"
        chapter_entries = chapter_reading_overrides.get(chapter_id, [])
        chapter_reading_map[chapter_id] = _merge_reading_overrides(
            global_reading_overrides, chapter_entries
        )

    voice_prompts: Dict[str, str] = {}
    try:
        for voice_id in sorted(set(chapter_voice_map.values())):
            voice_prompts[voice_id] = resolve_voice_prompt(
                voice_id, base_dir=base_dir
            )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    write_status(out_dir, "cloning", "Preparing voice")

    tts_model = TTSModel.load_model()
    _install_tts_warning_filter()
    sample_rate = int(tts_model.sample_rate)
    if manifest.get("sample_rate") != sample_rate:
        manifest["sample_rate"] = sample_rate
        atomic_write_json(manifest_path, manifest)

    voice_states: Dict[str, Any] = {}
    for voice_id, voice_prompt in voice_prompts.items():
        voice_states[voice_id] = tts_model.get_state_for_audio_prompt(voice_prompt)
    write_status(out_dir, "synthesizing")

    base_pad_samples = int(round(sample_rate * (pad_ms / 1000.0)))
    pad_tensors: Dict[int, Optional["torch.Tensor"]] = {}

    def pad_tensor_for(multiplier: int) -> Optional["torch.Tensor"]:
        multiplier = max(1, int(multiplier))
        if base_pad_samples <= 0:
            return None
        if multiplier not in pad_tensors:
            total_samples = base_pad_samples * multiplier
            pad_tensors[multiplier] = (
                torch.zeros(total_samples, dtype=torch.int16)
                if total_samples > 0
                else None
            )
        return pad_tensors[multiplier]

    segment_paths: List[Path] = []
    selected_ids = set(only_chapter_ids) if only_chapter_ids else None
    selected_indices = [
        idx
        for idx, entry in enumerate(manifest["chapters"])
        if not selected_ids or (entry.get("id") or "chapter") in selected_ids
    ]
    if selected_ids and not selected_indices:
        sys.stderr.write("No matching chapters found for synthesis.\n")
        return 2
    total_chunks = sum(len(chapter_chunks[idx]) for idx in selected_indices)
    if total_chunks <= 0:
        sys.stderr.write("No chunks selected for synthesis.\n")
        return 2

    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        overall_task = progress.add_task("Total", total=total_chunks)
        chapter_task = progress.add_task("Chapter", total=0)

        for ch_entry, chunks in zip(manifest["chapters"], chapter_chunks):
            chapter_id = ch_entry.get("id") or "chapter"
            chapter_title = ch_entry.get("title") or chapter_id
            chapter_total = len(chunks)
            if selected_ids and chapter_id not in selected_ids:
                continue

            progress.update(
                chapter_task,
                total=chapter_total,
                completed=0,
                description=f"{chapter_id}: {chapter_title}",
            )

            chapter_seg_dir = seg_dir / chapter_id

            for chunk_idx, chunk_text in enumerate(chunks, start=1):
                seg_path = chapter_seg_dir / f"{chunk_idx:06d}.wav"
                progress.update(
                    chapter_task,
                    description=f"{chapter_id}: {chapter_title} ({chunk_idx}/{chapter_total})",
                )

                if seg_path.exists() and is_valid_wav(seg_path):
                    segment_paths.append(seg_path)
                    dms = wav_duration_ms(seg_path)
                    if ch_entry["durations_ms"][chunk_idx - 1] != dms:
                        ch_entry["durations_ms"][chunk_idx - 1] = dms
                        atomic_write_json(manifest_path, manifest)
                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue

                tts_text = prepare_tts_text(
                    chunk_text, chapter_reading_map.get(chapter_id, [])
                )
                if not tts_text.strip():
                    sys.stderr.write(
                        f"Skipping empty chunk {chapter_id} ({chunk_idx}/{chapter_total}).\n"
                    )
                    ch_entry["durations_ms"][chunk_idx - 1] = 0
                    atomic_write_json(manifest_path, manifest)
                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue
                voice_id = chapter_voice_map.get(chapter_id, default_voice)
                voice_state = voice_states[voice_id]
                pause_multiplier = 1
                raw_pause = ch_entry.get("pause_multipliers")
                if (
                    isinstance(raw_pause, list)
                    and len(raw_pause) == chapter_total
                ):
                    try:
                        pause_multiplier = max(1, int(raw_pause[chunk_idx - 1]))
                    except (TypeError, ValueError):
                        pause_multiplier = 1

                sub_texts = split_tts_text_for_synthesis(tts_text, max_chars=max_chars)
                sub_total = len(sub_texts)
                audio_parts: List["torch.Tensor"] = []
                for sub_idx, sub_text in enumerate(sub_texts, start=1):
                    sub_text = sub_text.strip()
                    if not sub_text:
                        continue
                    if not _ends_with_sentence_punct(sub_text):
                        sub_text = f"{sub_text}."
                    with _tts_warning_context(
                        chapter_id, chunk_idx, chapter_total, sub_idx, sub_total
                    ):
                        audio_parts.append(tts_model.generate_audio(voice_state, sub_text))

                if not audio_parts:
                    with _tts_warning_context(chapter_id, chunk_idx, chapter_total, 1, 1):
                        audio_parts = [tts_model.generate_audio(voice_state, tts_text)]

                if len(audio_parts) == 1:
                    audio = audio_parts[0]
                else:
                    audio = torch.cat([part.flatten() for part in audio_parts], dim=0)
                a16 = tensor_to_int16(audio)

                pad_tensor = pad_tensor_for(pause_multiplier)
                if pad_tensor is not None and pad_tensor.numel() > 0:
                    a16 = torch.cat([a16, pad_tensor], dim=0)

                write_wav_mono_16k_or_24k(seg_path, a16, sample_rate=sample_rate)
                segment_paths.append(seg_path)
                dms = int(round(a16.numel() * 1000.0 / sample_rate))

                # Persist progress for restartability.
                ch_entry["durations_ms"][chunk_idx - 1] = dms
                atomic_write_json(manifest_path, manifest)

                progress.advance(chapter_task, 1)
                progress.advance(overall_task, 1)

    build_concat_file(segment_paths, concat_path, base_dir=out_dir)

    chapter_meta: List[Tuple[str, int]] = []
    for ch_entry in manifest["chapters"]:
        title = ch_entry.get("title") or ch_entry.get("id") or "Chapter"
        durations = ch_entry.get("durations_ms", [])
        total_ms = sum(int(d or 0) for d in durations)
        chapter_meta.append((title, total_ms))

    build_chapters_ffmeta(chapter_meta, chapters_path)

    write_status(out_dir, "done")
    return 0


def synthesize_chunk(
    out_dir: Path,
    chapter_id: str,
    chunk_index: int,
    voice: Optional[str] = None,
    voice_map_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> dict:
    _require_tts()
    if base_dir is None:
        base_dir = Path.cwd()
    if chunk_index < 0:
        raise ValueError("chunk_index must be >= 0")

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chapters = manifest.get("chapters", [])
    if not isinstance(chapters, list):
        raise ValueError("manifest.json chapters missing or invalid")

    entry = None
    entry_index = -1
    for idx, item in enumerate(chapters):
        if isinstance(item, dict) and item.get("id") == chapter_id:
            entry = item
            entry_index = idx
            break
    if entry is None:
        raise ValueError(f"Unknown chapter_id: {chapter_id}")

    chunks = entry.get("chunks")
    if not isinstance(chunks, list):
        chunks = []
    spans = entry.get("chunk_spans")
    if not isinstance(spans, list):
        spans = []
    chunk_count = len(chunks) or len(spans)
    if chunk_count <= 0:
        chunk_dir = out_dir / "chunks" / chapter_id
        if chunk_dir.exists():
            chunk_count = len([p for p in chunk_dir.glob("*.txt") if p.stem.isdigit()])
    if chunk_count <= 0:
        raise ValueError(f"No chunks available for chapter: {chapter_id}")
    if chunk_index >= chunk_count:
        raise ValueError(f"chunk_index out of range for {chapter_id}")

    chunk_text: Optional[str] = None
    if chunks and chunk_index < len(chunks):
        chunk_text = str(chunks[chunk_index])
    if chunk_text is None:
        chunk_path = out_dir / "chunks" / chapter_id / f"{chunk_index + 1:06d}.txt"
        if chunk_path.exists():
            chunk_text = chunk_path.read_text(encoding="utf-8").rstrip("\n")
    if chunk_text is None:
        raise ValueError(f"Chunk text missing for {chapter_id} #{chunk_index + 1}")

    durations = entry.get("durations_ms")
    if not isinstance(durations, list) or len(durations) != chunk_count:
        durations = [None] * chunk_count
        entry["durations_ms"] = durations
    span_pairs = _coerce_span_pairs(spans)
    pause_multipliers = entry.get("pause_multipliers")
    if not isinstance(pause_multipliers, list) or len(pause_multipliers) != chunk_count:
        computed_pause = [1] * chunk_count
        if span_pairs and len(span_pairs) == chunk_count:
            chapter_text = ""
            rel_path = entry.get("path")
            if isinstance(rel_path, str) and rel_path:
                clean_path = (out_dir.parent / rel_path).resolve()
                if clean_path.exists():
                    try:
                        chapter_text = read_clean_text(clean_path)
                    except OSError:
                        chapter_text = ""
            if chapter_text:
                computed_pause = compute_chunk_pause_multipliers(
                    chapter_text, span_pairs
                )
        pause_multipliers = computed_pause
        entry["pause_multipliers"] = pause_multipliers
    if entry_index >= 0 and entry_index < len(chapters) - 1 and pause_multipliers:
        normalized_pause = _normalize_pause_multipliers(
            pause_multipliers, chunk_count
        )
        normalized_pause[-1] = max(
            normalized_pause[-1], _CHAPTER_BREAK_PAD_MULTIPLIER
        )
        pause_multipliers = normalized_pause
        entry["pause_multipliers"] = pause_multipliers

    if voice is None:
        default_voice = manifest.get("voice") or DEFAULT_VOICE
    else:
        voice = voice.strip()
        if not voice or voice.lower() == "default":
            voice = DEFAULT_VOICE
        default_voice = voice
    default_voice = _normalize_voice_id(default_voice, DEFAULT_VOICE)

    voice_id = default_voice
    if voice_map_path:
        voice_map = _load_voice_map(voice_map_path)
        if voice_map:
            default_voice = _normalize_voice_id(voice_map.get("default"), default_voice)
            raw_chapter_voice = (
                voice_map.get("chapters", {}).get(chapter_id)
                if isinstance(voice_map.get("chapters"), dict)
                else None
            )
            voice_id = _normalize_voice_id(raw_chapter_voice, default_voice)
        else:
            voice_id = default_voice
    else:
        voice_id = _normalize_voice_id(entry.get("voice"), default_voice)

    voice_prompt = resolve_voice_prompt(voice_id, base_dir=base_dir)
    tts_model = TTSModel.load_model()
    _install_tts_warning_filter()
    sample_rate = int(tts_model.sample_rate)
    if manifest.get("sample_rate") != sample_rate:
        manifest["sample_rate"] = sample_rate

    max_chars = int(manifest.get("max_chars") or 400)
    pad_ms = int(manifest.get("pad_ms") or 300)
    base_pad_samples = int(round(sample_rate * (pad_ms / 1000.0)))
    try:
        pause_multiplier = max(1, int(pause_multipliers[chunk_index]))
    except (TypeError, ValueError, IndexError):
        pause_multiplier = 1
    pad_samples = base_pad_samples * pause_multiplier
    pad_tensor = torch.zeros(pad_samples, dtype=torch.int16) if pad_samples > 0 else None

    overrides_dir = out_dir
    if not _reading_overrides_path(overrides_dir).exists():
        overrides_dir = out_dir.parent
    global_reading_overrides, chapter_reading_overrides = _load_reading_overrides(
        overrides_dir
    )
    merged_reading_overrides = _merge_reading_overrides(
        global_reading_overrides,
        chapter_reading_overrides.get(chapter_id, []),
    )
    tts_text = prepare_tts_text(chunk_text, merged_reading_overrides)
    seg_path = out_dir / "segments" / chapter_id / f"{chunk_index + 1:06d}.wav"
    seg_path.parent.mkdir(parents=True, exist_ok=True)

    if not tts_text.strip():
        if seg_path.exists():
            seg_path.unlink()
        durations[chunk_index] = 0
        atomic_write_json(manifest_path, manifest)
        return {
            "status": "skipped",
            "chapter_id": chapter_id,
            "chunk_index": chunk_index,
            "duration_ms": 0,
        }

    voice_state = tts_model.get_state_for_audio_prompt(voice_prompt)
    sub_texts = split_tts_text_for_synthesis(tts_text, max_chars=max_chars)
    sub_total = len(sub_texts)
    audio_parts: List["torch.Tensor"] = []
    for sub_idx, sub_text in enumerate(sub_texts, start=1):
        with _tts_warning_context(
            chapter_id, chunk_index + 1, chunk_count, sub_idx, sub_total
        ):
            audio_parts.append(tts_model.generate_audio(voice_state, sub_text))
    if not audio_parts:
        with _tts_warning_context(chapter_id, chunk_index + 1, chunk_count, 1, 1):
            audio_parts = [tts_model.generate_audio(voice_state, tts_text)]
    if len(audio_parts) == 1:
        audio = audio_parts[0]
    else:
        audio = torch.cat([part.flatten() for part in audio_parts], dim=0)
    a16 = tensor_to_int16(audio)
    if pad_tensor is not None and pad_tensor.numel() > 0:
        a16 = torch.cat([a16, pad_tensor], dim=0)
    write_wav_mono_16k_or_24k(seg_path, a16, sample_rate=sample_rate)

    dms = wav_duration_ms(seg_path)
    durations[chunk_index] = dms
    atomic_write_json(manifest_path, manifest)
    return {
        "status": "ok",
        "chapter_id": chapter_id,
        "chunk_index": chunk_index,
        "duration_ms": dms,
    }


def synthesize_text(
    text_path: Path,
    voice: Optional[str],
    out_dir: Path,
    max_chars: int = 400,
    pad_ms: int = 300,
    chunk_mode: str = "sentence",
    rechunk: bool = False,
    voice_map_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> int:
    chapters = load_text_chapters(text_path)
    return synthesize(
        chapters=chapters,
        voice=voice,
        out_dir=out_dir,
        max_chars=max_chars,
        pad_ms=pad_ms,
        chunk_mode=chunk_mode,
        rechunk=rechunk,
        voice_map_path=voice_map_path,
        reading_overrides_dir=None,
        base_dir=base_dir,
    )


def synthesize_book(
    book_dir: Path,
    voice: Optional[str],
    out_dir: Optional[Path] = None,
    max_chars: int = 400,
    pad_ms: int = 300,
    chunk_mode: str = "sentence",
    rechunk: bool = False,
    voice_map_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> int:
    if out_dir is None:
        out_dir = book_dir / "tts"
    try:
        chapters = load_book_chapters(book_dir)
        write_combined_input(chapters, out_dir)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return synthesize(
        chapters=chapters,
        voice=voice,
        out_dir=out_dir,
        max_chars=max_chars,
        pad_ms=pad_ms,
        chunk_mode=chunk_mode,
        rechunk=rechunk,
        voice_map_path=voice_map_path,
        reading_overrides_dir=book_dir,
        base_dir=base_dir,
    )


def synthesize_book_sample(
    book_dir: Path,
    voice: Optional[str],
    out_dir: Optional[Path] = None,
    max_chars: int = 400,
    pad_ms: int = 300,
    chunk_mode: str = "sentence",
    rechunk: bool = False,
    voice_map_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> int:
    if out_dir is None:
        out_dir = book_dir / "tts"
    try:
        chapters = load_book_chapters(book_dir)
        write_combined_input(chapters, out_dir)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    if not chapters:
        sys.stderr.write("No chapters found for sampling.\n")
        return 2

    sample_id = chapters[0].id
    sample_dir = out_dir / "segments" / sample_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        chapters_meta = manifest.get("chapters")
        if isinstance(chapters_meta, list):
            for entry in chapters_meta:
                if entry.get("id") == sample_id:
                    chunks = entry.get("chunks")
                    if isinstance(chunks, list):
                        entry["durations_ms"] = [None] * len(chunks)
                    break
            atomic_write_json(manifest_path, manifest)

    return synthesize(
        chapters=chapters,
        voice=voice,
        out_dir=out_dir,
        max_chars=max_chars,
        pad_ms=pad_ms,
        chunk_mode=chunk_mode,
        rechunk=rechunk,
        wipe_segments=False,
        only_chapter_ids={sample_id},
        voice_map_path=voice_map_path,
        reading_overrides_dir=book_dir,
        base_dir=base_dir,
    )


# ----------------------------
# CLI
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=Path, help="Input UTF-8 .txt file")
    group.add_argument(
        "--book", type=Path, help="Book directory containing clean/toc.json"
    )
    ap.add_argument(
        "--voice",
        help="Voice prompt: built-in name, wav path, or hf:// URL",
    )
    ap.add_argument(
        "--voice-map",
        type=Path,
        help="Path to voice map JSON for per-chapter voices",
    )
    ap.add_argument(
        "--out",
        type=Path,
        help="Output directory (default: <book>/tts when using --book)",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=400,
        help="Max characters per chunk (default: 400)",
    )
    ap.add_argument(
        "--pad-ms",
        type=int,
        default=300,
        help="Silence to append to each chunk in ms (default: 300)",
    )
    ap.add_argument(
        "--chunk-mode",
        choices=["sentence"],
        default="sentence",
        help="Chunking strategy (default: sentence)",
    )
    ap.add_argument(
        "--rechunk",
        action="store_true",
        help="Ignore existing manifest and rechunk the input text",
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.book:
        return synthesize_book(
            book_dir=args.book,
            voice=args.voice,
            out_dir=args.out,
            max_chars=args.max_chars,
            pad_ms=args.pad_ms,
            chunk_mode=args.chunk_mode,
            rechunk=args.rechunk,
            voice_map_path=args.voice_map,
        )
    if not args.out:
        parser.error("--out is required when using --text")
    return synthesize_text(
        text_path=args.text,
        voice=args.voice,
        out_dir=args.out,
        max_chars=args.max_chars,
        pad_ms=args.pad_ms,
        chunk_mode=args.chunk_mode,
        rechunk=args.rechunk,
        voice_map_path=args.voice_map,
    )


if __name__ == "__main__":
    raise SystemExit(main())
