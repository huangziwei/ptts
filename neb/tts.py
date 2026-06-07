from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
import wave

import numpy as np
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

from . import language as language_util
from .text import apply_reading_overrides, prepare_tts_text, read_clean_text
from .text.common import (
    _CHAPTER_BREAK_PAD_MULTIPLIER,
    _CLAUSE_PUNCT,
    _CLOSING_PUNCT,
    _SECTION_BREAK_NEWLINES,
    _SECTION_BREAK_PAD_MULTIPLIER,
    _SENT_PUNCT,
    _TITLE_BREAK_NEWLINES,
    _TITLE_BREAK_PAD_MULTIPLIER,
    READING_OVERRIDES_FILENAME,
    _load_reading_overrides,
    _merge_reading_overrides,
    _reading_overrides_path,
)
from .voice import DEFAULT_VOICE, resolve_voice_prompt

try:
    import torch
    from pocket_tts import TTSModel
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    TTSModel = None

try:
    import maneko
except Exception:  # pragma: no cover - optional runtime dependency
    maneko = None


_TTS_WARNING_CONTEXT = contextvars.ContextVar("_TTS_WARNING_CONTEXT", default=None)
_TTS_WARNING_FILTER_INSTALLED = False
_TTS_WARNING_CONTEXT_LOCK = threading.Lock()
_TTS_WARNING_CONTEXT_STACK: List[dict[str, Any]] = []

_TTS_MODEL_CACHE: Dict[Tuple[str, int], Any] = {}
_TTS_MODEL_CACHE_LOCK = threading.Lock()


def _repo_root() -> Path:
    """Walk up from the cwd to the directory holding pyproject.toml (the repo)."""
    start = Path.cwd()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


def _ensure_hf_home() -> None:
    """Default HF_HOME to the repo-local cache so weights land in ./.cache/.

    maneko (and pocket-tts) resolve weights via HF_HOME; pointing it at the
    repo's .cache/huggingface keeps everything self-contained. An explicit
    HF_HOME in the environment always wins.
    """
    if os.environ.get("HF_HOME"):
        return
    os.environ["HF_HOME"] = str(_repo_root() / ".cache" / "huggingface")


def _load_tts_model(
    language: Optional[str],
    layers: Optional[int] = None,
) -> Any:
    """Return a cached pocket-tts model for (language, layers)."""
    _require_tts()
    lang = language_util.resolve_language(language)
    lay = language_util.resolve_layers(lang, layers)
    key = (lang, lay)
    with _TTS_MODEL_CACHE_LOCK:
        model = _TTS_MODEL_CACHE.get(key)
        if model is None:
            model_id = language_util.resolve_model_id(lang, lay)
            model = TTSModel.load_model(language=model_id)
            _TTS_MODEL_CACHE[key] = model
        return model


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
_ABBREV_SENT_RE = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Fig|Figs)\.$", re.IGNORECASE)
_SINGLE_INITIAL_RE = re.compile(r"\b[A-Z]\.$")
_NAME_INITIAL_RE = re.compile(r"\b([A-Z][a-z]+)\s+[A-Z]\.$")
_MULTI_INITIAL_RE = re.compile(r"(?:\b[A-Z]\.\s*){2,}$")
_REFERENCE_ABBREV_RE = re.compile(r"\b(?:vol|no|nos|p|pp|v|vv)\.$", re.IGNORECASE)
_REFERENCE_ABBREV_FOLLOW_RE = re.compile(
    r"""^[\"'(\[]*(?:\d|[IVXLCDM]+\b|[A-Za-z](?:\d+|[.-]\d+)?\b)""",
    re.IGNORECASE,
)
_ABBREV_WHITELIST = {
    "a.a.",
    "a.e.",
    "a.k.a.",
    "a.m.",
    "approx.",
    "b.a.",
    "b.f.",
    "b.f.a.",
    "b.s.",
    "b.s.n.",
    "c.e.o.",
    "c.f.o.",
    "c.g.",
    "c.i.a.",
    "c.o.o.",
    "c.p.a.",
    "c.s.",
    "ca.",
    "cf.",
    "d.c.",
    "d.d.s.",
    "d.h.",
    "d.o.",
    "d.v.m.",
    "e.e.",
    "e.g.",
    "e.m.",
    "et al.",
    "et seq.",
    "et seqq.",
    "etc.",
    "f.b.i.",
    "g.k.",
    "h.g.",
    "h.p.",
    "i.e.",
    "ibid.",
    "j.d.",
    "j.b.",
    "j.f.",
    "j.g.",
    "j.k.",
    "j.m.",
    "j.r.r.",
    "l.p.n.",
    "l.m.",
    "loc. cit.",
    "m.a.",
    "m.b.a.",
    "m.d.",
    "m.f.a.",
    "m.p.h.",
    "m.r.",
    "m.s.",
    "m.s.w.",
    "n.b.",
    "op. cit.",
    "p.e.",
    "p.g.",
    "p.m.",
    "ph.d.",
    "q.v.",
    "r.n.",
    "r.l.",
    "t.e.",
    "t.s.",
    "u.k.",
    "u.n.",
    "u.s.",
    "u.s.a.",
    "v.s.",
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

    if _REFERENCE_ABBREV_RE.search(tail):
        if _REFERENCE_ABBREV_FOLLOW_RE.match(paragraph[next_pos:]):
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

    return bool(_SINGLE_INITIAL_RE.search(tail))


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


def split_tts_text_for_synthesis(text: str, max_chars: int) -> List[str]:
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    spans = make_chunk_spans(text, max_chars=max_chars, chunk_mode="sentence")
    if not spans:
        return [text]
    return [text[start:end] for start, end in spans]


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
            "torch/pocket-tts not installed. They are optional (the default backend is "
            "maneko). Install the extra (`uv sync --extra torch`) and select it with "
            "NEB_TTS_BACKEND=torch."
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


def floats_to_int16(samples: Sequence[float]) -> np.ndarray:
    """Convert mono float audio (maneko's `list[float]`, ~[-1, 1]) to int16.

    Mirrors `tensor_to_int16`'s heuristic: if the peak looks like normalized
    float audio, scale by 32767; otherwise assume the values are already in
    int16 range.
    """
    a = np.asarray(samples, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return np.zeros(0, dtype=np.int16)
    max_abs = float(np.abs(a).max())
    if max_abs <= 1.5:
        a = np.clip(a, -1.0, 1.0) * 32767.0
    return np.rint(a).astype(np.int16)


def write_wav_mono_16k_or_24k(
    path: Path, samples_i16: np.ndarray, sample_rate: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    arr = np.ascontiguousarray(samples_i16, dtype=np.int16)
    data = arr.tobytes()

    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(data)

    tmp.replace(path)


# --- TTS backends --------------------------------------------------------------
# A thin interface over the inference engine so the synth pipeline is engine
# agnostic. The default is maneko (native Rust/candle q8); torch/pocket-tts is an
# optional fallback. Both return mono int16 numpy arrays; audio assembly (concat
# + pause padding) is done in numpy by the callers.


class _TTSBackend:
    name = "base"

    def sample_rate(self, language: Optional[str], layers: Optional[int]) -> int:
        raise NotImplementedError

    def prepare_voice(
        self, voice_prompt: str, language: Optional[str], layers: Optional[int]
    ) -> Any:
        raise NotImplementedError

    def generate(
        self,
        voice_handle: Any,
        text: str,
        language: Optional[str],
        layers: Optional[int],
    ) -> np.ndarray:
        raise NotImplementedError


class ManekoBackend(_TTSBackend):
    """Native maneko (Rust/candle q8). One resident Pocket; caches per language."""

    name = "maneko"

    def __init__(self) -> None:
        if maneko is None:
            raise RuntimeError(
                "maneko is not installed (the default TTS backend). Install it with "
                "`uv sync` (builds it from GitHub — needs a Rust toolchain), or select "
                "another backend with NEB_TTS_BACKEND=torch."
            )
        self._pocket = maneko.Pocket("cpu")

    @staticmethod
    def _stem(language: Optional[str], layers: Optional[int]) -> str:
        return language_util.resolve_maneko_language(language, layers)

    def sample_rate(self, language: Optional[str], layers: Optional[int]) -> int:
        return int(self._pocket.sample_rate(self._stem(language, layers)))

    def prepare_voice(
        self, voice_prompt: str, language: Optional[str], layers: Optional[int]
    ) -> Any:
        # maneko clones lazily and caches the voice-state per (voice, language)
        # internally, so there is nothing to precompute — just carry the path.
        return voice_prompt

    def generate(
        self,
        voice_handle: Any,
        text: str,
        language: Optional[str],
        layers: Optional[int],
    ) -> np.ndarray:
        floats = self._pocket.generate(text, self._stem(language, layers), voice_handle)
        return floats_to_int16(floats)


class TorchBackend(_TTSBackend):
    """Legacy pocket-tts (torch CPU) path, kept as an optional fallback."""

    name = "torch"

    def __init__(self) -> None:
        _require_tts()
        _install_tts_warning_filter()

    def sample_rate(self, language: Optional[str], layers: Optional[int]) -> int:
        return int(_load_tts_model(language, layers).sample_rate)

    def prepare_voice(
        self, voice_prompt: str, language: Optional[str], layers: Optional[int]
    ) -> Any:
        model = _load_tts_model(language, layers)
        return model.get_state_for_audio_prompt(voice_prompt)

    def generate(
        self,
        voice_handle: Any,
        text: str,
        language: Optional[str],
        layers: Optional[int],
    ) -> np.ndarray:
        model = _load_tts_model(language, layers)
        audio = model.generate_audio(voice_handle, text)
        return tensor_to_int16(audio).numpy()


_TTS_BACKENDS: Dict[str, _TTSBackend] = {}
_TTS_BACKEND_LOCK = threading.Lock()


def _make_backend(name: str) -> _TTSBackend:
    if name == "maneko":
        return ManekoBackend()
    if name == "torch":
        return TorchBackend()
    raise RuntimeError(
        f"Unknown NEB_TTS_BACKEND={name!r} (expected 'maneko', 'torch', or 'auto')."
    )


def _resolve_backend_name() -> str:
    selector = (os.environ.get("NEB_TTS_BACKEND") or "maneko").strip().lower()
    if selector == "auto":
        return "torch" if (torch is not None and TTSModel is not None) else "maneko"
    if selector in ("maneko", "torch"):
        return selector
    raise RuntimeError(
        f"Unknown NEB_TTS_BACKEND={selector!r} (expected 'maneko', 'torch', or 'auto')."
    )


def get_backend() -> _TTSBackend:
    """Return the configured TTS backend (cached per concrete engine).

    Selected by NEB_TTS_BACKEND: 'maneko' (default), 'torch', or 'auto' (torch if
    installed, else maneko). Re-reads the env each call so callers/tests can switch.
    """
    _ensure_hf_home()
    name = _resolve_backend_name()
    with _TTS_BACKEND_LOCK:
        backend = _TTS_BACKENDS.get(name)
        if backend is None:
            backend = _make_backend(name)
            _TTS_BACKENDS[name] = backend
        return backend


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
    language: Optional[str] = None,
    layers: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[List[str]], int]:
    language = language_util.normalize_language_tag(language)
    layers = language_util.resolve_layers(language, layers)
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
            "language": language,
            "layers": int(layers),
            "max_chars": int(max_chars),
            "pad_ms": int(pad_ms),
            "chunk_mode": chunk_mode,
            "chapters": manifest_chapters,
        }
        atomic_write_json(manifest_path, manifest)

    manifest["voice"] = voice
    manifest["language"] = language
    manifest["layers"] = int(layers)
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
    layers: Optional[int] = None,
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
    language = _resolve_book_language(book_dir)
    manifest, _chapter_chunks, _pad_ms = prepare_manifest(
        chapters=chapters,
        out_dir=out_dir,
        voice=voice,
        max_chars=max_chars,
        pad_ms=pad_ms,
        chunk_mode=chunk_mode,
        rechunk=rechunk,
        language=language,
        layers=layers,
    )
    return manifest


def _resolve_book_language(book_dir: Path) -> str:
    """Read language from <book_dir>/clean/toc.json; fallback to default."""
    toc_path = book_dir / "clean" / "toc.json"
    if not toc_path.exists():
        return language_util.DEFAULT_LANGUAGE
    try:
        toc = json.loads(toc_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return language_util.DEFAULT_LANGUAGE
    metadata = toc.get("metadata") if isinstance(toc, dict) else None
    raw = metadata.get("language") if isinstance(metadata, dict) else None
    return language_util.resolve_language(raw)


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
    language: Optional[str] = None,
    layers: Optional[int] = None,
) -> int:
    backend = get_backend()
    language = language_util.normalize_language_tag(language)
    layers = language_util.resolve_layers(language, layers)

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
            language=language,
            layers=layers,
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

    sample_rate = backend.sample_rate(language, layers)
    if manifest.get("sample_rate") != sample_rate:
        manifest["sample_rate"] = sample_rate
        atomic_write_json(manifest_path, manifest)

    voice_states: Dict[str, Any] = {}
    for voice_id, voice_prompt in voice_prompts.items():
        voice_states[voice_id] = backend.prepare_voice(voice_prompt, language, layers)
    write_status(out_dir, "synthesizing")

    base_pad_samples = int(round(sample_rate * (pad_ms / 1000.0)))
    pad_arrays: Dict[int, Optional[np.ndarray]] = {}

    def pad_array_for(multiplier: int) -> Optional[np.ndarray]:
        multiplier = max(1, int(multiplier))
        if base_pad_samples <= 0:
            return None
        if multiplier not in pad_arrays:
            total_samples = base_pad_samples * multiplier
            pad_arrays[multiplier] = (
                np.zeros(total_samples, dtype=np.int16)
                if total_samples > 0
                else None
            )
        return pad_arrays[multiplier]

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
                    chunk_text,
                    chapter_reading_map.get(chapter_id, []),
                    language=language,
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
                audio_parts: List[np.ndarray] = []
                for sub_idx, sub_text in enumerate(sub_texts, start=1):
                    sub_text = sub_text.strip()
                    if not sub_text:
                        continue
                    if not _ends_with_sentence_punct(sub_text):
                        sub_text = f"{sub_text}."
                    with _tts_warning_context(
                        chapter_id, chunk_idx, chapter_total, sub_idx, sub_total
                    ):
                        audio_parts.append(
                            backend.generate(voice_state, sub_text, language, layers)
                        )

                if not audio_parts:
                    with _tts_warning_context(chapter_id, chunk_idx, chapter_total, 1, 1):
                        audio_parts = [
                            backend.generate(voice_state, tts_text, language, layers)
                        ]

                if len(audio_parts) == 1:
                    a16 = audio_parts[0]
                else:
                    a16 = np.concatenate(audio_parts)

                pad_array = pad_array_for(pause_multiplier)
                if pad_array is not None and pad_array.size > 0:
                    a16 = np.concatenate([a16, pad_array])

                write_wav_mono_16k_or_24k(seg_path, a16, sample_rate=sample_rate)
                segment_paths.append(seg_path)
                dms = int(round(a16.size * 1000.0 / sample_rate))

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
    backend = get_backend()
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
    language = manifest.get("language")
    layers = manifest.get("layers")
    sample_rate = backend.sample_rate(language, layers)
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
    pad_array = np.zeros(pad_samples, dtype=np.int16) if pad_samples > 0 else None

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
    tts_text = prepare_tts_text(
        chunk_text,
        merged_reading_overrides,
        language=manifest.get("language") or "english",
    )
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

    voice_state = backend.prepare_voice(voice_prompt, language, layers)
    sub_texts = split_tts_text_for_synthesis(tts_text, max_chars=max_chars)
    sub_total = len(sub_texts)
    audio_parts: List[np.ndarray] = []
    for sub_idx, sub_text in enumerate(sub_texts, start=1):
        with _tts_warning_context(
            chapter_id, chunk_index + 1, chunk_count, sub_idx, sub_total
        ):
            audio_parts.append(
                backend.generate(voice_state, sub_text, language, layers)
            )
    if not audio_parts:
        with _tts_warning_context(chapter_id, chunk_index + 1, chunk_count, 1, 1):
            audio_parts = [
                backend.generate(voice_state, tts_text, language, layers)
            ]
    if len(audio_parts) == 1:
        a16 = audio_parts[0]
    else:
        a16 = np.concatenate(audio_parts)
    if pad_array is not None and pad_array.size > 0:
        a16 = np.concatenate([a16, pad_array])
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
    language: Optional[str] = None,
    layers: Optional[int] = None,
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
        language=language,
        layers=layers,
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
    language: Optional[str] = None,
    layers: Optional[int] = None,
) -> int:
    if out_dir is None:
        out_dir = book_dir / "tts"
    try:
        chapters = load_book_chapters(book_dir)
        write_combined_input(chapters, out_dir)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    resolved_language = language or _resolve_book_language(book_dir)
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
        language=resolved_language,
        layers=layers,
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
    language: Optional[str] = None,
    layers: Optional[int] = None,
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
        language=language or _resolve_book_language(book_dir),
        layers=layers,
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
        help="Voice prompt: wav path or hf:// URL",
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
    ap.add_argument(
        "--language",
        help="Pocket-tts language override (english, french, german, italian, portuguese, spanish)",
    )
    ap.add_argument(
        "--layers",
        type=int,
        choices=[6, 24],
        help="Flow-LM transformer layer count (6=fast, 24=higher quality)",
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
            language=args.language,
            layers=args.layers,
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
        language=args.language,
        layers=args.layers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
