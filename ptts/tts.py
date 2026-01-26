from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_ABBREV_DOT_RE = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St)\.", re.IGNORECASE)
_ABBREV_SENT_RE = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St)\.$", re.IGNORECASE)
_SINGLE_INITIAL_RE = re.compile(r"\b[A-Z]\.$")
_NAME_INITIAL_RE = re.compile(r"\b([A-Z][a-z]+)\s+[A-Z]\.$")
_MULTI_INITIAL_RE = re.compile(r"(?:\b[A-Z]\.){2,}$")
_ABBREV_WHITELIST = {
    "a.k.a.",
    "a.m.",
    "b.a.",
    "b.f.a.",
    "b.s.",
    "b.s.n.",
    "c.e.o.",
    "c.f.o.",
    "c.i.a.",
    "c.o.o.",
    "c.p.a.",
    "cf.",
    "d.c.",
    "d.d.s.",
    "d.o.",
    "d.v.m.",
    "e.g.",
    "et al.",
    "etc.",
    "f.b.i.",
    "i.e.",
    "j.d.",
    "l.p.n.",
    "m.a.",
    "m.b.a.",
    "m.d.",
    "m.f.a.",
    "m.p.h.",
    "m.s.",
    "m.s.w.",
    "p.e.",
    "p.m.",
    "ph.d.",
    "r.n.",
    "u.k.",
    "u.n.",
    "u.s.",
    "u.s.a.",
    "viz.",
    "vs.",
}
_DOT_SPACE_DOT_RE = re.compile(r"(?<=\.)\s+(?=[A-Za-z]\.)")
_LAST_DOT_TOKEN_RE = re.compile(r"([A-Za-z][A-Za-z'-]*\.)\s*$")
_NEXT_DOT_TOKEN_RE = re.compile(r"([A-Za-z][A-Za-z'-]*\.)")
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
_ROMAN_STANDALONE_RE = re.compile(r"^(?P<num>[IVXLCDM]+)(?P<trail>[^A-Za-z0-9]*)$", re.IGNORECASE)
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

    if _is_whitelisted_abbrev_boundary(tail, paragraph, next_pos):
        return True

    if _ABBREV_SENT_RE.search(tail):
        if next_lower and next_lower in _SENTENCE_STARTERS:
            return False
        return True

    if _MULTI_INITIAL_RE.search(tail):
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
    if chunk_mode not in ("sentence", "packed"):
        raise ValueError(f"Unsupported chunk_mode: {chunk_mode}")

    spans: List[Tuple[int, int]] = []
    for para_start, para_end in split_paragraph_spans(text):
        paragraph = text[para_start:para_end]
        sentence_spans = split_sentence_spans(paragraph, para_start)
        if chunk_mode == "sentence":
            for sent_start, sent_end in sentence_spans:
                if sent_end - sent_start > max_chars:
                    spans.extend(
                        split_span_by_words(text, sent_start, sent_end, max_chars)
                    )
                else:
                    spans.append((sent_start, sent_end))
        else:
            buf_start: Optional[int] = None
            buf_end: Optional[int] = None
            for sent_start, sent_end in sentence_spans:
                if sent_end - sent_start > max_chars:
                    if buf_start is not None and buf_end is not None:
                        spans.append((buf_start, buf_end))
                        buf_start = None
                        buf_end = None
                    spans.extend(
                        split_span_by_words(text, sent_start, sent_end, max_chars)
                    )
                    continue
                if buf_start is None:
                    buf_start, buf_end = sent_start, sent_end
                    continue
                if sent_end - buf_start > max_chars:
                    spans.append((buf_start, buf_end))
                    buf_start, buf_end = sent_start, sent_end
                else:
                    buf_end = sent_end
            if buf_start is not None and buf_end is not None:
                spans.append((buf_start, buf_end))
    return spans


def make_chunks(text: str, max_chars: int, chunk_mode: str = "sentence") -> List[str]:
    spans = make_chunk_spans(text, max_chars=max_chars, chunk_mode=chunk_mode)
    return [text[start:end] for start, end in spans]


def normalize_abbreviations(text: str) -> str:
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
    thousands, rest = divmod(value, 1000)
    if rest == 0:
        return f"{_WORD_ONES[thousands]} thousand"
    return f"{_WORD_ONES[thousands]} thousand {_int_to_words(rest)}"


def _normalize_roman_numerals(text: str) -> str:
    def replace_heading(match: re.Match[str]) -> str:
        number = _roman_to_int(match.group("num"))
        if number is None:
            return match.group(0)
        return f"{match.group('label')} {_int_to_words(number)}"

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


def prepare_tts_text(text: str) -> str:
    text = normalize_abbreviations(text)
    text = _normalize_roman_numerals(text)
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
        existing_mode = manifest.get("chunk_mode", "packed")
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
            chapter_chunks.append(chunks)
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
                    "durations_ms": [None] * len(chunks),
                }
            )

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
    max_chars: int = 800,
    pad_ms: int = 150,
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
    max_chars: int = 800,
    pad_ms: int = 150,
    chunk_mode: str = "sentence",
    rechunk: bool = False,
    wipe_segments: Optional[bool] = None,
    only_chapter_ids: Optional[set[str]] = None,
    voice_map_path: Optional[Path] = None,
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
    sample_rate = int(tts_model.sample_rate)
    if manifest.get("sample_rate") != sample_rate:
        manifest["sample_rate"] = sample_rate
        atomic_write_json(manifest_path, manifest)

    voice_states: Dict[str, Any] = {}
    for voice_id, voice_prompt in voice_prompts.items():
        voice_states[voice_id] = tts_model.get_state_for_audio_prompt(voice_prompt)
    write_status(out_dir, "synthesizing")

    pad_samples = int(round(sample_rate * (pad_ms / 1000.0)))
    pad_tensor = torch.zeros(pad_samples, dtype=torch.int16) if pad_samples > 0 else None

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
                segment_paths.append(seg_path)

                progress.update(
                    chapter_task,
                    description=f"{chapter_id}: {chapter_title} ({chunk_idx}/{chapter_total})",
                )

                if seg_path.exists() and is_valid_wav(seg_path):
                    dms = wav_duration_ms(seg_path)
                    if ch_entry["durations_ms"][chunk_idx - 1] != dms:
                        ch_entry["durations_ms"][chunk_idx - 1] = dms
                        atomic_write_json(manifest_path, manifest)
                    progress.advance(chapter_task, 1)
                    progress.advance(overall_task, 1)
                    continue

                tts_text = prepare_tts_text(chunk_text)
                voice_id = chapter_voice_map.get(chapter_id, default_voice)
                voice_state = voice_states[voice_id]
                audio = tts_model.generate_audio(voice_state, tts_text)
                a16 = tensor_to_int16(audio)

                if pad_tensor is not None and pad_tensor.numel() > 0:
                    a16 = torch.cat([a16, pad_tensor], dim=0)

                write_wav_mono_16k_or_24k(seg_path, a16, sample_rate=sample_rate)
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


def synthesize_text(
    text_path: Path,
    voice: Optional[str],
    out_dir: Path,
    max_chars: int = 800,
    pad_ms: int = 150,
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
        base_dir=base_dir,
    )


def synthesize_book(
    book_dir: Path,
    voice: Optional[str],
    out_dir: Optional[Path] = None,
    max_chars: int = 800,
    pad_ms: int = 150,
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
        base_dir=base_dir,
    )


def synthesize_book_sample(
    book_dir: Path,
    voice: Optional[str],
    out_dir: Optional[Path] = None,
    max_chars: int = 800,
    pad_ms: int = 150,
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
        default=800,
        help="Max characters per chunk (default: 800)",
    )
    ap.add_argument(
        "--pad-ms",
        type=int,
        default=150,
        help="Silence to append to each chunk in ms (default: 150)",
    )
    ap.add_argument(
        "--chunk-mode",
        choices=["sentence", "packed"],
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
