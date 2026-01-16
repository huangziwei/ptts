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
from .voice import resolve_voice_prompt

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


def _next_word(text: str, start: int) -> str:
    match = re.search(r"[A-Za-z][A-Za-z'’\-]*", text[start:])
    if not match:
        return ""
    return match.group(0)


def _should_skip_sentence_split(paragraph: str, end: int, next_pos: int) -> bool:
    tail = paragraph[:end]
    next_word = _next_word(paragraph, next_pos)
    next_lower = next_word.lower()

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


def split_span_by_words(
    text: str, start: int, end: int, max_chars: int
) -> List[Tuple[int, int]]:
    segment = text[start:end]
    words = list(re.finditer(r"\S+", segment))
    if not words:
        return []
    spans: List[Tuple[int, int]] = []
    chunk_start = start + words[0].start()
    chunk_end = start + words[0].end()
    for word in words[1:]:
        word_start = start + word.start()
        word_end = start + word.end()
        if word_end - chunk_start > max_chars and chunk_end > chunk_start:
            spans.append((chunk_start, chunk_end))
            chunk_start = word_start
            chunk_end = word_end
        else:
            chunk_end = word_end
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


def prepare_tts_text(text: str) -> str:
    text = normalize_abbreviations(text)
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


def synthesize(
    chapters: Sequence[ChapterInput],
    voice: Optional[str],
    out_dir: Path,
    max_chars: int = 800,
    pad_ms: int = 150,
    chunk_mode: str = "sentence",
    rechunk: bool = False,
    base_dir: Optional[Path] = None,
) -> int:
    _require_tts()

    if base_dir is None:
        base_dir = Path.cwd()
    try:
        voice_prompt = resolve_voice_prompt(voice, base_dir=base_dir)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = out_dir / "segments"
    manifest_path = out_dir / "manifest.json"
    concat_path = out_dir / "concat.txt"
    chapters_path = out_dir / "chapters.ffmeta"

    try:
        manifest, chapter_chunks, pad_ms = prepare_manifest(
            chapters=chapters,
            out_dir=out_dir,
            voice=voice,
            max_chars=max_chars,
            pad_ms=pad_ms,
            chunk_mode=chunk_mode,
            rechunk=rechunk,
        )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    if rechunk and seg_dir.exists():
        shutil.rmtree(seg_dir)

    tts_model = TTSModel.load_model()
    sample_rate = int(tts_model.sample_rate)
    if manifest.get("sample_rate") != sample_rate:
        manifest["sample_rate"] = sample_rate
        atomic_write_json(manifest_path, manifest)

    voice_state = tts_model.get_state_for_audio_prompt(voice_prompt)

    pad_samples = int(round(sample_rate * (pad_ms / 1000.0)))
    pad_tensor = torch.zeros(pad_samples, dtype=torch.int16) if pad_samples > 0 else None

    segment_paths: List[Path] = []
    total_chunks = sum(len(chunks) for chunks in chapter_chunks)

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

    return 0


def synthesize_text(
    text_path: Path,
    voice: Optional[str],
    out_dir: Path,
    max_chars: int = 800,
    pad_ms: int = 150,
    chunk_mode: str = "sentence",
    rechunk: bool = False,
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
    )


if __name__ == "__main__":
    raise SystemExit(main())
