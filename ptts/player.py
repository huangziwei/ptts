from __future__ import annotations

import html
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, List, Optional, Union

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from . import epub as epub_util
from . import sanitize
from .text import guess_title_from_path, read_clean_text
from .voice import BUILTIN_VOICES, DEFAULT_VOICE, resolve_voice_prompt

def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _no_store(data: dict) -> JSONResponse:
    return JSONResponse(data, headers={"Cache-Control": "no-store"})

def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


def _rules_path(base_dir: Path) -> Path:
    return base_dir / ".codex" / "ptts-rules.json"


def _load_rules_payload(base_dir: Path) -> dict:
    rules_path = _rules_path(base_dir)
    if not rules_path.exists():
        return {
            "replace_defaults": False,
            "drop_chapter_title_patterns": [],
            "section_cutoff_patterns": [],
            "remove_patterns": [],
            "paragraph_breaks": "double",
        }
    data = json.loads(rules_path.read_text(encoding="utf-8"))
    data.setdefault("replace_defaults", False)
    data.setdefault("drop_chapter_title_patterns", [])
    data.setdefault("section_cutoff_patterns", [])
    data.setdefault("remove_patterns", [])
    data.setdefault("paragraph_breaks", "double")
    return data


def _write_rules_payload(base_dir: Path, payload: dict) -> None:
    paragraph_breaks = str(payload.get("paragraph_breaks", "double") or "double").strip().lower()
    if paragraph_breaks not in sanitize.PARAGRAPH_BREAK_OPTIONS:
        paragraph_breaks = "double"
    data = {
        "replace_defaults": bool(payload.get("replace_defaults", False)),
        "drop_chapter_title_patterns": list(
            payload.get("drop_chapter_title_patterns", [])
        ),
        "section_cutoff_patterns": list(payload.get("section_cutoff_patterns", [])),
        "remove_patterns": list(payload.get("remove_patterns", [])),
        "paragraph_breaks": paragraph_breaks,
    }
    if not data["replace_defaults"]:
        for key, defaults in sanitize.DEFAULT_RULES.items():
            data[key] = [entry for entry in data[key] if entry not in defaults]
    rules_path = _rules_path(base_dir)
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(rules_path, data)


def _highlight_ranges(text: str, patterns: list[re.Pattern]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            if match.start() == match.end():
                continue
            ranges.append((match.start(), match.end()))
    if not ranges:
        return []
    ranges.sort(key=lambda item: (item[0], item[1]))
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _render_highlight(text: str, ranges: list[tuple[int, int]]) -> str:
    if not ranges:
        return html.escape(text)
    parts: list[str] = []
    last = 0
    for start, end in ranges:
        parts.append(html.escape(text[last:start]))
        parts.append(f"<mark>{html.escape(text[start:end])}</mark>")
        last = end
    parts.append(html.escape(text[last:]))
    return "".join(parts)


def _load_chapter_text(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _pick_preview_chapter(clean_toc: dict, chapter_index: Optional[int]) -> dict:
    chapters = clean_toc.get("chapters", []) if isinstance(clean_toc, dict) else []
    if not chapters:
        return {}
    if chapter_index is not None:
        for entry in chapters:
            if entry.get("index") == chapter_index:
                return entry
    return chapters[0]


def _find_clean_chapter(clean_toc: dict, chapter_index: int) -> Optional[dict]:
    chapters = clean_toc.get("chapters", []) if isinstance(clean_toc, dict) else []
    for entry in chapters:
        if entry.get("index") == chapter_index:
            return entry
    return None


def _resolve_raw_path(book_dir: Path, raw_toc: dict, clean_entry: dict) -> Optional[Path]:
    source_index = clean_entry.get("source_index")
    if source_index is None:
        return None
    for entry in raw_toc.get("chapters", []):
        if entry.get("index") == source_index:
            rel = entry.get("path")
            if rel:
                return book_dir / rel
    return None


def _slug_from_title(title: str, fallback: str) -> str:
    base = title.strip() if title else fallback
    slug = epub_util.slugify(base)
    return slug or "book"


def _source_type_from_toc(toc: dict) -> str:
    source = str(toc.get("source_epub") or "")
    suffix = Path(source).suffix.lower()
    if suffix == ".txt":
        return "txt"
    if suffix == ".epub":
        return "epub"
    return "unknown"


def _resolve_book_dir(root_dir: Path, book_id: str) -> Path:
    candidate = (root_dir / book_id).resolve()
    if root_dir not in candidate.parents and candidate != root_dir:
        raise HTTPException(status_code=404, detail="Book not found.")
    if not (candidate / "clean" / "toc.json").exists():
        raise HTTPException(status_code=404, detail="Book not found.")
    return candidate


def _sample_chapter_info(book_dir: Path) -> tuple[str, str]:
    manifest = _load_json(book_dir / "tts" / "manifest.json")
    chapters = manifest.get("chapters") if isinstance(manifest, dict) else []
    if isinstance(chapters, list) and chapters:
        entry = chapters[0]
        return str(entry.get("id") or ""), str(entry.get("title") or "")

    clean_toc = _load_json(book_dir / "clean" / "toc.json")
    entries = clean_toc.get("chapters") if isinstance(clean_toc, dict) else []
    if isinstance(entries, list) and entries:
        entry = entries[0]
        title = str(entry.get("title") or "")
        rel_path = entry.get("path") or ""
        if rel_path:
            return Path(rel_path).stem, title
        idx = entry.get("index") or 1
        slug = epub_util.slugify(title or "chapter")
        return f"{int(idx):04d}-{slug}", title

    return "", ""

def _book_summary(book_dir: Path) -> dict:
    toc = _load_json(book_dir / "clean" / "toc.json")
    metadata = toc.get("metadata", {}) if isinstance(toc, dict) else {}
    cover = metadata.get("cover") or {}
    cover_path = cover.get("path") or ""
    cover_url = f"/audio/{book_dir.name}/{cover_path}" if cover_path else ""
    chapters = toc.get("chapters", []) if isinstance(toc, dict) else []
    source_type = _source_type_from_toc(toc) if isinstance(toc, dict) else "unknown"
    has_audio = (book_dir / "tts" / "manifest.json").exists()
    return {
        "id": book_dir.name,
        "title": metadata.get("title") or book_dir.name,
        "authors": metadata.get("authors") or [],
        "year": metadata.get("year") or "",
        "cover_url": cover_url,
        "has_audio": has_audio,
        "chapter_count": len(chapters) if isinstance(chapters, list) else 0,
        "source_type": source_type,
    }


def _normalize_voice_value(value: object, repo_root: Path) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    cleaned = value.strip()
    if not cleaned:
        return ""
    if cleaned.lower() == "default":
        return DEFAULT_VOICE
    candidate = Path(cleaned)
    if candidate.is_absolute():
        try:
            rel = candidate.relative_to(repo_root)
        except ValueError:
            return cleaned
        return rel.as_posix()
    return cleaned


def _normalize_metadata_text(value: Optional[str]) -> str:
    return str(value or "").strip()


def _normalize_authors(raw: Union[str, List[str], None]) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    else:
        items = re.split(r"[,\n]+", str(raw))
    cleaned: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _default_book_voice(book_dir: Path, repo_root: Path) -> str:
    manifest = _load_json(book_dir / "tts" / "manifest.json")
    fallback = _normalize_voice_value(manifest.get("voice"), repo_root)
    return fallback or DEFAULT_VOICE


def _voice_map_path(book_dir: Path) -> Path:
    return book_dir / "voice-map.json"


def _sanitize_voice_map(payload: dict, repo_root: Path, fallback_default: str) -> dict:
    default_voice = _normalize_voice_value(payload.get("default"), repo_root)
    if not default_voice:
        default_voice = fallback_default or DEFAULT_VOICE
    chapters: dict[str, str] = {}
    raw_chapters = payload.get("chapters", {})
    if isinstance(raw_chapters, dict):
        for key, value in raw_chapters.items():
            voice = _normalize_voice_value(value, repo_root)
            if not voice or voice == default_voice:
                continue
            chapters[str(key)] = voice
    return {"default": default_voice, "chapters": chapters}


def _load_voice_map(book_dir: Path, repo_root: Path) -> dict:
    path = _voice_map_path(book_dir)
    if not path.exists():
        return {"default": _default_book_voice(book_dir, repo_root), "chapters": {}}
    data = _load_json(path)
    return _sanitize_voice_map(
        data if isinstance(data, dict) else {},
        repo_root,
        fallback_default=_default_book_voice(book_dir, repo_root),
    )


def _book_details(book_dir: Path, repo_root: Path) -> dict:
    toc = _load_json(book_dir / "clean" / "toc.json")
    metadata = toc.get("metadata", {}) if isinstance(toc, dict) else {}
    cover = metadata.get("cover") or {}
    cover_path = cover.get("path") or ""
    cover_url = f"/audio/{book_dir.name}/{cover_path}" if cover_path else ""
    source_type = _source_type_from_toc(toc) if isinstance(toc, dict) else "unknown"

    manifest = _load_json(book_dir / "tts" / "manifest.json")
    chapters: List[dict] = []
    pad_ms = 0
    last_voice = ""
    if manifest and isinstance(manifest.get("chapters"), list):
        try:
            pad_ms = int(manifest.get("pad_ms") or 0)
        except (TypeError, ValueError):
            pad_ms = 0
        last_voice = _normalize_voice_value(manifest.get("voice"), repo_root)
        for entry in manifest["chapters"]:
            chunk_spans = entry.get("chunk_spans", [])
            if not isinstance(chunk_spans, list):
                chunk_spans = []
            clean_text = ""
            rel_path = entry.get("path")
            if rel_path:
                clean_path = book_dir / rel_path
                if clean_path.exists():
                    clean_text = read_clean_text(clean_path)
            chapters.append(
                {
                    "id": entry.get("id") or "",
                    "title": entry.get("title") or entry.get("id") or "Chapter",
                    "chunk_spans": chunk_spans,
                    "chunk_count": len(chunk_spans),
                    "clean_text": clean_text,
                }
            )

    return {
        "book": {
            "id": book_dir.name,
            "title": metadata.get("title") or book_dir.name,
            "authors": metadata.get("authors") or [],
            "year": metadata.get("year") or "",
            "cover_url": cover_url,
            "has_audio": bool(chapters),
            "pad_ms": pad_ms,
            "last_voice": last_voice,
            "source_type": source_type,
        },
        "chapters": chapters,
        "audio_base": f"/audio/{book_dir.name}/tts/segments",
    }

def _playback_path(book_dir: Path) -> Path:
    return book_dir / "playback.json"


def _m4b_library_dir(root_dir: Path) -> Path:
    return root_dir / "_m4b"


def _merge_output_path(book_dir: Path) -> Path:
    library_dir = _m4b_library_dir(book_dir.parent)
    return library_dir / f"{book_dir.name}.m4b"


def _merge_ready(book_dir: Path) -> bool:
    manifest = _load_json(book_dir / "tts" / "manifest.json")
    if not manifest:
        return False
    progress = _compute_progress(manifest)
    if not progress or not progress.get("total"):
        return False
    return progress.get("done", 0) >= progress.get("total", 0)


def _ffmpeg_install_command() -> str:
    if sys.platform == "darwin":
        installer = shutil.which("brew")
        if installer is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. Install Homebrew (https://brew.sh/) "
                "or install ffmpeg manually, then retry."
            )
        install_cmd = f"{installer} install ffmpeg"
    else:
        installer = shutil.which("apt-get") or shutil.which("apt")
        if installer is None:
            raise RuntimeError(
                "ffmpeg not found on PATH and apt-get is unavailable. "
                "Install ffmpeg manually, then retry."
            )
        install_cmd = f"{installer} update && {installer} install -y ffmpeg"

    return f"export DEBIAN_FRONTEND=noninteractive; {install_cmd}"


def _build_merge_command(
    book_dir: Path,
    output_path: Path,
    overwrite: bool,
    install_ffmpeg: bool,
    progress_path: Path,
) -> list[str]:
    merge_cmd = [
        "uv",
        "run",
        "ptts",
        "merge",
        "--book",
        str(book_dir),
        "--output",
        str(output_path),
        "--progress-file",
        str(progress_path),
    ]
    if overwrite:
        merge_cmd.append("--overwrite")
    if not install_ffmpeg:
        return merge_cmd

    install_cmd = _ffmpeg_install_command()
    merge_cmd_str = " ".join(shlex.quote(part) for part in merge_cmd)
    combined = f"{install_cmd} && {merge_cmd_str}"
    return ["bash", "-lc", combined]


def _sanitize_playback(data: dict) -> dict:
    last = data.get("last_played")
    if not isinstance(last, int) or last < 0:
        last = None
    furthest = data.get("furthest_played")
    if not isinstance(furthest, int) or furthest < 0:
        furthest = None
    bookmarks: List[dict] = []
    raw_marks = data.get("bookmarks")
    if isinstance(raw_marks, list):
        for entry in raw_marks:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("index")
            if not isinstance(idx, int) or idx < 0:
                continue
            cleaned = {"index": idx}
            label = entry.get("label")
            created_at = entry.get("created_at")
            if isinstance(label, str):
                cleaned["label"] = label
            if isinstance(created_at, int):
                cleaned["created_at"] = created_at
            bookmarks.append(cleaned)
    return {
        "last_played": last,
        "furthest_played": furthest,
        "bookmarks": bookmarks,
    }




def _compute_progress(manifest: dict) -> dict:
    chapters = manifest.get("chapters", [])
    total = 0
    done = 0
    current = None
    global_index = 0

    for entry in chapters if isinstance(chapters, list) else []:
        chunks = entry.get("chunks", [])
        durations = entry.get("durations_ms", [])
        if not isinstance(chunks, list):
            chunks = []
        total += len(chunks)
        for idx in range(len(chunks)):
            global_index += 1
            if idx < len(durations) and durations[idx] is not None:
                done += 1
                continue
            if current is None:
                current = {
                    "chapter_id": entry.get("id") or "",
                    "chapter_title": entry.get("title") or "",
                    "chunk_index": idx + 1,
                    "chunk_total": len(chunks),
                    "global_index": global_index,
                }

    percent = (done / total * 100.0) if total else 0.0
    return {
        "total": total,
        "done": done,
        "percent": round(percent, 2),
        "current": current,
    }


def _compute_chapter_progress(manifest: dict, chapter_id: str) -> dict:
    chapters = manifest.get("chapters", [])
    entry = None
    for item in chapters if isinstance(chapters, list) else []:
        if (item.get("id") or "") == chapter_id:
            entry = item
            break
    if not entry:
        return {}
    chunks = entry.get("chunks", [])
    durations = entry.get("durations_ms", [])
    if not isinstance(chunks, list):
        chunks = []
    total = len(chunks)
    done = 0
    current = None
    for idx in range(total):
        if idx < len(durations) and durations[idx] is not None:
            done += 1
            continue
        if current is None:
            current = {
                "chapter_id": entry.get("id") or "",
                "chapter_title": entry.get("title") or "",
                "chunk_index": idx + 1,
                "chunk_total": total,
                "global_index": idx + 1,
            }
    percent = (done / total * 100.0) if total else 0.0
    return {
        "total": total,
        "done": done,
        "percent": round(percent, 2),
        "current": current,
    }


def _ffmpeg_log_path(repo_root: Path) -> Path:
    return repo_root / ".cache" / "ffmpeg-install.log"


def _spawn_ffmpeg_install(
    repo_root: Path,
) -> tuple[Optional["FfmpegJob"], Optional[str]]:
    if shutil.which("ffmpeg") is not None:
        return None, None
    try:
        install_cmd = _ffmpeg_install_command()
    except RuntimeError as exc:
        return None, str(exc)

    log_path = _ffmpeg_log_path(repo_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", "-lc", install_cmd],
        cwd=str(repo_root),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return (
        FfmpegJob(
            process=process,
            started_at=time.time(),
            log_path=log_path,
            log_handle=log_handle,
        ),
        None,
    )


def _load_tts_status(book_dir: Path) -> dict:
    path = book_dir / "tts" / "status.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class SynthJob:
    book_id: str
    book_dir: Path
    process: subprocess.Popen
    started_at: float
    log_path: Path
    voice: str
    max_chars: int
    pad_ms: int
    chunk_mode: str
    rechunk: bool
    mode: str = "tts"
    sample_chapter_id: Optional[str] = None
    sample_chapter_title: Optional[str] = None
    log_handle: Optional[IO[str]] = None
    exit_code: Optional[int] = None
    ended_at: Optional[float] = None


@dataclass
class MergeJob:
    book_id: str
    book_dir: Path
    process: subprocess.Popen
    started_at: float
    log_path: Path
    output_path: Path
    install_ffmpeg: bool
    progress_path: Path
    log_handle: Optional[IO[str]] = None
    exit_code: Optional[int] = None
    ended_at: Optional[float] = None


@dataclass
class FfmpegJob:
    process: subprocess.Popen
    started_at: float
    log_path: Path
    log_handle: Optional[IO[str]] = None
    exit_code: Optional[int] = None
    ended_at: Optional[float] = None


class SynthRequest(BaseModel):
    book_id: str
    voice: Optional[str] = None
    max_chars: int = 800
    pad_ms: int = 150
    chunk_mode: str = "sentence"
    rechunk: bool = False
    use_voice_map: bool = False


class MergeRequest(BaseModel):
    book_id: str
    overwrite: bool = False


class StopRequest(BaseModel):
    book_id: str


class ClearRequest(BaseModel):
    book_id: str


class RulesPayload(BaseModel):
    drop_chapter_title_patterns: List[str] = []
    section_cutoff_patterns: List[str] = []
    remove_patterns: List[str] = []
    paragraph_breaks: str = "double"
    replace_defaults: bool = False


class VoiceMapPayload(BaseModel):
    default: Optional[str] = None
    chapters: dict = {}


class ChapterAction(BaseModel):
    book_id: str
    title: str
    chapter_index: Optional[int] = None


class SanitizeRequest(BaseModel):
    book_id: str


class CleanEditPayload(BaseModel):
    book_id: str
    chapter_index: int
    text: str


class PlaybackPayload(BaseModel):
    last_played: Optional[int] = None
    furthest_played: Optional[int] = None
    bookmarks: List[dict] = []


class DeleteBookRequest(BaseModel):
    book_id: str


class MetadataPayload(BaseModel):
    book_id: str
    title: Optional[str] = None
    authors: Union[str, List[str], None] = None
    year: Optional[str] = None


def create_app(root_dir: Path) -> FastAPI:
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    app = FastAPI()
    root_dir = root_dir.resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    repo_root = _find_repo_root(root_dir)
    jobs: dict[str, SynthJob] = {}
    merge_jobs: dict[str, MergeJob] = {}
    ffmpeg_job: Optional[FfmpegJob] = None
    ffmpeg_error: Optional[str] = None

    app.mount("/audio", StaticFiles(directory=str(root_dir)), name="audio")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        context = {"request": request, "root_dir": str(root_dir)}
        return templates.TemplateResponse("player.html", context)

    @app.get("/api/books")
    def list_books() -> JSONResponse:
        books: List[dict] = []
        for child in sorted(root_dir.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            if child.name == "_m4b":
                continue
            toc_path = child / "clean" / "toc.json"
            if not toc_path.exists():
                continue
            books.append(_book_summary(child))
        def sort_key(entry: dict) -> str:
            title = str(entry.get("title") or "").strip().lower()
            if title.startswith(("the ", "a ", "an ")):
                parts = title.split(" ", 1)
                if len(parts) == 2:
                    title = parts[1]
            return title or str(entry.get("id") or "").lower()

        books.sort(key=sort_key)
        return _no_store({"root": str(root_dir), "books": books})

    @app.get("/api/books/{book_id}")
    def get_book(book_id: str) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        return _no_store(_book_details(book_dir, repo_root))

    @app.get("/api/books/{book_id}/voices")
    def get_book_voices(book_id: str) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        payload = _load_voice_map(book_dir, repo_root)
        return _no_store(payload)

    @app.post("/api/books/{book_id}/voices")
    def set_book_voices(book_id: str, payload: VoiceMapPayload) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        data = _sanitize_voice_map(
            payload.dict(),
            repo_root,
            fallback_default=_default_book_voice(book_dir, repo_root),
        )
        path = _voice_map_path(book_dir)
        _atomic_write_json(path, data)
        return _no_store(data)

    @app.post("/api/books/delete")
    def delete_book(payload: DeleteBookRequest) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        synth_job = jobs.get(payload.book_id)
        if synth_job and synth_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop TTS before deleting.")
        merge_job = merge_jobs.get(payload.book_id)
        if merge_job and merge_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop merge before deleting.")
        m4b_path = _merge_output_path(book_dir)
        if m4b_path.exists():
            m4b_path.unlink()
        if book_dir.exists():
            shutil.rmtree(book_dir)
        return _no_store({"status": "deleted", "book_id": payload.book_id})

    @app.post("/api/books/metadata")
    def update_metadata(payload: MetadataPayload) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        updates: dict[str, object] = {}
        if payload.title is not None:
            updates["title"] = _normalize_metadata_text(payload.title)
        if payload.authors is not None:
            updates["authors"] = _normalize_authors(payload.authors)
        if payload.year is not None:
            updates["year"] = _normalize_metadata_text(payload.year)

        for path in (book_dir / "toc.json", book_dir / "clean" / "toc.json"):
            if not path.exists():
                continue
            data = _load_json(path)
            if not isinstance(data, dict):
                continue
            metadata = data.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            for key, value in updates.items():
                metadata[key] = value
            data["metadata"] = metadata
            _atomic_write_json(path, data)

        metadata_payload: dict = {}
        clean_toc = _load_json(book_dir / "clean" / "toc.json")
        if isinstance(clean_toc, dict):
            meta = clean_toc.get("metadata")
            if isinstance(meta, dict):
                metadata_payload = meta
        if not metadata_payload:
            raw_toc = _load_json(book_dir / "toc.json")
            if isinstance(raw_toc, dict):
                meta = raw_toc.get("metadata")
                if isinstance(meta, dict):
                    metadata_payload = meta

        return _no_store(
            {
                "status": "ok",
                "metadata": metadata_payload,
                "book": _book_summary(book_dir),
            }
        )

    @app.get("/api/voices")
    def list_voices() -> JSONResponse:
        voices_dir = repo_root / "voices"
        local: List[dict] = []
        if voices_dir.exists():
            for wav in sorted(voices_dir.glob("*.wav")):
                try:
                    rel = wav.relative_to(repo_root)
                    value = rel.as_posix()
                except ValueError:
                    value = str(wav)
                local.append({"label": wav.stem, "value": value})
        builtin = [
            {"label": name, "value": name}
            for name in sorted(BUILTIN_VOICES.keys())
        ]
        return _no_store(
            {"local": local, "builtin": builtin, "default": DEFAULT_VOICE}
        )

    @app.get("/api/chunk-status")
    def chunk_status(
        book_id: str,
        chapter_id: str,
        chunk: int,
    ) -> JSONResponse:
        if chunk < 1:
            return _no_store({"exists": False})
        book_dir = _resolve_book_dir(root_dir, book_id)
        wav_path = (
            book_dir
            / "tts"
            / "segments"
            / chapter_id
            / f"{chunk:06d}.wav"
        )
        exists = wav_path.exists() and wav_path.is_file() and wav_path.stat().st_size > 0
        return _no_store({"exists": exists})

    @app.get("/api/playback")
    def playback_get(book_id: str) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        path = _playback_path(book_dir)
        exists = path.exists()
        data = _load_json(path) if exists else {}
        payload = _sanitize_playback(data)
        payload["exists"] = exists
        return _no_store(payload)

    @app.post("/api/playback")
    def playback_set(book_id: str, payload: PlaybackPayload) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        cleaned = _sanitize_playback(payload.dict())
        cleaned["updated_unix"] = int(time.time())
        _atomic_write_json(_playback_path(book_dir), cleaned)
        cleaned["exists"] = True
        return _no_store(cleaned)

    @app.post("/api/ingest")
    def ingest_file(file: UploadFile = File(...), override: bool = False) -> JSONResponse:
        filename = file.filename or ""
        suffix = Path(filename).suffix.lower()
        if suffix not in {".epub", ".txt"}:
            raise HTTPException(
                status_code=400,
                detail="Only .epub or .txt files are supported.",
            )

        tmp_path = None
        title = Path(filename).stem
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                shutil.copyfileobj(file.file, handle)
                tmp_path = Path(handle.name)

            if suffix == ".epub":
                try:
                    book = epub_util.read_epub(tmp_path)
                    metadata = epub_util.extract_metadata(book)
                    if metadata.get("title"):
                        title = str(metadata.get("title") or "").strip() or title
                except Exception:
                    metadata = {}
            else:
                title = guess_title_from_path(tmp_path) or title

            slug = _slug_from_title(title, Path(filename).stem)
            out_dir = root_dir / slug
            if out_dir.exists():
                if not override:
                    return JSONResponse(
                        status_code=409,
                        content={
                            "detail": f"Book already exists: {slug}",
                            "book_id": slug,
                        },
                    )
                synth_job = jobs.get(slug)
                if synth_job and synth_job.process.poll() is None:
                    return JSONResponse(
                        status_code=409,
                        content={
                            "detail": "Stop TTS before overwriting.",
                            "book_id": slug,
                        },
                    )
                merge_job = merge_jobs.get(slug)
                if merge_job and merge_job.process.poll() is None:
                    return JSONResponse(
                        status_code=409,
                        content={
                            "detail": "Stop merge before overwriting.",
                            "book_id": slug,
                        },
                    )
                resolved = out_dir.resolve()
                if root_dir not in resolved.parents and resolved != root_dir:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid book path.",
                    )
                m4b_path = _merge_output_path(resolved)
                if m4b_path.exists():
                    m4b_path.unlink()
                if resolved.is_dir():
                    shutil.rmtree(resolved)
                else:
                    resolved.unlink()
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "ingest.log"
            stage = "ingest"
            with log_path.open("w", encoding="utf-8") as log_handle:
                cmd = [
                    "uv",
                    "run",
                    "ptts",
                    "ingest",
                    "--input",
                    str(tmp_path),
                    "--out",
                    str(out_dir),
                ]
                proc = subprocess.run(
                    cmd,
                    cwd=str(repo_root),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                if proc.returncode == 0:
                    stage = "sanitize"
                    log_handle.write("\n--- sanitize ---\n")
                    cmd = [
                        "uv",
                        "run",
                        "ptts",
                        "sanitize",
                        "--book",
                        str(out_dir),
                        "--overwrite",
                    ]
                    proc = subprocess.run(
                        cmd,
                        cwd=str(repo_root),
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
            if proc.returncode != 0:
                detail = (
                    f"Sanitize failed. Check {log_path}."
                    if stage == "sanitize"
                    else f"Ingest failed. Check {log_path}."
                )
                raise HTTPException(
                    status_code=400,
                    detail=detail,
                )
            return _no_store(
                {
                    "status": "ok",
                    "book_id": slug,
                    "title": title,
                    "sanitized": True,
                }
            )
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

    @app.get("/api/sanitize/preview")
    def sanitize_preview(
        book_id: str, chapter: Optional[int] = None
    ) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        clean_toc = _load_json(book_dir / "clean" / "toc.json")
        raw_toc = _load_json(book_dir / "toc.json")
        report = _load_json(book_dir / "clean" / "report.json")
        if not clean_toc:
            raise HTTPException(status_code=404, detail="Missing clean/toc.json.")
        rules = sanitize.load_rules(None, repo_root)
        selected = _pick_preview_chapter(clean_toc, chapter)

        clean_path = None
        clean_rel = selected.get("path") if selected else None
        if clean_rel:
            clean_path = book_dir / clean_rel
        raw_path = _resolve_raw_path(book_dir, raw_toc, selected)

        raw_text = _load_chapter_text(raw_path)
        clean_text = _load_chapter_text(clean_path)
        patterns = sanitize.compile_patterns(rules.remove_patterns)
        raw_ranges = _highlight_ranges(raw_text, patterns)

        dropped = []
        for entry in report.get("chapters", []):
            if entry.get("dropped"):
                dropped.append(
                    {
                        "index": entry.get("index"),
                        "title": entry.get("title") or "",
                    }
                )

        payload = {
            "book_id": book_id,
            "chapters": clean_toc.get("chapters", []),
            "selected": selected,
            "raw_text": _render_highlight(raw_text, raw_ranges),
            "clean_text": clean_text,
            "dropped": dropped,
            "rules": {
            "replace_defaults": rules.replace_defaults,
            "drop_chapter_title_patterns": rules.drop_chapter_title_patterns,
            "section_cutoff_patterns": rules.section_cutoff_patterns,
            "remove_patterns": rules.remove_patterns,
            "paragraph_breaks": rules.paragraph_breaks,
        },
    }
        return _no_store(payload)

    @app.post("/api/sanitize/rules")
    def sanitize_rules(payload: RulesPayload) -> JSONResponse:
        _write_rules_payload(repo_root, payload.dict())
        return _no_store(payload.dict())

    @app.post("/api/sanitize/drop")
    def sanitize_drop(payload: ChapterAction) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        synth_job = jobs.get(payload.book_id)
        if synth_job and synth_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop TTS before editing.")
        merge_job = merge_jobs.get(payload.book_id)
        if merge_job and merge_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop merge before editing.")
        rules = _load_rules_payload(repo_root)
        pattern = f"^{re.escape(payload.title)}$"
        patterns = list(rules.get("drop_chapter_title_patterns", []))
        if pattern not in patterns:
            patterns.append(pattern)
        rules["drop_chapter_title_patterns"] = patterns
        _write_rules_payload(repo_root, rules)
        try:
            dropped = sanitize.drop_chapter(
                book_dir=book_dir,
                title=payload.title,
                chapter_index=payload.chapter_index,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _no_store({"status": "ok", "pattern": pattern, "dropped": dropped})

    @app.post("/api/sanitize/restore")
    def sanitize_restore(payload: ChapterAction) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        synth_job = jobs.get(payload.book_id)
        if synth_job and synth_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop TTS before editing.")
        merge_job = merge_jobs.get(payload.book_id)
        if merge_job and merge_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop merge before editing.")
        rules = _load_rules_payload(repo_root)
        pattern = f"^{re.escape(payload.title)}$"
        patterns = list(rules.get("drop_chapter_title_patterns", []))
        if pattern in patterns:
            patterns = [p for p in patterns if p != pattern]
        rules["drop_chapter_title_patterns"] = patterns
        _write_rules_payload(repo_root, rules)
        try:
            restored = sanitize.restore_chapter(
                book_dir=book_dir,
                title=payload.title,
                chapter_index=payload.chapter_index,
                base_dir=repo_root,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _no_store({"status": "ok", "pattern": pattern, "restored": restored})

    @app.post("/api/sanitize/run")
    def sanitize_run(payload: SanitizeRequest) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        synth_job = jobs.get(payload.book_id)
        if synth_job and synth_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop TTS before sanitizing.")
        merge_job = merge_jobs.get(payload.book_id)
        if merge_job and merge_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop merge before sanitizing.")
        try:
            sanitize.sanitize_book(book_dir=book_dir, overwrite=True, base_dir=repo_root)
            tts_cleared = sanitize.refresh_chunks(book_dir=book_dir)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _no_store({"status": "ok", "tts_cleared": tts_cleared})

    @app.post("/api/sanitize/clean")
    def sanitize_clean(payload: CleanEditPayload) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        synth_job = jobs.get(payload.book_id)
        if synth_job and synth_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop TTS before editing text.")
        merge_job = merge_jobs.get(payload.book_id)
        if merge_job and merge_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop merge before editing text.")

        clean_toc = _load_json(book_dir / "clean" / "toc.json")
        if not clean_toc:
            raise HTTPException(status_code=404, detail="Missing clean/toc.json.")
        entry = _find_clean_chapter(clean_toc, payload.chapter_index)
        rel_path = entry.get("path") if entry else None
        if not rel_path:
            raise HTTPException(status_code=404, detail="Chapter not found.")
        clean_path = book_dir / rel_path
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_path.write_text(payload.text.rstrip() + "\n", encoding="utf-8")

        try:
            tts_cleared = sanitize.refresh_chunks(book_dir=book_dir)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _no_store(
            {"status": "ok", "chapter_index": payload.chapter_index, "tts_cleared": tts_cleared}
        )

    @app.get("/api/synth/status")
    def synth_status(book_id: str) -> JSONResponse:
        nonlocal ffmpeg_job, ffmpeg_error
        book_dir = _resolve_book_dir(root_dir, book_id)
        manifest_path = book_dir / "tts" / "manifest.json"
        manifest = _load_json(manifest_path)
        progress = _compute_progress(manifest) if manifest else None
        overall_progress = progress
        tts_status = _load_tts_status(book_dir)
        manifest_created = 0
        if manifest:
            try:
                manifest_created = int(manifest.get("created_unix") or 0)
            except (TypeError, ValueError):
                manifest_created = 0

        job = jobs.get(book_id)
        running = False
        exit_code = None
        mode = "tts"
        sample_chapter = None
        if job:
            exit_code = job.process.poll()
            mode = job.mode or "tts"
            sample_chapter = job.sample_chapter_id
            if exit_code is None:
                running = True
            else:
                job.exit_code = exit_code
                job.ended_at = job.ended_at or time.time()
                if job.log_handle:
                    job.log_handle.close()
                    job.log_handle = None
        if running and job and job.rechunk:
            started_at = int(job.started_at)
            if manifest_created and manifest_created < started_at:
                progress = None

        if manifest and mode == "sample" and sample_chapter:
            progress = _compute_chapter_progress(manifest, sample_chapter) or None

        ffmpeg_status = (
            "installed" if shutil.which("ffmpeg") is not None else "missing"
        )
        ffmpeg_log = ""
        if ffmpeg_job:
            ffmpeg_log = str(ffmpeg_job.log_path.relative_to(repo_root))
            ffmpeg_exit = ffmpeg_job.process.poll()
            if ffmpeg_exit is None:
                ffmpeg_status = "installing"
            else:
                ffmpeg_job.exit_code = ffmpeg_exit
                ffmpeg_job.ended_at = ffmpeg_job.ended_at or time.time()
                if ffmpeg_job.log_handle:
                    ffmpeg_job.log_handle.close()
                    ffmpeg_job.log_handle = None
                if ffmpeg_exit != 0:
                    ffmpeg_status = "error"
                    if not ffmpeg_error:
                        ffmpeg_error = (
                            f"ffmpeg install failed. Check {ffmpeg_log}."
                        )
        if ffmpeg_status == "installed":
            ffmpeg_error = None

        log_path = ""
        if job:
            try:
                log_path = str(job.log_path.relative_to(book_dir))
            except ValueError:
                log_path = str(job.log_path)

        payload = {
            "book_id": book_id,
            "running": running,
            "exit_code": exit_code,
            "progress": progress,
            "log_path": log_path,
            "stage": "idle",
            "ffmpeg_status": ffmpeg_status,
            "ffmpeg_error": ffmpeg_error,
            "ffmpeg_log_path": ffmpeg_log,
            "mode": mode,
            "tts_status": tts_status,
            "overall_progress": overall_progress,
        }
        if running:
            status_stage = str(tts_status.get("stage") or "")
            if status_stage == "cloning":
                payload["stage"] = "cloning"
            elif not progress or not progress.get("total"):
                payload["stage"] = "chunking"
            else:
                payload["stage"] = "sampling" if mode == "sample" else "synthesizing"
        elif mode == "sample" and job and job.exit_code == 0:
            payload["stage"] = "sampled"
        elif progress and progress.get("total") and progress.get("done") >= progress.get("total"):
            payload["stage"] = "done"
        return _no_store(payload)

    @app.post("/api/synth/start")
    def synth_start(payload: SynthRequest) -> JSONResponse:
        nonlocal ffmpeg_job, ffmpeg_error
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        if payload.chunk_mode not in ("sentence", "packed"):
            raise HTTPException(status_code=400, detail="Invalid chunk_mode.")

        existing = jobs.get(payload.book_id)
        if existing and existing.process.poll() is None:
            raise HTTPException(status_code=409, detail="TTS is already running.")

        use_voice_map = bool(payload.use_voice_map)
        voice_value = payload.voice
        voice_map_path = None
        if use_voice_map:
            voice_map_path = _voice_map_path(book_dir)
            voice_map = _load_voice_map(book_dir, repo_root)
            voice_value = voice_map.get("default") or voice_value

        try:
            voice_prompt = resolve_voice_prompt(voice_value, base_dir=repo_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if shutil.which("ffmpeg") is None:
            if ffmpeg_job and ffmpeg_job.process.poll() is None:
                pass
            else:
                ffmpeg_error = None
                ffmpeg_job, error = _spawn_ffmpeg_install(repo_root)
                if error:
                    ffmpeg_error = error

        tts_dir = book_dir / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
        if payload.rechunk:
            seg_dir = tts_dir / "segments"
            if seg_dir.exists():
                shutil.rmtree(seg_dir)
        log_path = tts_dir / "synth.log"
        log_handle = log_path.open("w", encoding="utf-8")

        cmd = [
            "uv",
            "run",
            "--with",
            "pocket-tts",
            "ptts",
            "synth",
            "--book",
            str(book_dir),
            "--voice",
            voice_prompt,
            "--max-chars",
            str(payload.max_chars),
            "--pad-ms",
            str(payload.pad_ms),
            "--chunk-mode",
            payload.chunk_mode,
        ]
        if use_voice_map and voice_map_path and voice_map_path.exists():
            cmd += ["--voice-map", str(voice_map_path)]
        if payload.rechunk:
            cmd.append("--rechunk")

        env = os.environ.copy()
        process = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

        jobs[payload.book_id] = SynthJob(
            book_id=payload.book_id,
            book_dir=book_dir,
            process=process,
            started_at=time.time(),
            log_path=log_path,
            voice=voice_prompt,
            max_chars=payload.max_chars,
            pad_ms=payload.pad_ms,
            chunk_mode=payload.chunk_mode,
            rechunk=payload.rechunk,
            mode="tts",
            log_handle=log_handle,
        )

        return _no_store({"status": "started", "book_id": payload.book_id})

    @app.post("/api/synth/sample")
    def synth_sample(payload: SynthRequest) -> JSONResponse:
        nonlocal ffmpeg_job, ffmpeg_error
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        if payload.chunk_mode not in ("sentence", "packed"):
            raise HTTPException(status_code=400, detail="Invalid chunk_mode.")
        if payload.use_voice_map:
            raise HTTPException(
                status_code=400,
                detail="Sample is disabled in Advanced Mode.",
            )

        existing = jobs.get(payload.book_id)
        if existing and existing.process.poll() is None:
            raise HTTPException(status_code=409, detail="TTS is already running.")

        try:
            voice_prompt = resolve_voice_prompt(payload.voice, base_dir=repo_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if shutil.which("ffmpeg") is None:
            if ffmpeg_job and ffmpeg_job.process.poll() is None:
                pass
            else:
                ffmpeg_error = None
                ffmpeg_job, error = _spawn_ffmpeg_install(repo_root)
                if error:
                    ffmpeg_error = error

        tts_dir = book_dir / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
        log_path = tts_dir / "sample.log"
        log_handle = log_path.open("w", encoding="utf-8")
        sample_id, sample_title = _sample_chapter_info(book_dir)
        if not sample_id:
            log_handle.close()
            raise HTTPException(status_code=404, detail="No chapters available to sample.")

        sample_seg_dir = tts_dir / "segments" / sample_id
        if sample_seg_dir.exists():
            shutil.rmtree(sample_seg_dir)

        manifest_path = tts_dir / "manifest.json"
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
                _atomic_write_json(manifest_path, manifest)

        cmd = [
            "uv",
            "run",
            "--with",
            "pocket-tts",
            "ptts",
            "sample",
            "--book",
            str(book_dir),
            "--voice",
            voice_prompt,
            "--max-chars",
            str(payload.max_chars),
            "--pad-ms",
            str(payload.pad_ms),
            "--chunk-mode",
            payload.chunk_mode,
        ]
        if payload.rechunk:
            cmd.append("--rechunk")

        env = os.environ.copy()
        process = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

        jobs[payload.book_id] = SynthJob(
            book_id=payload.book_id,
            book_dir=book_dir,
            process=process,
            started_at=time.time(),
            log_path=log_path,
            voice=voice_prompt,
            max_chars=payload.max_chars,
            pad_ms=payload.pad_ms,
            chunk_mode=payload.chunk_mode,
            rechunk=payload.rechunk,
            mode="sample",
            sample_chapter_id=sample_id,
            sample_chapter_title=sample_title or None,
            log_handle=log_handle,
        )

        return _no_store({"status": "started", "book_id": payload.book_id})

    @app.post("/api/synth/stop")
    def synth_stop(payload: StopRequest) -> JSONResponse:
        job = jobs.get(payload.book_id)
        if not job or job.process.poll() is not None:
            return _no_store({"status": "not_running", "book_id": payload.book_id})

        job.process.terminate()
        try:
            job.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            job.process.kill()
            job.process.wait(timeout=2)

        job.exit_code = job.process.returncode
        job.ended_at = time.time()
        if job.log_handle:
            job.log_handle.close()
            job.log_handle = None

        return _no_store(
            {"status": "stopped", "book_id": payload.book_id, "exit_code": job.exit_code}
        )

    @app.post("/api/tts/clear")
    def clear_tts(payload: ClearRequest) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        synth_job = jobs.get(payload.book_id)
        if synth_job and synth_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop TTS before clearing cache.")
        merge_job = merge_jobs.get(payload.book_id)
        if merge_job and merge_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="Stop merge before clearing cache.")
        tts_dir = book_dir / "tts"
        if tts_dir.exists():
            seg_dir = tts_dir / "segments"
            if seg_dir.exists():
                shutil.rmtree(seg_dir)
            manifest_path = tts_dir / "manifest.json"
            if manifest_path.exists():
                manifest = _load_json(manifest_path)
                chapters = manifest.get("chapters", [])
                changed = False
                if isinstance(chapters, list):
                    for entry in chapters:
                        if not isinstance(entry, dict):
                            continue
                        chunks = entry.get("chunks", [])
                        if not isinstance(chunks, list) or not chunks:
                            continue
                        durations = entry.get("durations_ms")
                        reset = not isinstance(durations, list) or len(durations) != len(chunks)
                        if not reset and any(value is not None for value in durations):
                            reset = True
                        if reset:
                            entry["durations_ms"] = [None] * len(chunks)
                            changed = True
                if changed:
                    _atomic_write_json(manifest_path, manifest)
        return _no_store({"status": "cleared", "book_id": payload.book_id})

    @app.get("/api/merge/status")
    def merge_status(book_id: str) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        output_path = _merge_output_path(book_dir)
        tts_dir = book_dir / "tts"
        progress_path = tts_dir / "merge.progress.json"
        job = merge_jobs.get(book_id)
        running = False
        exit_code = None
        stage = "idle"
        if job:
            exit_code = job.process.poll()
            if exit_code is None:
                running = True
            else:
                job.exit_code = exit_code
                job.ended_at = job.ended_at or time.time()
                if job.log_handle:
                    job.log_handle.close()
                    job.log_handle = None
        log_path = ""
        if job:
            try:
                log_path = str(job.log_path.relative_to(book_dir))
            except ValueError:
                log_path = str(job.log_path)
        if running:
            if job and job.install_ffmpeg and shutil.which("ffmpeg") is None:
                stage = "installing"
            else:
                stage = "merging"
        elif output_path.exists():
            stage = "done"
        elif exit_code is not None and exit_code != 0:
            stage = "failed"
        payload = {
            "book_id": book_id,
            "running": running,
            "exit_code": exit_code,
            "output_path": str(output_path),
            "output_exists": output_path.exists(),
            "log_path": log_path,
            "stage": stage,
            "progress": _load_json(progress_path) if progress_path.exists() else {},
        }
        return _no_store(payload)

    @app.post("/api/merge/start")
    def merge_start(payload: MergeRequest) -> JSONResponse:
        nonlocal ffmpeg_job
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        output_path = _merge_output_path(book_dir)

        existing = merge_jobs.get(payload.book_id)
        if existing and existing.process.poll() is None:
            raise HTTPException(status_code=409, detail="Merge is already running.")

        if not _merge_ready(book_dir):
            raise HTTPException(status_code=409, detail="TTS is not complete.")

        if output_path.exists() and not payload.overwrite:
            raise HTTPException(status_code=409, detail="Output file already exists.")

        if ffmpeg_job and ffmpeg_job.process.poll() is None:
            raise HTTPException(status_code=409, detail="ffmpeg install in progress.")

        tts_dir = book_dir / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
        log_path = tts_dir / "merge.log"
        progress_path = tts_dir / "merge.progress.json"
        if progress_path.exists():
            progress_path.unlink()
        log_handle = log_path.open("w", encoding="utf-8")

        install_ffmpeg = shutil.which("ffmpeg") is None
        try:
            cmd = _build_merge_command(
                book_dir=book_dir,
                output_path=output_path,
                overwrite=payload.overwrite,
                install_ffmpeg=install_ffmpeg,
                progress_path=progress_path,
            )
        except RuntimeError as exc:
            log_handle.write(f"{exc}\n")
            log_handle.close()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        env = os.environ.copy()
        process = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

        merge_jobs[payload.book_id] = MergeJob(
            book_id=payload.book_id,
            book_dir=book_dir,
            process=process,
            started_at=time.time(),
            log_path=log_path,
            output_path=output_path,
            install_ffmpeg=install_ffmpeg,
            progress_path=progress_path,
            log_handle=log_handle,
        )

        return _no_store(
            {
                "status": "started",
                "book_id": payload.book_id,
                "output_path": str(output_path),
            }
        )

    @app.get("/api/m4b/download")
    def download_m4b(book_id: str) -> FileResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        output_path = _merge_output_path(book_dir)
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="M4B not found.")
        return FileResponse(
            path=str(output_path),
            media_type="audio/x-m4b",
            filename=output_path.name,
        )

    return app


def run(root_dir: Path, host: str, port: int) -> None:
    import uvicorn

    app = create_app(root_dir=root_dir)
    uvicorn.run(app, host=host, port=port)
