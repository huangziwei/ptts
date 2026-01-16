from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .text import read_clean_text

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


def _resolve_book_dir(root_dir: Path, book_id: str) -> Path:
    candidate = (root_dir / book_id).resolve()
    if root_dir not in candidate.parents and candidate != root_dir:
        raise HTTPException(status_code=404, detail="Book not found.")
    if not (candidate / "clean" / "toc.json").exists():
        raise HTTPException(status_code=404, detail="Book not found.")
    return candidate


def _book_summary(book_dir: Path) -> dict:
    toc = _load_json(book_dir / "clean" / "toc.json")
    metadata = toc.get("metadata", {}) if isinstance(toc, dict) else {}
    cover = metadata.get("cover") or {}
    cover_path = cover.get("path") or ""
    cover_url = f"/audio/{book_dir.name}/{cover_path}" if cover_path else ""
    chapters = toc.get("chapters", []) if isinstance(toc, dict) else []
    has_audio = (book_dir / "tts" / "manifest.json").exists()
    return {
        "id": book_dir.name,
        "title": metadata.get("title") or book_dir.name,
        "authors": metadata.get("authors") or [],
        "year": metadata.get("year") or "",
        "cover_url": cover_url,
        "has_audio": has_audio,
        "chapter_count": len(chapters) if isinstance(chapters, list) else 0,
    }


def _book_details(book_dir: Path) -> dict:
    toc = _load_json(book_dir / "clean" / "toc.json")
    metadata = toc.get("metadata", {}) if isinstance(toc, dict) else {}
    cover = metadata.get("cover") or {}
    cover_path = cover.get("path") or ""
    cover_url = f"/audio/{book_dir.name}/{cover_path}" if cover_path else ""

    manifest = _load_json(book_dir / "tts" / "manifest.json")
    chapters: List[dict] = []
    if manifest and isinstance(manifest.get("chapters"), list):
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
        },
        "chapters": chapters,
        "audio_base": f"/audio/{book_dir.name}/tts/segments",
    }

def _playback_path(book_dir: Path) -> Path:
    return book_dir / "playback.json"


def _sanitize_playback(data: dict) -> dict:
    last = data.get("last_played")
    if not isinstance(last, int) or last < 0:
        last = None
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
    return {"last_played": last, "bookmarks": bookmarks}


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
    log_handle: Optional[IO[str]] = None
    exit_code: Optional[int] = None
    ended_at: Optional[float] = None


class SynthRequest(BaseModel):
    book_id: str
    voice: str
    max_chars: int = 800
    pad_ms: int = 150
    chunk_mode: str = "sentence"
    rechunk: bool = False


class StopRequest(BaseModel):
    book_id: str


class PlaybackPayload(BaseModel):
    last_played: Optional[int] = None
    bookmarks: List[dict] = []


def create_app(root_dir: Path) -> FastAPI:
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    app = FastAPI()
    root_dir = root_dir.resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    repo_root = _find_repo_root(root_dir)
    jobs: dict[str, SynthJob] = {}

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
            toc_path = child / "clean" / "toc.json"
            if not toc_path.exists():
                continue
            books.append(_book_summary(child))
        return _no_store({"root": str(root_dir), "books": books})

    @app.get("/api/books/{book_id}")
    def get_book(book_id: str) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        return _no_store(_book_details(book_dir))

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

    @app.get("/api/synth/status")
    def synth_status(book_id: str) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, book_id)
        manifest_path = book_dir / "tts" / "manifest.json"
        manifest = _load_json(manifest_path)
        progress = _compute_progress(manifest) if manifest else None

        job = jobs.get(book_id)
        running = False
        exit_code = None
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

        payload = {
            "book_id": book_id,
            "running": running,
            "exit_code": exit_code,
            "progress": progress,
            "log_path": log_path,
            "stage": "idle",
        }
        if running:
            if not progress or not progress.get("total"):
                payload["stage"] = "chunking"
            else:
                payload["stage"] = "synthesizing"
        elif progress and progress.get("total") and progress.get("done") >= progress.get("total"):
            payload["stage"] = "done"
        return _no_store(payload)

    @app.post("/api/synth/start")
    def synth_start(payload: SynthRequest) -> JSONResponse:
        book_dir = _resolve_book_dir(root_dir, payload.book_id)
        if payload.chunk_mode not in ("sentence", "packed"):
            raise HTTPException(status_code=400, detail="Invalid chunk_mode.")

        existing = jobs.get(payload.book_id)
        if existing and existing.process.poll() is None:
            raise HTTPException(status_code=409, detail="TTS is already running.")

        voice_path = Path(payload.voice)
        if not voice_path.is_absolute():
            voice_path = (repo_root / voice_path).resolve()
        if not voice_path.exists():
            raise HTTPException(status_code=400, detail="Voice file not found.")

        tts_dir = book_dir / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
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
            str(voice_path),
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
            voice=str(voice_path),
            max_chars=payload.max_chars,
            pad_ms=payload.pad_ms,
            chunk_mode=payload.chunk_mode,
            rechunk=payload.rechunk,
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

    return app


def run(root_dir: Path, host: str, port: int) -> None:
    import uvicorn

    app = create_app(root_dir=root_dir)
    uvicorn.run(app, host=host, port=port)
