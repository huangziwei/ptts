from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _no_store(data: dict) -> JSONResponse:
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


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
            chunks = entry.get("chunks", [])
            if not isinstance(chunks, list):
                chunks = []
            chapters.append(
                {
                    "id": entry.get("id") or "",
                    "title": entry.get("title") or entry.get("id") or "Chapter",
                    "chunks": chunks,
                    "chunk_count": len(chunks),
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


def create_app(root_dir: Path) -> FastAPI:
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    app = FastAPI()
    root_dir = root_dir.resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

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

    return app


def run(root_dir: Path, host: str, port: int) -> None:
    import uvicorn

    app = create_app(root_dir=root_dir)
    uvicorn.run(app, host=host, port=port)
