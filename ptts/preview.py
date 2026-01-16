from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from . import sanitize
import re


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_clean_toc(book_dir: Path) -> dict:
    return _load_json(book_dir / "clean" / "toc.json")


def _load_raw_toc(book_dir: Path) -> dict:
    return _load_json(book_dir / "toc.json")


def _load_report(book_dir: Path) -> dict:
    return _load_json(book_dir / "clean" / "report.json")


def _rules_to_text(rules: List[str]) -> str:
    return "\n".join(rules)


def _text_to_rules(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line]


def _compile_highlight_patterns(rules: sanitize.Ruleset) -> List:
    return sanitize.compile_patterns(rules.remove_patterns)


def _highlight_ranges(text: str, patterns: List) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            if match.start() == match.end():
                continue
            ranges.append((match.start(), match.end()))
    if not ranges:
        return []
    ranges.sort(key=lambda item: (item[0], item[1]))
    merged: List[Tuple[int, int]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _render_highlight(text: str, ranges: List[Tuple[int, int]]) -> str:
    if not ranges:
        return html.escape(text)
    parts: List[str] = []
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


def _pick_chapter(clean_toc: dict, chapter_id: Optional[str]) -> dict:
    chapters = clean_toc.get("chapters", []) if isinstance(clean_toc, dict) else []
    if not chapters:
        return {}
    if chapter_id:
        for entry in chapters:
            if str(entry.get("index")) == chapter_id:
                return entry
    return chapters[0]


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


def create_app(book_dir: Path, base_dir: Path) -> FastAPI:
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    app = FastAPI()

    def load_rules_file() -> Dict[str, object]:
        rules_path = base_dir / ".codex" / "ptts-rules.json"
        if not rules_path.exists():
            return {
                "replace_defaults": False,
                "drop_chapter_title_patterns": [],
                "section_cutoff_patterns": [],
                "remove_patterns": [],
            }
        data = json.loads(rules_path.read_text(encoding="utf-8"))
        data.setdefault("replace_defaults", False)
        data.setdefault("drop_chapter_title_patterns", [])
        data.setdefault("section_cutoff_patterns", [])
        data.setdefault("remove_patterns", [])
        return data

    def write_rules_file(payload: Dict[str, object]) -> None:
        rules_path = base_dir / ".codex" / "ptts-rules.json"
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        rules_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, chapter: Optional[str] = None) -> HTMLResponse:
        clean_toc = _load_clean_toc(book_dir)
        raw_toc = _load_raw_toc(book_dir)
        report = _load_report(book_dir)
        rules = sanitize.load_rules(None, base_dir)
        selected = _pick_chapter(clean_toc, chapter)

        clean_path = None
        if selected:
            clean_rel = selected.get("path")
            if clean_rel:
                clean_path = book_dir / clean_rel
        raw_path = _resolve_raw_path(book_dir, raw_toc, selected)

        raw_text = _load_chapter_text(raw_path)
        clean_text = _load_chapter_text(clean_path)

        patterns = _compile_highlight_patterns(rules)
        raw_ranges = _highlight_ranges(raw_text, patterns)

        context = {
            "request": request,
            "book_dir": str(book_dir),
            "chapters": clean_toc.get("chapters", []),
            "selected": selected,
            "raw_text": _render_highlight(raw_text, raw_ranges),
            "clean_text": html.escape(clean_text),
            "rules": rules,
            "rules_text": {
                "drop": _rules_to_text(rules.drop_chapter_title_patterns),
                "cutoff": _rules_to_text(rules.section_cutoff_patterns),
                "remove": _rules_to_text(rules.remove_patterns),
            },
            "report": report,
        }
        return templates.TemplateResponse("preview.html", context)

    @app.post("/rules")
    def save_rules(
        drop_patterns: str = Form(""),
        cutoff_patterns: str = Form(""),
        remove_patterns: str = Form(""),
        replace_defaults: Optional[str] = Form(None),
    ) -> RedirectResponse:
        payload = {
            "replace_defaults": bool(replace_defaults),
            "drop_chapter_title_patterns": _text_to_rules(drop_patterns),
            "section_cutoff_patterns": _text_to_rules(cutoff_patterns),
            "remove_patterns": _text_to_rules(remove_patterns),
        }
        write_rules_file(payload)
        return RedirectResponse(url="/", status_code=303)

    @app.post("/sanitize")
    def run_sanitize() -> RedirectResponse:
        sanitize.sanitize_book(book_dir, overwrite=True, base_dir=base_dir)
        return RedirectResponse(url="/", status_code=303)

    @app.post("/drop")
    def drop_chapter(
        title: str = Form(""),
        chapter: Optional[str] = Form(None),
    ) -> RedirectResponse:
        if title:
            payload = load_rules_file()
            patterns = list(payload.get("drop_chapter_title_patterns", []))
            pattern = f"^{re.escape(title)}$"
            if pattern not in patterns:
                patterns.append(pattern)
                payload["drop_chapter_title_patterns"] = patterns
                write_rules_file(payload)
                sanitize.sanitize_book(book_dir, overwrite=True, base_dir=base_dir)
        redirect = f"/?chapter={chapter}" if chapter else "/"
        return RedirectResponse(url=redirect, status_code=303)

    @app.post("/restore")
    def restore_chapter(
        title: str = Form(""),
        chapter: Optional[str] = Form(None),
    ) -> RedirectResponse:
        if title:
            payload = load_rules_file()
            patterns = list(payload.get("drop_chapter_title_patterns", []))
            pattern = f"^{re.escape(title)}$"
            if pattern in patterns:
                patterns = [p for p in patterns if p != pattern]
                payload["drop_chapter_title_patterns"] = patterns
                write_rules_file(payload)
                sanitize.sanitize_book(book_dir, overwrite=True, base_dir=base_dir)
        redirect = f"/?chapter={chapter}" if chapter else "/"
        return RedirectResponse(url=redirect, status_code=303)

    return app


def run(book_dir: Path, host: str, port: int) -> None:
    import uvicorn

    base_dir = Path.cwd()
    app = create_app(book_dir=book_dir, base_dir=base_dir)
    uvicorn.run(app, host=host, port=port)
