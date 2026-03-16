from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


def read_clean_text(path: Path) -> str:
    s = path.read_text(encoding="utf-8", errors="strict")
    s = s.replace("\u02bc", "'").replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip() + "\n"


def _extract_markdown_title(lines: Iterable[str]) -> str:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            return heading
        return ""
    return ""


def title_from_filename(path: Path) -> str:
    stem = path.stem.strip()
    if not stem:
        return ""
    stem = re.sub(r"^[0-9]+[-_ ]+", "", stem)
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return stem or path.stem


def guess_title_from_path(path: Path) -> str:
    heading = ""
    try:
        with path.open("r", encoding="utf-8") as handle:
            heading = _extract_markdown_title(handle)
    except (OSError, UnicodeDecodeError):
        heading = ""
    if heading:
        return heading
    fallback = title_from_filename(path)
    return fallback or path.stem or "text"
