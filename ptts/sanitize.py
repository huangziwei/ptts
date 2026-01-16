from __future__ import annotations

import json
import re
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

RULE_KEYS = (
    "drop_chapter_title_patterns",
    "section_cutoff_patterns",
    "remove_patterns",
)

DEFAULT_RULES: Dict[str, List[str]] = {
    "drop_chapter_title_patterns": [
        r"^table of contents$",
        r"^contents$",
        r"^copyright$",
        r"^title$",
        r"^title page$",
        r"^endorsements?$",
        r"^discover more$",
        r"^discover your next great read$",
        r"^about (the )?authors?$",
        r"^name index$",
        r"^subject index$",
        r"^references$",
        r"^bibliography$",
        r"^index$",
        r"^notes$",
        r"^endnotes$",
        r"^footnotes$",
    ],
    "section_cutoff_patterns": [
        r"^\s*references\s*$",
        r"^\s*bibliography\s*$",
        r"^\s*notes\s*$",
        r"^\s*endnotes\s*$",
        r"^\s*footnotes\s*$",
    ],
    "remove_patterns": [
        r"\((?:[A-Z][A-Za-z'\-]+(?:\s+et\s+al\.)?|"
        r"[A-Z][A-Za-z'\-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z'\-]+)?),"
        r"\s*(?:19|20)\d{2}[a-z]?(?:,\s*p{1,2}\.\s*\d+(?:-\d+)?)?\)",
        r"\((?:[A-Z][A-Za-z'\-]+(?:\s+et\s+al\.)?|"
        r"[A-Z][A-Za-z'\-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z'\-]+)?)+"
        r"\s+(?:19|20)\d{2}[a-z]?(?:,\s*p{1,2}\.\s*\d+(?:-\d+)?)?\)",
        r"\[[0-9,\s-]+\]",
        r"(?<=\w)\[(\d+)\]",
        r"(?<=\w)\((\d+)\)",
    ],
}


@dataclass(frozen=True)
class Ruleset:
    drop_chapter_title_patterns: List[str]
    section_cutoff_patterns: List[str]
    remove_patterns: List[str]
    source_path: Optional[Path] = None
    replace_defaults: bool = False


@dataclass
class ChapterResult:
    index: int
    title: str
    raw_path: str
    clean_path: str
    dropped: bool
    drop_reason: str
    cutoff_reason: str
    raw_chars: int
    clean_chars: int


def load_rules(
    rules_path: Optional[Path], base_dir: Path
) -> Ruleset:
    rules = deepcopy(DEFAULT_RULES)
    source_path = None
    replace_defaults = False

    if rules_path is None:
        candidate = base_dir / ".codex" / "ptts-rules.json"
        if candidate.exists():
            rules_path = candidate

    if rules_path is not None:
        source_path = rules_path
        data = json.loads(rules_path.read_text(encoding="utf-8"))
        replace_defaults = bool(data.get("replace_defaults", False))
        if replace_defaults:
            rules = {key: [] for key in RULE_KEYS}
        for key in RULE_KEYS:
            if key in data:
                value = data[key]
                if not isinstance(value, list):
                    raise ValueError(f"Rules key '{key}' must be a list.")
                rules[key].extend(str(item) for item in value)

    return Ruleset(
        drop_chapter_title_patterns=rules["drop_chapter_title_patterns"],
        section_cutoff_patterns=rules["section_cutoff_patterns"],
        remove_patterns=rules["remove_patterns"],
        source_path=source_path,
        replace_defaults=replace_defaults,
    )


def format_title_chapter(metadata: dict) -> str:
    raw_title = str(metadata.get("title") or "").strip()
    title = raw_title
    subtitle = ""
    if ":" in raw_title:
        title, subtitle = [part.strip() for part in raw_title.split(":", 1)]
    year = str(metadata.get("year") or "").strip()
    authors = metadata.get("authors") or []

    headline = title or ""
    if subtitle:
        headline = f"{title}: {subtitle}" if title else subtitle
    author_line = ", ".join(a.strip() for a in authors if str(a).strip())
    if author_line:
        author_line = f"by {author_line}"

    if year:
        headline = f"{headline}, {year}" if headline else year

    lines: List[str] = []
    if headline:
        lines.append(headline)
    if author_line:
        if lines:
            lines.append("")
        lines.append(author_line)

    return "\n".join(lines).strip()


def compile_patterns(patterns: Iterable[str]) -> List[re.Pattern]:
    compiled: List[re.Pattern] = []
    for pattern in patterns:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message="Possible nested set.*",
            )
            compiled.append(
                re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE)
            )
    return compiled


def normalize_text(text: str, unwrap_lines: bool = True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r'(^|[\s“("])\[(?P<l>[A-Za-z])\](?=[a-z])', r"\1\g<l>", text)
    if unwrap_lines:
        text = re.sub(r"-\n(?=\w)", "-", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def apply_section_cutoff(
    text: str, patterns: List[re.Pattern]
) -> Tuple[str, str]:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return text[: match.start()].rstrip(), pattern.pattern
    return text, ""


def apply_remove_patterns(
    text: str, patterns: List[re.Pattern]
) -> Tuple[str, Dict[str, int]]:
    stats: Dict[str, int] = {}
    for pattern in patterns:
        text, count = pattern.subn("", text)
        stats[pattern.pattern] = stats.get(pattern.pattern, 0) + count
    return text, stats


def should_drop_title(
    title: str, patterns: List[re.Pattern]
) -> str:
    for pattern in patterns:
        if pattern.search(title):
            return pattern.pattern
    return ""


def sanitize_book(
    book_dir: Path,
    rules_path: Optional[Path] = None,
    overwrite: bool = False,
    base_dir: Optional[Path] = None,
) -> int:
    toc_path = book_dir / "toc.json"
    raw_dir = book_dir / "raw" / "chapters"
    clean_dir = book_dir / "clean" / "chapters"
    report_path = book_dir / "clean" / "report.json"

    if not toc_path.exists():
        raise FileNotFoundError(f"Missing toc.json at {toc_path}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw chapters at {raw_dir}")

    if clean_dir.exists():
        existing = [p for p in clean_dir.iterdir() if p.is_file()]
        if existing and not overwrite:
            raise FileExistsError(
                "Clean chapters already exist. Use --overwrite to regenerate."
            )
        if overwrite:
            for path in existing:
                path.unlink()

    if base_dir is None:
        base_dir = Path.cwd()
    rules = load_rules(rules_path, base_dir)
    drop_patterns = compile_patterns(rules.drop_chapter_title_patterns)
    cutoff_patterns = compile_patterns(rules.section_cutoff_patterns)
    remove_patterns = compile_patterns(rules.remove_patterns)

    toc = json.loads(toc_path.read_text(encoding="utf-8"))
    metadata = toc.get("metadata", {}) if isinstance(toc, dict) else {}
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list):
        raise ValueError("Invalid toc.json: chapters must be a list.")

    clean_dir.mkdir(parents=True, exist_ok=True)
    report_entries: List[ChapterResult] = []
    pattern_stats: Dict[str, int] = {}
    dropped = 0
    clean_entries: List[dict] = []

    title_text = format_title_chapter(metadata)
    if title_text:
        title_path = clean_dir / "0000-title.txt"
        title_path.write_text(title_text + "\n", encoding="utf-8")
        clean_entries.append(
            {
                "index": 1,
                "title": metadata.get("title") or "Title",
                "path": title_path.relative_to(book_dir).as_posix(),
                "source_index": 0,
                "kind": "title",
            }
        )

    for entry in chapters:
        title = str(entry.get("title") or "").strip()
        raw_rel = entry.get("path")
        if not raw_rel:
            continue
        raw_path = book_dir / Path(raw_rel)
        clean_path = clean_dir / Path(raw_rel).name

        drop_reason = should_drop_title(title, drop_patterns)
        if drop_reason:
            dropped += 1
            report_entries.append(
                ChapterResult(
                    index=int(entry.get("index", len(report_entries) + 1)),
                    title=title,
                    raw_path=str(raw_path),
                    clean_path=str(clean_path),
                    dropped=True,
                    drop_reason=drop_reason,
                    cutoff_reason="",
                    raw_chars=0,
                    clean_chars=0,
                )
            )
            continue

        if not raw_path.exists():
            report_entries.append(
                ChapterResult(
                    index=int(entry.get("index", len(report_entries) + 1)),
                    title=title,
                    raw_path=str(raw_path),
                    clean_path=str(clean_path),
                    dropped=True,
                    drop_reason="missing_raw",
                    cutoff_reason="",
                    raw_chars=0,
                    clean_chars=0,
                )
            )
            dropped += 1
            continue

        raw_text = raw_path.read_text(encoding="utf-8")
        raw_text = normalize_text(raw_text)
        cutoff_text, cutoff_reason = apply_section_cutoff(
            raw_text, cutoff_patterns
        )
        cleaned, stats = apply_remove_patterns(cutoff_text, remove_patterns)
        cleaned = normalize_text(cleaned)

        clean_path.write_text(cleaned + "\n", encoding="utf-8")

        for pattern, count in stats.items():
            if count:
                pattern_stats[pattern] = pattern_stats.get(pattern, 0) + count

        report_entries.append(
            ChapterResult(
                index=int(entry.get("index", len(report_entries) + 1)),
                title=title,
                raw_path=str(raw_path),
                clean_path=str(clean_path),
                dropped=False,
                drop_reason="",
                cutoff_reason=cutoff_reason,
                raw_chars=len(raw_text),
                clean_chars=len(cleaned),
            )
        )
        clean_entries.append(
            {
                "index": len(clean_entries) + 1,
                "title": title,
                "path": clean_path.relative_to(book_dir).as_posix(),
                "source_index": entry.get("index", None),
                "kind": "chapter",
            }
        )

    report = {
        "created_unix": int(time.time()),
        "book_dir": str(book_dir),
        "rules_source": str(rules.source_path) if rules.source_path else "",
        "replace_defaults": rules.replace_defaults,
        "rules": {
            "drop_chapter_title_patterns": rules.drop_chapter_title_patterns,
            "section_cutoff_patterns": rules.section_cutoff_patterns,
            "remove_patterns": rules.remove_patterns,
        },
        "stats": {
            "total_chapters": len(report_entries),
            "dropped_chapters": dropped,
            "removed_by_pattern": pattern_stats,
            "added_title_chapter": bool(title_text),
        },
        "title_chapter": {
            "text": title_text,
            "path": clean_entries[0]["path"] if title_text else "",
        },
        "chapters": [
            {
                "index": entry.index,
                "title": entry.title,
                "raw_path": entry.raw_path,
                "clean_path": entry.clean_path,
                "dropped": entry.dropped,
                "drop_reason": entry.drop_reason,
                "cutoff_reason": entry.cutoff_reason,
                "raw_chars": entry.raw_chars,
                "clean_chars": entry.clean_chars,
            }
            for entry in report_entries
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    clean_toc = {
        "created_unix": int(time.time()),
        "source_epub": toc.get("source_epub", ""),
        "metadata": metadata,
        "chapters": clean_entries,
    }
    clean_toc_path = book_dir / "clean" / "toc.json"
    clean_toc_path.write_text(
        json.dumps(clean_toc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return len(clean_entries)
