from __future__ import annotations

import json
import re
import shutil
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import tts as tts_util
from .text import read_clean_text

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


def _should_preserve_lines(lines: List[str]) -> bool:
    if len(lines) <= 1:
        return False
    for line in lines:
        if line.lstrip().startswith(("—", "-", "–", "•", "*")):
            return True
    non_empty = [line.strip() for line in lines if line.strip()]
    if len(non_empty) >= 3:
        avg_len = sum(len(line) for line in non_empty) / len(non_empty)
        if avg_len < 60:
            return True
    return False


def normalize_text(text: str, unwrap_lines: bool = True) -> str:
    text = text.replace("\u02bc", "'").replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r'(^|[\s“("])\[(?P<l>[A-Za-z])\](?=[a-z])', r"\1\g<l>", text)
    text = re.sub(r"-\n(?=\w)", "-", text)

    if unwrap_lines:
        blocks = re.split(r"\n\s*\n", text)
        normalized_blocks: List[str] = []
        for block in blocks:
            lines = block.split("\n")
            if _should_preserve_lines(lines):
                merged = "\n".join(line.strip() for line in lines if line.strip())
            else:
                merged = " ".join(line.strip() for line in lines if line.strip())
            merged = re.sub(r"[ \t]{2,}", " ", merged).strip()
            if merged:
                normalized_blocks.append(merged)
        text = "\n\n".join(normalized_blocks)
    else:
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


def refresh_chunks(
    book_dir: Path,
    max_chars: int = 800,
    pad_ms: int = 150,
    chunk_mode: str = "sentence",
) -> bool:
    tts_dir = book_dir / "tts"
    tts_cleared = tts_dir.exists()
    if tts_dir.exists():
        shutil.rmtree(tts_dir)
    tts_util.chunk_book(
        book_dir=book_dir,
        out_dir=tts_dir,
        max_chars=max_chars,
        pad_ms=pad_ms,
        chunk_mode=chunk_mode,
        rechunk=True,
    )
    return tts_cleared


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _reindex_clean_entries(entries: List[dict]) -> None:
    for idx, entry in enumerate(entries, start=1):
        entry["index"] = idx


def _match_report_entry(entry: dict, rel_path: str, title: str) -> bool:
    clean_path = str(entry.get("clean_path") or "")
    if clean_path.endswith(rel_path):
        return True
    if Path(clean_path).name == Path(rel_path).name:
        return True
    return (entry.get("title") or "") == title


def _update_report(
    book_dir: Path,
    rel_path: str,
    title: str,
    dropped: bool,
    drop_reason: str = "",
    cutoff_reason: str = "",
    raw_chars: Optional[int] = None,
    clean_chars: Optional[int] = None,
) -> None:
    report_path = book_dir / "clean" / "report.json"
    if not report_path.exists():
        return
    report = _load_json(report_path)
    chapters = report.get("chapters", [])
    if not isinstance(chapters, list):
        return
    target = None
    for entry in chapters:
        if isinstance(entry, dict) and _match_report_entry(entry, rel_path, title):
            target = entry
            break
    if not target:
        return

    target["dropped"] = dropped
    target["drop_reason"] = drop_reason if dropped else ""
    if cutoff_reason:
        target["cutoff_reason"] = cutoff_reason
    if raw_chars is not None:
        target["raw_chars"] = raw_chars
    if clean_chars is not None:
        target["clean_chars"] = clean_chars
    target["clean_path"] = str((book_dir / rel_path).resolve())

    stats = report.get("stats", {}) if isinstance(report.get("stats"), dict) else {}
    stats["dropped_chapters"] = sum(
        1 for entry in chapters if isinstance(entry, dict) and entry.get("dropped")
    )
    stats["total_chapters"] = len(chapters)
    report["stats"] = stats
    _write_json(report_path, report)


def _drop_tts_chapter(book_dir: Path, chapter_id: str) -> None:
    tts_dir = book_dir / "tts"
    if not tts_dir.exists():
        return
    for dir_name in ("chunks", "segments"):
        target = tts_dir / dir_name / chapter_id
        if target.exists():
            shutil.rmtree(target)
    manifest_path = tts_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = _load_json(manifest_path)
    chapters = manifest.get("chapters", [])
    if not isinstance(chapters, list):
        return
    filtered = [
        entry
        for entry in chapters
        if isinstance(entry, dict) and (entry.get("id") or "") != chapter_id
    ]
    if len(filtered) == len(chapters):
        return
    manifest["chapters"] = filtered
    tts_util.atomic_write_json(manifest_path, manifest)


def drop_chapter(
    book_dir: Path,
    title: str,
    chapter_index: Optional[int] = None,
) -> bool:
    clean_toc_path = book_dir / "clean" / "toc.json"
    if not clean_toc_path.exists():
        raise FileNotFoundError(f"Missing clean/toc.json at {clean_toc_path}")
    clean_toc = _load_json(clean_toc_path)
    entries = clean_toc.get("chapters", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("clean/toc.json contains no chapters.")

    target = None
    if chapter_index is not None:
        for entry in entries:
            if entry.get("index") == chapter_index:
                target = entry
                break
    if target is None:
        for entry in entries:
            if (entry.get("title") or "") == title:
                target = entry
                break
    if not target:
        return False
    if target.get("kind") == "title":
        raise ValueError("Title chapter cannot be dropped.")

    rel_path = target.get("path") or ""
    chapter_id = tts_util.chapter_id_from_path(
        int(target.get("index") or 0),
        str(target.get("title") or ""),
        rel_path or None,
    )

    if rel_path:
        clean_path = book_dir / rel_path
        if clean_path.exists():
            clean_path.unlink()

    entries = [entry for entry in entries if entry is not target]
    _reindex_clean_entries(entries)
    clean_toc["chapters"] = entries
    _write_json(clean_toc_path, clean_toc)

    if rel_path:
        _update_report(
            book_dir,
            rel_path=rel_path,
            title=str(target.get("title") or ""),
            dropped=True,
            drop_reason="manual_drop",
            clean_chars=0,
        )
    _drop_tts_chapter(book_dir, chapter_id)
    return True


def restore_chapter(
    book_dir: Path,
    title: str,
    chapter_index: Optional[int] = None,
    base_dir: Optional[Path] = None,
) -> bool:
    toc_path = book_dir / "toc.json"
    if not toc_path.exists():
        raise FileNotFoundError(f"Missing toc.json at {toc_path}")
    toc = _load_json(toc_path)
    chapters = toc.get("chapters", [])
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("toc.json contains no chapters.")

    raw_entry = None
    if chapter_index is not None:
        for entry in chapters:
            if entry.get("index") == chapter_index:
                raw_entry = entry
                break
    if raw_entry is None:
        for entry in chapters:
            if (entry.get("title") or "") == title:
                raw_entry = entry
                break
    if not raw_entry:
        return False

    raw_rel = raw_entry.get("path") or ""
    if not raw_rel:
        raise FileNotFoundError("Missing raw chapter path.")
    raw_path = book_dir / raw_rel
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw chapter at {raw_path}")

    if base_dir is None:
        base_dir = Path.cwd()
    rules = load_rules(None, base_dir)
    drop_patterns = compile_patterns(rules.drop_chapter_title_patterns)
    cutoff_patterns = compile_patterns(rules.section_cutoff_patterns)
    remove_patterns = compile_patterns(rules.remove_patterns)

    raw_title = str(raw_entry.get("title") or "").strip()
    drop_reason = should_drop_title(raw_title, drop_patterns)
    if drop_reason:
        raise ValueError("Chapter title still matches drop rules.")

    raw_text = raw_path.read_text(encoding="utf-8")
    raw_text = normalize_text(raw_text)
    cutoff_text, cutoff_reason = apply_section_cutoff(raw_text, cutoff_patterns)
    cleaned, _stats = apply_remove_patterns(cutoff_text, remove_patterns)
    cleaned = normalize_text(cleaned)

    clean_dir = book_dir / "clean" / "chapters"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_rel = (clean_dir / Path(raw_rel).name).relative_to(book_dir).as_posix()
    clean_path = book_dir / clean_rel
    clean_path.write_text(cleaned + "\n", encoding="utf-8")
    tts_text = read_clean_text(clean_path)

    clean_toc_path = book_dir / "clean" / "toc.json"
    if not clean_toc_path.exists():
        raise FileNotFoundError(f"Missing clean/toc.json at {clean_toc_path}")
    clean_toc = _load_json(clean_toc_path)
    clean_entries = clean_toc.get("chapters", [])
    if not isinstance(clean_entries, list):
        clean_entries = []

    for entry in clean_entries:
        if (entry.get("path") or "") == clean_rel:
            return False

    source_index = raw_entry.get("index")
    new_entry = {
        "index": 0,
        "title": raw_title or "Chapter",
        "path": clean_rel,
        "source_index": source_index,
        "kind": "chapter",
    }

    insert_at = len(clean_entries)
    for idx, entry in enumerate(clean_entries):
        entry_source = entry.get("source_index")
        if entry_source is None:
            continue
        if source_index is not None and entry_source >= source_index:
            insert_at = idx
            break
    clean_entries.insert(insert_at, new_entry)
    _reindex_clean_entries(clean_entries)
    clean_toc["chapters"] = clean_entries
    _write_json(clean_toc_path, clean_toc)

    _update_report(
        book_dir,
        rel_path=clean_rel,
        title=raw_title,
        dropped=False,
        cutoff_reason=cutoff_reason,
        raw_chars=len(raw_text),
        clean_chars=len(cleaned),
    )

    tts_dir = book_dir / "tts"
    manifest_path = tts_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        manifest_chapters = manifest.get("chapters", [])
        if isinstance(manifest_chapters, list):
            chapter_id = tts_util.chapter_id_from_path(
                int(new_entry.get("index") or 0),
                raw_title,
                clean_rel,
            )
            max_chars = int(manifest.get("max_chars") or 800)
            pad_ms = int(manifest.get("pad_ms") or 150)
            chunk_mode = str(manifest.get("chunk_mode") or "sentence")
            spans = tts_util.make_chunk_spans(
                tts_text, max_chars=max_chars, chunk_mode=chunk_mode
            )
            chunks = [tts_text[start:end] for start, end in spans]
            if not chunks:
                raise ValueError("No chunks generated for restored chapter.")
            span_list = [[start, end] for start, end in spans]
            chunk_dir = tts_dir / "chunks" / chapter_id
            tts_util.write_chunk_files(chunks, chunk_dir, overwrite=True)

            restored_entry = {
                "index": new_entry.get("index"),
                "id": chapter_id,
                "title": raw_title or chapter_id,
                "path": clean_rel,
                "text_sha256": tts_util.sha256_str(tts_text),
                "chunks": chunks,
                "chunk_spans": span_list,
                "durations_ms": [None] * len(chunks),
            }

            order_ids = [
                tts_util.chapter_id_from_path(
                    int(entry.get("index") or 0),
                    str(entry.get("title") or ""),
                    str(entry.get("path") or ""),
                )
                for entry in clean_entries
            ]
            positions = {cid: idx for idx, cid in enumerate(order_ids)}
            existing = [
                entry
                for entry in manifest_chapters
                if isinstance(entry, dict)
                and (entry.get("id") or "") in positions
                and (entry.get("id") or "") != chapter_id
            ]
            insert_at = len(existing)
            for idx, entry in enumerate(existing):
                if positions.get(entry.get("id") or "", 0) > positions.get(chapter_id, 0):
                    insert_at = idx
                    break
            existing.insert(insert_at, restored_entry)
            manifest["chapters"] = existing
            manifest["max_chars"] = max_chars
            manifest["pad_ms"] = pad_ms
            manifest["chunk_mode"] = chunk_mode
            tts_util.atomic_write_json(manifest_path, manifest)
    return True
