from __future__ import annotations

import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import List, Sequence, Tuple


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH.")


def _build_ffmpeg_cmd(
    concat_path: Path,
    chapters_path: Path,
    output_path: Path,
    bitrate: str,
    overwrite: bool,
    cover_path: Path | None = None,
    progress: bool = False,
) -> list[str]:
    cmd = ["ffmpeg"]
    cmd.append("-y" if overwrite else "-n")
    cmd += ["-f", "concat", "-safe", "0", "-i", str(concat_path)]
    cmd += ["-i", str(chapters_path)]
    if cover_path is not None:
        cmd += ["-i", str(cover_path)]
    cmd += ["-map", "0:a:0", "-map_metadata", "1", "-map_chapters", "1"]
    if cover_path is not None:
        filter_graph = (
            "[2:v]split=2[cover][bg0];"
            "[bg0]scale=if(gt(iw\\,ih)\\,iw\\,ih):if(gt(iw\\,ih)\\,iw\\,ih),"
            "boxblur=20:1[bg];"
            "[cover]scale=if(gt(iw\\,ih)\\,iw\\,ih):if(gt(iw\\,ih)\\,iw\\,ih):"
            "force_original_aspect_ratio=decrease[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
        )
        cmd += [
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-c:v",
            "mjpeg",
            "-disposition:v:0",
            "attached_pic",
            "-metadata:s:v",
            "title=Cover",
            "-metadata:s:v",
            "comment=Cover (front)",
        ]
    if progress:
        cmd += ["-progress", "pipe:1", "-nostats"]
    cmd += ["-c:a", "aac", "-b:a", bitrate, "-ac", "1", str(output_path)]
    return cmd


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object in {path}")
    return data


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _ffmeta_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\\n")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
    )


def _load_book_metadata(book_dir: Path) -> dict:
    for path in (book_dir / "clean" / "toc.json", book_dir / "toc.json"):
        if path.exists():
            data = _load_json(path)
            meta = data.get("metadata")
            if isinstance(meta, dict):
                return meta
    return {}


def _resolve_cover_path(book_dir: Path, metadata: dict) -> Path | None:
    cover = metadata.get("cover")
    if not isinstance(cover, dict):
        return None
    path = cover.get("path")
    if not path:
        return None
    cover_path = (book_dir / path).resolve()
    if cover_path.exists():
        return cover_path
    return None


def _metadata_tags(metadata: dict) -> dict:
    tags: dict[str, str] = {}
    title = str(metadata.get("title") or "").strip()
    if title:
        tags["title"] = title
        tags["album"] = title
    authors = metadata.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    if isinstance(authors, list):
        cleaned = [str(a).strip() for a in authors if str(a).strip()]
        if cleaned:
            author_text = ", ".join(cleaned)
            tags["artist"] = author_text
            tags["album_artist"] = author_text
    year = str(metadata.get("year") or "").strip()
    if year:
        tags["date"] = year
    language = str(metadata.get("language") or "").strip()
    if language:
        tags["language"] = language
    return tags


def _parse_timecode(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) != 3:
        return 0
    hours, minutes, seconds = parts
    try:
        secs = float(seconds)
        total = (int(hours) * 3600.0) + (int(minutes) * 60.0) + secs
    except ValueError:
        return 0
    return int(round(total * 1000.0))


def _write_progress(
    path: Path, stage: str, out_time_ms: int, total_ms: int
) -> None:
    percent = 0.0
    if total_ms > 0:
        percent = min(100.0, (out_time_ms / total_ms) * 100.0)
    payload = {
        "stage": stage,
        "out_time_ms": int(out_time_ms),
        "total_ms": int(total_ms),
        "percent": round(percent, 2),
    }
    _atomic_write_json(path, payload)


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    if rate <= 0:
        return 0
    return int(round(frames * 1000.0 / rate))


def _resolve_chunk_count(entry: dict, chapter_dir: Path) -> int:
    for key in ("durations_ms", "chunks", "chunk_spans"):
        value = entry.get(key)
        if isinstance(value, list) and value:
            return len(value)
    if chapter_dir.exists():
        return len(sorted(chapter_dir.glob("*.wav")))
    return 0


def _build_concat_file(
    segment_paths: List[Path], concat_path: Path, base_dir: Path
) -> None:
    lines = []
    for path in segment_paths:
        rel = path.relative_to(base_dir).as_posix()
        lines.append(f"file '{rel}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_chapters_ffmeta(
    chapters: Sequence[Tuple[str, int]],
    ffmeta_path: Path,
    metadata: dict | None = None,
) -> None:
    out = [";FFMETADATA1"]
    if metadata:
        for key, value in _metadata_tags(metadata).items():
            out.append(f"{key}={_ffmeta_escape(value)}")
    t = 0
    for title, duration in chapters:
        start = t
        end = t + max(int(duration), 1)
        out.append("")
        out.append("[CHAPTER]")
        out.append("TIMEBASE=1/1000")
        out.append(f"START={start}")
        out.append(f"END={end}")
        out.append(f"title={title}")
        t = end
    ffmeta_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _ensure_merge_inputs(tts_dir: Path, metadata: dict) -> tuple[Path, Path, int]:
    concat_path = tts_dir / "concat.txt"
    chapters_path = tts_dir / "chapters.ffmeta"
    manifest_path = tts_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    chapters = manifest.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("manifest.json contains no chapters.")

    seg_root = tts_dir / "segments"
    segment_paths: List[Path] = []
    chapter_meta: List[Tuple[str, int]] = []
    missing: List[Path] = []
    total_ms = 0

    for entry in chapters:
        if not isinstance(entry, dict):
            continue
        chapter_id = entry.get("id") or "chapter"
        title = entry.get("title") or chapter_id
        chapter_dir = seg_root / chapter_id
        chunk_count = _resolve_chunk_count(entry, chapter_dir)
        if chunk_count <= 0:
            raise ValueError(f"No chunks found for chapter: {chapter_id}")

        durations = entry.get("durations_ms")
        durations_list = durations if isinstance(durations, list) else []
        chapter_total_ms = 0

        for idx in range(1, chunk_count + 1):
            seg_path = chapter_dir / f"{idx:06d}.wav"
            if not seg_path.exists():
                missing.append(seg_path)
                continue
            segment_paths.append(seg_path)
            duration_ms = None
            if idx - 1 < len(durations_list):
                candidate = durations_list[idx - 1]
                if isinstance(candidate, (int, float)) and candidate > 0:
                    duration_ms = int(candidate)
            if duration_ms is None:
                duration_ms = _wav_duration_ms(seg_path)
            chapter_total_ms += duration_ms

        chapter_meta.append((title, chapter_total_ms))
        total_ms += chapter_total_ms

    if missing:
        sample = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} segment(s). Sample:\n{sample}"
        )

    _build_concat_file(segment_paths, concat_path, base_dir=tts_dir)
    _build_chapters_ffmeta(chapter_meta, chapters_path, metadata=metadata)
    return concat_path, chapters_path, total_ms


def _run_ffmpeg_with_progress(
    cmd: list[str],
    progress_path: Path,
    total_ms: int,
) -> int:
    _write_progress(progress_path, "merging", 0, total_ms)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )
    out_time_ms = 0
    pending_time_ms = None
    if proc.stdout:
        for line in proc.stdout:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key == "out_time":
                pending_time_ms = _parse_timecode(value)
            elif key == "out_time_ms":
                if pending_time_ms is None:
                    try:
                        pending_time_ms = int(value)
                    except ValueError:
                        pending_time_ms = None
            elif key == "progress":
                if pending_time_ms is not None:
                    if total_ms > 0 and pending_time_ms > total_ms * 100:
                        out_time_ms = int(pending_time_ms / 1000)
                    else:
                        out_time_ms = pending_time_ms
                    pending_time_ms = None
                if value == "end":
                    out_time_ms = total_ms or out_time_ms
                    _write_progress(progress_path, "done", out_time_ms, total_ms)
                elif value == "continue":
                    _write_progress(progress_path, "merging", out_time_ms, total_ms)
    proc.wait()
    if proc.returncode != 0:
        _write_progress(progress_path, "failed", out_time_ms, total_ms)
    return int(proc.returncode)


def merge_book(
    book_dir: Path,
    output_path: Path,
    bitrate: str = "64k",
    overwrite: bool = False,
    progress_path: Path | None = None,
) -> int:
    book_dir = book_dir.resolve()
    output_path = output_path.resolve()
    tts_dir = (book_dir / "tts").resolve()
    metadata = _load_book_metadata(book_dir)
    cover_path = _resolve_cover_path(book_dir, metadata)
    concat_path, chapters_path, total_ms = _ensure_merge_inputs(tts_dir, metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _require_ffmpeg()

    cmd = _build_ffmpeg_cmd(
        concat_path=concat_path,
        chapters_path=chapters_path,
        output_path=output_path,
        bitrate=bitrate,
        overwrite=overwrite,
        cover_path=cover_path,
        progress=progress_path is not None,
    )
    if progress_path is not None:
        return _run_ffmpeg_with_progress(cmd, progress_path, total_ms)
    proc = subprocess.run(cmd, cwd=str(tts_dir))
    return int(proc.returncode)
