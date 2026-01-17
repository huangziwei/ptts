from __future__ import annotations

import json
import shutil
import subprocess
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
) -> list[str]:
    cmd = ["ffmpeg"]
    cmd.append("-y" if overwrite else "-n")
    cmd += ["-f", "concat", "-safe", "0", "-i", str(concat_path)]
    cmd += ["-i", str(chapters_path)]
    cmd += ["-map", "0:a:0", "-map_metadata", "1", "-map_chapters", "1"]
    cmd += ["-c:a", "aac", "-b:a", bitrate, "-ac", "1", str(output_path)]
    return cmd


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object in {path}")
    return data


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
    chapters: Sequence[Tuple[str, int]], ffmeta_path: Path
) -> None:
    out = [";FFMETADATA1"]
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


def _ensure_merge_inputs(tts_dir: Path) -> tuple[Path, Path]:
    concat_path = tts_dir / "concat.txt"
    chapters_path = tts_dir / "chapters.ffmeta"
    if concat_path.exists() and chapters_path.exists():
        return concat_path, chapters_path

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
        total_ms = 0

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
            total_ms += duration_ms

        chapter_meta.append((title, total_ms))

    if missing:
        sample = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} segment(s). Sample:\n{sample}"
        )

    _build_concat_file(segment_paths, concat_path, base_dir=tts_dir)
    _build_chapters_ffmeta(chapter_meta, chapters_path)
    return concat_path, chapters_path


def merge_book(
    book_dir: Path,
    output_path: Path,
    bitrate: str = "64k",
    overwrite: bool = False,
) -> int:
    book_dir = book_dir.resolve()
    output_path = output_path.resolve()
    tts_dir = (book_dir / "tts").resolve()
    concat_path, chapters_path = _ensure_merge_inputs(tts_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _require_ffmpeg()

    cmd = _build_ffmpeg_cmd(
        concat_path=concat_path,
        chapters_path=chapters_path,
        output_path=output_path,
        bitrate=bitrate,
        overwrite=overwrite,
    )
    proc = subprocess.run(cmd, cwd=str(tts_dir))
    return int(proc.returncode)
