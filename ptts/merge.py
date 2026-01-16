from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


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


def merge_book(
    book_dir: Path,
    output_path: Path,
    bitrate: str = "64k",
    overwrite: bool = False,
) -> int:
    tts_dir = book_dir / "tts"
    concat_path = tts_dir / "concat.txt"
    chapters_path = tts_dir / "chapters.ffmeta"

    if not concat_path.exists():
        raise FileNotFoundError(f"Missing concat file: {concat_path}")
    if not chapters_path.exists():
        raise FileNotFoundError(f"Missing chapter metadata: {chapters_path}")

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
