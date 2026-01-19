from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from . import epub as epub_util
from . import merge as merge_util
from . import player as player_util
from . import sanitize as sanitize_util
from . import tts as tts_util


def _ingest(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Input EPUB not found: {input_path}\n")
        return 2

    out_dir = Path(args.out)
    raw_dir = out_dir / "raw" / "chapters"

    if raw_dir.exists():
        existing = [p for p in raw_dir.iterdir() if p.is_file()]
        if existing and not args.overwrite:
            sys.stderr.write(
                "Raw chapters already exist. Use --overwrite to regenerate.\n"
            )
            return 2

    raw_dir.mkdir(parents=True, exist_ok=True)

    book = epub_util.read_epub(input_path)
    metadata = epub_util.extract_metadata(book)
    cover = epub_util.extract_cover_image(book)
    if cover:
        cover_path = _write_cover_image(cover, out_dir)
        if cover_path:
            cover_info = metadata.get("cover") or {}
            cover_info.setdefault("id", cover.get("id", ""))
            cover_info.setdefault("href", cover.get("href", ""))
            cover_info["path"] = cover_path.relative_to(out_dir).as_posix()
            cover_info["media_type"] = cover.get("media_type") or cover_info.get(
                "media_type", ""
            )
            metadata["cover"] = cover_info
    chapters = epub_util.extract_chapters(book, prefer_toc=True)

    if not chapters:
        sys.stderr.write("No chapters found in EPUB.\n")
        return 2

    toc_items = []
    for idx, chapter in enumerate(chapters, start=1):
        title = chapter.title or f"Chapter {idx}"
        slug = epub_util.slugify(title)
        filename = f"{idx:04d}-{slug}.txt"
        out_path = raw_dir / filename

        out_path.write_text(chapter.text.rstrip() + "\n", encoding="utf-8")
        toc_items.append(
            {
                "index": idx,
                "title": title,
                "href": chapter.href,
                "source": chapter.source,
                "path": out_path.relative_to(out_dir).as_posix(),
            }
        )

    toc_data = {
        "created_unix": int(time.time()),
        "source_epub": str(input_path),
        "metadata": metadata,
        "chapters": toc_items,
    }

    toc_path = out_dir / "toc.json"
    toc_path.write_text(
        json.dumps(toc_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(toc_items)} chapters to {raw_dir}")
    print(f"TOC metadata saved to {toc_path}")
    return 0


def _cover_extension(media_type: str, href: str) -> str:
    media_type = (media_type or "").lower()
    if media_type == "image/jpeg":
        return ".jpg"
    if media_type == "image/png":
        return ".png"
    if media_type == "image/webp":
        return ".webp"
    if media_type == "image/gif":
        return ".gif"
    if href:
        suffix = Path(href).suffix.lower()
        if suffix:
            return suffix
    return ".jpg"


def _write_cover_image(cover: dict, out_dir: Path) -> Optional[Path]:
    data = cover.get("data")
    if not data:
        return None
    ext = _cover_extension(cover.get("media_type", ""), cover.get("href", ""))
    path = out_dir / f"cover{ext}"
    path.write_bytes(data)
    return path


def _not_implemented(command: str) -> int:
    sys.stderr.write(f"Command not implemented yet: {command}\n")
    return 2


def _find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


def _is_http_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _download_to(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ptts clone"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _coerce_voice_name(raw: Optional[str], source_name: str) -> str:
    value = raw or source_name
    base = Path(value).name.strip()
    if not base:
        return "voice"
    suffix = Path(base).suffix.lower()
    if suffix in {".mp3", ".wav"}:
        base = Path(base).stem
    base = base.strip()
    return base or "voice"


def _build_clone_ffmpeg_cmd(
    input_path: Path,
    output_path: Path,
    start: str,
    duration: str,
) -> list[str]:
    filter_chain = (
        "aresample=24000,"
        "silenceremove=start_periods=1:start_duration=0.20:start_threshold=-40dB,"
        "areverse,silenceremove=start_periods=1:start_duration=0.20:start_threshold=-40dB,areverse,"
        "highpass=f=80,lowpass=f=11900,afftdn=nr=12:nf=-45,"
        "loudnorm=I=-18:TP=-1.5:LRA=11,alimiter=limit=0.98"
    )
    return [
        "ffmpeg",
        "-y",
        "-ss",
        start,
        "-t",
        duration,
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-af",
        filter_chain,
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def _clone(args: argparse.Namespace) -> int:
    source = str(args.source)
    repo_root = _find_repo_root(Path.cwd())
    voices_dir = repo_root / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    start = args.start or "00:00:00"
    duration_value = float(args.duration or 0)
    if duration_value <= 0:
        sys.stderr.write("--duration must be a positive number of seconds.\n")
        return 2
    duration = str(duration_value)

    if shutil.which("ffmpeg") is None:
        sys.stderr.write("ffmpeg not found on PATH.\n")
        return 2

    if _is_http_url(source):
        parsed = urllib.parse.urlparse(source)
        filename = Path(parsed.path).name or "voice.mp3"
        if not Path(filename).suffix:
            filename = f"{filename}.mp3"
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / filename
            try:
                _download_to(source, input_path)
            except Exception as exc:
                sys.stderr.write(f"Download failed: {exc}\n")
                return 2
            output_name = _coerce_voice_name(args.name, Path(filename).stem)
            output_path = voices_dir / f"{output_name}.wav"
            cmd = _build_clone_ffmpeg_cmd(
                input_path, output_path, start, duration
            )
            result = subprocess.run(cmd)
            if result.returncode != 0:
                sys.stderr.write("ffmpeg failed to process the audio.\n")
                return 2
            print(f"Wrote {output_path}")
            return 0

    input_path = Path(source).expanduser()
    if not input_path.exists():
        sys.stderr.write(f"Input file not found: {input_path}\n")
        return 2
    output_name = _coerce_voice_name(args.name, input_path.stem)
    output_path = voices_dir / f"{output_name}.wav"
    cmd = _build_clone_ffmpeg_cmd(input_path, output_path, start, duration)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.stderr.write("ffmpeg failed to process the audio.\n")
        return 2
    print(f"Wrote {output_path}")
    return 0


def _sanitize(args: argparse.Namespace) -> int:
    book_dir = Path(args.book)
    rules_path = Path(args.rules) if args.rules else None
    try:
        written = sanitize_util.sanitize_book(
            book_dir=book_dir,
            rules_path=rules_path,
            overwrite=args.overwrite,
            base_dir=Path.cwd(),
        )
        tts_cleared = sanitize_util.refresh_chunks(book_dir=book_dir)
    except Exception as exc:
        sys.stderr.write(f"Sanitize failed: {exc}\n")
        return 2

    print(f"Wrote {written} cleaned chapters to {book_dir / 'clean' / 'chapters'}")
    print(f"Report saved to {book_dir / 'clean' / 'report.json'}")
    if tts_cleared:
        print("Cleared TTS cache and prepared chunks.")
    else:
        print("Prepared chunks for TTS.")
    return 0


def _merge(args: argparse.Namespace) -> int:
    book_dir = Path(args.book)
    output_path = Path(args.output)
    try:
        return merge_util.merge_book(
            book_dir=book_dir,
            output_path=output_path,
            bitrate=args.bitrate,
            overwrite=args.overwrite,
            progress_path=Path(args.progress_file) if args.progress_file else None,
        )
    except Exception as exc:
        sys.stderr.write(f"Merge failed: {exc}\n")
        return 2


def _synth(args: argparse.Namespace) -> int:
    book_dir = Path(args.book) if args.book else None
    text_path = Path(args.text) if args.text else None
    out_dir = Path(args.out) if args.out else None
    voice_map = Path(args.voice_map) if args.voice_map else None

    if book_dir is not None:
        return tts_util.synthesize_book(
            book_dir=book_dir,
            voice=args.voice,
            out_dir=out_dir,
            max_chars=args.max_chars,
            pad_ms=args.pad_ms,
            chunk_mode=args.chunk_mode,
            rechunk=args.rechunk,
            voice_map_path=voice_map,
        )

    if text_path is None or out_dir is None:
        sys.stderr.write("--text and --out are required when not using --book.\n")
        return 2

    return tts_util.synthesize_text(
        text_path=text_path,
        voice=args.voice,
        out_dir=out_dir,
        max_chars=args.max_chars,
        pad_ms=args.pad_ms,
        chunk_mode=args.chunk_mode,
        rechunk=args.rechunk,
        voice_map_path=voice_map,
    )


def _sample(args: argparse.Namespace) -> int:
    book_dir = Path(args.book)
    out_dir = Path(args.out) if args.out else None
    voice_map = Path(args.voice_map) if args.voice_map else None
    return tts_util.synthesize_book_sample(
        book_dir=book_dir,
        voice=args.voice,
        out_dir=out_dir,
        max_chars=args.max_chars,
        pad_ms=args.pad_ms,
        chunk_mode=args.chunk_mode,
        rechunk=args.rechunk,
        voice_map_path=voice_map,
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ptts")
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest", help="Extract chapters from an EPUB")
    ingest.add_argument("--input", required=True, help="Path to input .epub")
    ingest.add_argument(
        "--out",
        "--output",
        required=True,
        dest="out",
        help="Output book directory (e.g., out/book)",
    )
    ingest.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing raw chapters"
    )
    ingest.set_defaults(func=_ingest)

    run = subparsers.add_parser("run", help="Run the full pipeline (not yet implemented)")
    run.add_argument("--input", required=True, help="Path to input .epub")
    run.add_argument("--output", required=True, help="Path to output .m4b")
    run.add_argument(
        "--voice",
        help="Voice prompt: built-in name, wav path, or hf:// URL",
    )
    run.set_defaults(func=lambda _args: _not_implemented("run"))

    sanitize = subparsers.add_parser(
        "sanitize", help="Clean chapter text"
    )
    sanitize.add_argument("--book", required=True, help="Book output directory")
    sanitize.add_argument(
        "--rules",
        help="Path to JSON rules file (defaults to .codex/ptts-rules.json if present)",
    )
    sanitize.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing cleaned output"
    )
    sanitize.set_defaults(func=_sanitize)

    chunk = subparsers.add_parser("chunk", help="Chunk chapters (not yet implemented)")
    chunk.add_argument("--book", required=True, help="Book output directory")
    chunk.set_defaults(func=lambda _args: _not_implemented("chunk"))

    synth = subparsers.add_parser("synth", help="Synthesize audio")
    synth_group = synth.add_mutually_exclusive_group(required=True)
    synth_group.add_argument("--text", help="Input UTF-8 .txt file")
    synth_group.add_argument("--book", help="Book output directory")
    synth.add_argument(
        "--out",
        help="Output directory (default: <book>/tts when using --book)",
    )
    synth.add_argument(
        "--voice",
        help="Voice prompt: built-in name, wav path, or hf:// URL",
    )
    synth.add_argument(
        "--voice-map",
        help="Path to voice map JSON for per-chapter voices",
    )
    synth.add_argument("--max-chars", type=int, default=800)
    synth.add_argument("--pad-ms", type=int, default=150)
    synth.add_argument(
        "--chunk-mode",
        choices=["sentence", "packed"],
        default="sentence",
        help="Chunking strategy (default: sentence)",
    )
    synth.add_argument("--rechunk", action="store_true")
    synth.set_defaults(func=_synth)

    sample = subparsers.add_parser("sample", help="Generate a voice sample (first chapter)")
    sample.add_argument("--book", required=True, help="Book output directory")
    sample.add_argument(
        "--out",
        help="Output directory (default: <book>/tts)",
    )
    sample.add_argument(
        "--voice",
        help="Voice prompt: built-in name, wav path, or hf:// URL",
    )
    sample.add_argument(
        "--voice-map",
        help="Path to voice map JSON for per-chapter voices",
    )
    sample.add_argument("--max-chars", type=int, default=800)
    sample.add_argument("--pad-ms", type=int, default=150)
    sample.add_argument(
        "--chunk-mode",
        choices=["sentence", "packed"],
        default="sentence",
        help="Chunking strategy (default: sentence)",
    )
    sample.add_argument("--rechunk", action="store_true")
    sample.set_defaults(func=_sample)

    clone = subparsers.add_parser("clone", help="Create a voice sample from audio")
    clone.add_argument("source", help="URL or local path to an .mp3 file")
    clone.add_argument(
        "--name",
        help="Output voice name (default: input filename without extension)",
    )
    clone.add_argument(
        "--start",
        default="00:00:00",
        help="Start timestamp (HH:MM:SS or seconds)",
    )
    clone.add_argument(
        "--duration",
        type=float,
        default=10,
        help="Duration in seconds",
    )
    clone.set_defaults(func=_clone)

    merge = subparsers.add_parser("merge", help="Merge audio into M4B")
    merge.add_argument("--book", required=True, help="Book output directory")
    merge.add_argument("--output", required=True, help="Path to output .m4b")
    merge.add_argument("--bitrate", default="64k", help="Audio bitrate (default: 64k)")
    merge.add_argument(
        "--overwrite", action="store_true", help="Overwrite output if it exists"
    )
    merge.add_argument(
        "--progress-file",
        help="Write merge progress to JSON file",
    )
    merge.set_defaults(func=_merge)

    play = subparsers.add_parser("play", help="Play generated audio in a web UI")
    play.add_argument(
        "--root",
        default="out",
        help="Root folder containing book outputs (default: out)",
    )
    play.add_argument("--host", default="127.0.0.1")
    play.add_argument("--port", type=int, default=8002)
    play.set_defaults(
        func=lambda args: player_util.run(
            Path(args.root), host=args.host, port=args.port
        )
    )

    clean = subparsers.add_parser(
        "clean", help="Remove generated audio artifacts (not yet implemented)"
    )
    clean.add_argument("--book", help="Book output directory")
    clean.add_argument("--all", action="store_true", help="Clean all books under out/")
    clean.set_defaults(func=lambda _args: _not_implemented("clean"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return int(args.func(args))
