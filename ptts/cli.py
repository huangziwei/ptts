from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import epub as epub_util
from . import merge as merge_util
from . import preview as preview_util
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


def _not_implemented(command: str) -> int:
    sys.stderr.write(f"Command not implemented yet: {command}\n")
    return 2


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
    except Exception as exc:
        sys.stderr.write(f"Sanitize failed: {exc}\n")
        return 2

    print(f"Wrote {written} cleaned chapters to {book_dir / 'clean' / 'chapters'}")
    print(f"Report saved to {book_dir / 'clean' / 'report.json'}")
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
        )
    except Exception as exc:
        sys.stderr.write(f"Merge failed: {exc}\n")
        return 2


def _synth(args: argparse.Namespace) -> int:
    book_dir = Path(args.book) if args.book else None
    text_path = Path(args.text) if args.text else None
    out_dir = Path(args.out) if args.out else None

    if book_dir is not None:
        return tts_util.synthesize_book(
            book_dir=book_dir,
            voice=args.voice,
            out_dir=out_dir,
            max_chars=args.max_chars,
            pad_ms=args.pad_ms,
            rechunk=args.rechunk,
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
        rechunk=args.rechunk,
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
    run.add_argument("--voice", required=True, help="Voice prompt: wav path or hf:// URL")
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
    synth.add_argument("--voice", required=True, help="Voice prompt: wav path or hf:// URL")
    synth.add_argument("--max-chars", type=int, default=800)
    synth.add_argument("--pad-ms", type=int, default=150)
    synth.add_argument("--rechunk", action="store_true")
    synth.set_defaults(func=_synth)

    merge = subparsers.add_parser("merge", help="Merge audio into M4B")
    merge.add_argument("--book", required=True, help="Book output directory")
    merge.add_argument("--output", required=True, help="Path to output .m4b")
    merge.add_argument("--bitrate", default="64k", help="Audio bitrate (default: 64k)")
    merge.add_argument(
        "--overwrite", action="store_true", help="Overwrite output if it exists"
    )
    merge.set_defaults(func=_merge)

    preview = subparsers.add_parser(
        "preview", help="Preview chapters in a web UI"
    )
    preview.add_argument("book", help="Book output directory")
    preview.add_argument("--host", default="127.0.0.1")
    preview.add_argument("--port", type=int, default=8001)
    preview.set_defaults(
        func=lambda args: preview_util.run(
            Path(args.book), host=args.host, port=args.port
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
