from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import epub as epub_util


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
        "sanitize", help="Clean chapter text (not yet implemented)"
    )
    sanitize.add_argument("--book", required=True, help="Book output directory")
    sanitize.set_defaults(func=lambda _args: _not_implemented("sanitize"))

    chunk = subparsers.add_parser("chunk", help="Chunk chapters (not yet implemented)")
    chunk.add_argument("--book", required=True, help="Book output directory")
    chunk.set_defaults(func=lambda _args: _not_implemented("chunk"))

    synth = subparsers.add_parser("synth", help="Synthesize audio (not yet implemented)")
    synth.add_argument("--book", required=True, help="Book output directory")
    synth.add_argument("--voice", required=True, help="Voice prompt: wav path or hf:// URL")
    synth.set_defaults(func=lambda _args: _not_implemented("synth"))

    package = subparsers.add_parser(
        "package", help="Package M4B (not yet implemented)"
    )
    package.add_argument("--book", required=True, help="Book output directory")
    package.add_argument("--output", required=True, help="Path to output .m4b")
    package.set_defaults(func=lambda _args: _not_implemented("package"))

    preview = subparsers.add_parser(
        "preview", help="Preview chapters in a web UI (not yet implemented)"
    )
    preview.add_argument("--book", required=True, help="Book output directory")
    preview.set_defaults(func=lambda _args: _not_implemented("preview"))

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
