import json
import wave
from pathlib import Path

import pytest

from ptts import merge


def _write_wav(path: Path, duration_ms: int = 120, rate: int = 24_000) -> None:
    frames = int(rate * duration_ms / 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * frames)


def _write_manifest(tts_dir: Path, chapters: list[dict]) -> None:
    payload = {"chapters": chapters}
    tts_dir.mkdir(parents=True, exist_ok=True)
    (tts_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_chapter_segments_stops_at_first_missing_chunk(tmp_path: Path) -> None:
    tts_dir = tmp_path / "tts"
    seg_root = tts_dir / "segments"
    _write_manifest(
        tts_dir,
        chapters=[
            {"id": "c1", "title": "Chapter 1", "chunks": ["a", "b", "c"]},
            {"id": "c2", "title": "Chapter 2", "chunks": ["d"]},
        ],
    )
    _write_wav(seg_root / "c1" / "000001.wav", duration_ms=120)
    _write_wav(seg_root / "c1" / "000003.wav", duration_ms=120)
    _write_wav(seg_root / "c2" / "000001.wav", duration_ms=120)

    chapters, total_ms = merge._load_chapter_segments(tts_dir)

    assert len(chapters) == 1
    assert chapters[0]["title"] == "Chapter 1"
    assert [path.name for path in chapters[0]["segments"]] == ["000001.wav"]
    assert total_ms == 120


def test_load_chapter_segments_raises_when_no_prefix_is_available(
    tmp_path: Path,
) -> None:
    tts_dir = tmp_path / "tts"
    seg_root = tts_dir / "segments"
    _write_manifest(
        tts_dir,
        chapters=[{"id": "c1", "title": "Chapter 1", "chunks": ["a", "b"]}],
    )
    _write_wav(seg_root / "c1" / "000002.wav", duration_ms=120)

    with pytest.raises(FileNotFoundError, match="No synthesized segments available"):
        merge._load_chapter_segments(tts_dir)
