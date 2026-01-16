from pathlib import Path

from ptts import tts


def test_make_chunks_adds_punctuation() -> None:
    chunks = tts.make_chunks("Hello world", max_chars=50)
    assert chunks == ["Hello world."]


def test_make_chunks_normalizes_abbreviations() -> None:
    chunks = tts.make_chunks("Mr. Poe went home.", max_chars=50)
    assert chunks == ["Mr Poe went home."]


def test_write_chunk_files_creates_files(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "chunks"
    chunks = ["One.", "Two."]
    paths = tts.write_chunk_files(chunks, chunk_dir, overwrite=True)

    assert paths[0].read_text(encoding="utf-8") == "One.\n"
    assert paths[1].read_text(encoding="utf-8") == "Two.\n"


def test_prepare_manifest_writes_chunks(tmp_path: Path) -> None:
    chapter = tts.ChapterInput(
        index=1,
        id="0001-test",
        title="Test",
        text="Hello world",
        path=None,
    )
    out_dir = tmp_path / "out"
    manifest, chunks, pad_ms = tts.prepare_manifest(
        chapters=[chapter],
        out_dir=out_dir,
        voice="voice.wav",
        max_chars=50,
        pad_ms=150,
        chunk_mode="sentence",
        rechunk=False,
    )

    assert pad_ms == 150
    assert chunks == [["Hello world."]]
    assert (out_dir / "chunks" / "0001-test" / "000001.txt").exists()
    assert manifest["chapters"][0]["chunks"] == ["Hello world."]
