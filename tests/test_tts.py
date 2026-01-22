from pathlib import Path

from ptts import tts


def test_make_chunks_preserves_text() -> None:
    chunks = tts.make_chunks("Hello world", max_chars=50)
    assert chunks == ["Hello world"]


def test_prepare_tts_text_adds_punctuation() -> None:
    assert tts.prepare_tts_text("Hello world") == "Hello world."


def test_prepare_tts_text_normalizes_abbreviations() -> None:
    assert tts.prepare_tts_text("Mr. Poe went home.") == "Mr Poe went home."


def test_make_chunks_keeps_name_initials_together() -> None:
    text = (
        "We are grateful to Irwin Z. Hoffman for his help."
    )
    chunks = tts.make_chunks(text, max_chars=200)
    assert chunks == [text]


def test_make_chunks_skips_k_initial_split() -> None:
    text = "K. said hello. Then left."
    chunks = tts.make_chunks(text, max_chars=200)
    assert chunks == ["K. said hello.", "Then left."]


def test_make_chunks_skips_us_initial_split() -> None:
    text = "The U.S. Army is here."
    chunks = tts.make_chunks(text, max_chars=200)
    assert chunks == [text]


def test_make_chunks_skips_common_abbrev_split() -> None:
    text = "This is e.g. a test."
    chunks = tts.make_chunks(text, max_chars=200)
    assert chunks == [text]


def test_make_chunks_skips_phd_split() -> None:
    text = "She entered the Ph.D. program."
    chunks = tts.make_chunks(text, max_chars=200)
    assert chunks == [text]


def test_make_chunks_skips_spaced_phd_split() -> None:
    text = "She entered the Ph. D. program."
    chunks = tts.make_chunks(text, max_chars=200)
    assert chunks == [text]


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
    assert chunks == [["Hello world"]]
    assert (out_dir / "chunks" / "0001-test" / "000001.txt").exists()
    assert manifest["chapters"][0]["chunks"] == ["Hello world"]
    assert manifest["chapters"][0]["chunk_spans"] == [[0, 11]]
