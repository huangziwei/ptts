from pathlib import Path

from ptts import tts


def test_make_chunks_preserves_text() -> None:
    chunks = tts.make_chunks("Hello world", max_chars=50)
    assert chunks == ["Hello world"]


def test_prepare_tts_text_adds_punctuation() -> None:
    assert tts.prepare_tts_text("Hello world") == "Hello world."


def test_prepare_tts_text_normalizes_abbreviations() -> None:
    assert tts.prepare_tts_text("Mr. Poe went home.") == "Mr Poe went home."


def test_prepare_tts_text_normalizes_roman_heading() -> None:
    assert tts.prepare_tts_text("Chapter I") == "Chapter one."
    assert tts.prepare_tts_text("Part IV.") == "Part four."


def test_prepare_tts_text_normalizes_roman_title() -> None:
    assert tts.prepare_tts_text("I") == "one."


def test_prepare_tts_text_skips_roman_i_pronoun() -> None:
    text = "In this chapter I continue to investigate."
    assert tts.prepare_tts_text(text) == text


def test_prepare_tts_text_normalizes_roman_heading_with_comma() -> None:
    text = "In chapter I, we continue."
    expected = "In chapter one, we continue."
    assert tts.prepare_tts_text(text) == expected


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


def test_make_chunks_splits_after_quoted_period() -> None:
    text = 'She said "Hello." Then left.'
    chunks = tts.make_chunks(text, max_chars=200)
    assert chunks == ['She said "Hello."', "Then left."]


def test_prepare_tts_text_normalizes_label_numbers() -> None:
    assert (
        tts.prepare_tts_text("Figure 2.1 shows")
        == "Figure two point one shows."
    )


def test_prepare_tts_text_normalizes_large_numbers() -> None:
    assert (
        tts.prepare_tts_text("Population 1,000,000")
        == "Population one million."
    )
    assert (
        tts.prepare_tts_text("Population 1000000")
        == "Population one million."
    )


def test_prepare_tts_text_transliterates_pali_sanskrit() -> None:
    text = "Saṃyukta-āgama, Dīrgha-āgama, Saḷāyatanavibhaṅga-sutta, and Nibbāna."
    expected = "Samyukta-aagama, Diirgha-aagama, Salaayatanavibhangga-sutta, and Nibbaana."
    assert tts.prepare_tts_text(text) == expected


def test_prepare_tts_text_expands_abbreviations() -> None:
    text = "Prof. Smith references Fig. 1.1."
    expected = "Professor Smith references Figure one point one."
    assert tts.prepare_tts_text(text) == expected


def test_prepare_tts_text_expands_etc() -> None:
    text = "Lists and so on, etc."
    expected = "Lists and so on, et cetera."
    assert tts.prepare_tts_text(text) == expected


def test_prepare_tts_text_expands_latin_abbrev() -> None:
    text = "This is, i.e., the key point; e.g. a sample."
    expected = "This is, that is, the key point; for example a sample."
    assert tts.prepare_tts_text(text) == expected


def test_prepare_tts_text_expands_vs_viz() -> None:
    text = "Compare A vs. B; viz. the minimal case."
    expected = "Compare A versus B; namely the minimal case."
    assert tts.prepare_tts_text(text) == expected


def test_prepare_tts_text_strips_double_quotes() -> None:
    text = 'He said "bible" should be read aloud.'
    expected = "He said bible should be read aloud."
    assert tts.prepare_tts_text(text) == expected


def test_prepare_tts_text_strips_single_quotes_but_keeps_apostrophes() -> None:
    text = "He said 'bible' should not change don't."
    expected = "He said bible should not change don't."
    assert tts.prepare_tts_text(text) == expected


def test_prepare_tts_text_keeps_leading_elisions() -> None:
    text = "'Tis the season."
    expected = "'Tis the season."
    assert tts.prepare_tts_text(text) == expected


def test_prepare_tts_text_normalizes_roman_decimal() -> None:
    text = "See I.1 and II.3 in the appendix."
    expected = "See one point one and two point three in the appendix."
    assert tts.prepare_tts_text(text) == expected


def test_make_chunks_skips_us_initial_split() -> None:
    text = "The U.S. Army is here."
    chunks = tts.make_chunks(text, max_chars=200)
    assert chunks == [text]


def test_make_chunks_skips_fig_split() -> None:
    text = "In Fig. 1.1 (below) we see the pattern."
    chunks = tts.make_chunks(text, max_chars=200)
    assert chunks == [text]


def test_make_chunks_skips_multi_initial_split_with_lowercase_following() -> None:
    text = (
        "Little of the present work would have come about without the generous support "
        "that I received from my parents, K. R. and T. F. Steffens."
    )
    chunks = tts.make_chunks(text, max_chars=300)
    assert chunks == [text]


def test_make_chunks_splits_after_etc_sentence() -> None:
    text = "Literally translated, one \"contemplates body in body\", or \"feelings in feelings\", etc. This slightly peculiar expression requires further consideration."
    chunks = tts.make_chunks(text, max_chars=200)
    assert chunks == [
        "Literally translated, one \"contemplates body in body\", or \"feelings in feelings\", etc.",
        "This slightly peculiar expression requires further consideration.",
    ]


def test_make_chunks_skips_ellipsis_split_inside_sentence() -> None:
    text = (
        "It is, in my father's words, \"an inquiry ... and a lamentation,\" "
        "yes, but it aspires to greater things."
    )
    chunks = tts.make_chunks(text, max_chars=300)
    assert chunks == [text]


def test_make_chunks_skips_punct_only_paragraph() -> None:
    text = "First sentence.\n\n'\"\n\nSecond sentence."
    chunks = tts.make_chunks(text, max_chars=200)
    assert chunks == ["First sentence.", "Second sentence."]


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


def test_make_chunks_skips_vol_no_number_split() -> None:
    text = "Vol. 3.7 contains details. No. 5.2 is missing."
    chunks = tts.make_chunks(text, max_chars=200)
    assert chunks == ["Vol. 3.7 contains details.", "No. 5.2 is missing."]


def test_make_chunks_skips_vol_no_roman_letter_split() -> None:
    text = "Vol. IV contains details. No. A is missing."
    chunks = tts.make_chunks(text, max_chars=200)
    assert chunks == ["Vol. IV contains details.", "No. A is missing."]


def test_make_chunks_prefers_clause_punctuation_for_long_sentences() -> None:
    text = "Alice Johnson, Bob Smith, Carol Jones, Dave Brown."
    chunks = tts.make_chunks(text, max_chars=35)
    assert chunks == ["Alice Johnson, Bob Smith,", "Carol Jones, Dave Brown."]


def test_compute_chunk_pause_multipliers_from_break_strength() -> None:
    text = "Chapter Title\n\n\n\n\nFirst section.\n\n\nSecond section."
    spans = tts.make_chunk_spans(text, max_chars=200)
    assert [text[start:end] for start, end in spans] == [
        "Chapter Title",
        "First section.",
        "Second section.",
    ]
    assert tts.compute_chunk_pause_multipliers(text, spans) == [5, 3, 1]


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
        pad_ms=300,
        chunk_mode="sentence",
        rechunk=False,
    )

    assert pad_ms == 300
    assert chunks == [["Hello world"]]
    assert (out_dir / "chunks" / "0001-test" / "000001.txt").exists()
    assert manifest["chapters"][0]["chunks"] == ["Hello world"]
    assert manifest["chapters"][0]["chunk_spans"] == [[0, 11]]
    assert manifest["chapters"][0]["pause_multipliers"] == [1]
