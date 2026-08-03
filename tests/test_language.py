import pytest

from neb import language


def test_normalize_defaults_when_missing() -> None:
    assert language.normalize_language_tag(None) == "english"
    assert language.normalize_language_tag("") == "english"
    assert language.normalize_language_tag("   ") == "english"


def test_normalize_accepts_pocket_tts_names() -> None:
    for name in language.POCKET_TTS_LANGUAGES:
        assert language.normalize_language_tag(name) == name
        assert language.normalize_language_tag(name.upper()) == name


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("en", "english"),
        ("eng", "english"),
        ("en-US", "english"),
        ("en_GB", "english"),
        ("de", "german"),
        ("de-DE", "german"),
        ("deu", "german"),
        ("fr-CA", "french"),
        ("pt-BR", "portuguese"),
        ("es-419", "spanish"),
        ("it", "italian"),
    ],
)
def test_normalize_maps_iso_codes(tag: str, expected: str) -> None:
    assert language.normalize_language_tag(tag) == expected


def test_normalize_rejects_unsupported_language() -> None:
    with pytest.raises(language.UnsupportedLanguageError):
        language.normalize_language_tag("ja")
    with pytest.raises(language.UnsupportedLanguageError):
        language.normalize_language_tag("zh-Hans")


def test_display_name_capitalizes() -> None:
    assert language.display_name("german") == "German"
    assert language.display_name("english") == "English"
    assert language.display_name("") == ""
    assert language.display_name(None) == ""


def test_resolve_language_normalizes_locale_tags() -> None:
    assert language.resolve_language("en-GB") == "english"
    assert language.resolve_language("de-DE") == "german"
    assert language.resolve_language("fr_CA") == "french"


def test_resolve_language_falls_back_for_unsupported() -> None:
    assert language.resolve_language("ja") == "english"
    assert language.resolve_language(None) == "english"
    assert language.resolve_language("") == "english"


def test_available_layers_for_each_language() -> None:
    assert language.available_layers("english") == [6]
    assert language.available_layers("french") == [24]
    assert language.available_layers("german") == [6, 24]
    assert language.available_layers("italian") == [6, 24]
    assert language.available_layers("portuguese") == [6, 24]
    assert language.available_layers("spanish") == [6, 24]


def test_available_layers_accepts_iso_tag() -> None:
    assert language.available_layers("de-DE") == [6, 24]
    assert language.available_layers("fr-CA") == [24]


def test_default_layers_matches_upstream_recommendations() -> None:
    assert language.default_layers("english") == 6
    assert language.default_layers("french") == 24
    assert language.default_layers("german") == 24
    assert language.default_layers("italian") == 6
    assert language.default_layers("portuguese") == 6
    assert language.default_layers("spanish") == 24


def test_resolve_layers_accepts_supported_values() -> None:
    assert language.resolve_layers("german", 6) == 6
    assert language.resolve_layers("german", 24) == 24
    assert language.resolve_layers("italian", 6) == 6
    assert language.resolve_layers("italian", 24) == 24


def test_resolve_layers_falls_back_when_unavailable() -> None:
    assert language.resolve_layers("english", 24) == 6
    assert language.resolve_layers("french", 6) == 24
    assert language.resolve_layers("german", None) == 24
    assert language.resolve_layers("english", 99) == 6


def test_resolve_model_id_for_known_pairs() -> None:
    assert language.resolve_model_id("english", 6) == "english"
    assert language.resolve_model_id("german", 6) == "german"
    assert language.resolve_model_id("german", 24) == "german_24l"
    assert language.resolve_model_id("french", 24) == "french_24l"
    assert language.resolve_model_id("spanish", 24) == "spanish_24l"


def test_resolve_model_id_falls_back_to_default_layers() -> None:
    assert language.resolve_model_id("english", 24) == "english"
    assert language.resolve_model_id("french", 6) == "french_24l"
    assert language.resolve_model_id("german", None) == "german_24l"


def test_resolve_model_id_accepts_iso_tag() -> None:
    assert language.resolve_model_id("de-DE", 6) == "german"
    assert language.resolve_model_id("pt-BR", 24) == "portuguese_24l"


def test_resolve_maneko_language_maps_english_to_dated_stem() -> None:
    # English is the one special case: maneko's validated default checkpoint.
    assert language.resolve_maneko_language("english", 6) == "english_2026-04"
    assert language.resolve_maneko_language("en-US", None) == "english_2026-04"
    assert language.resolve_maneko_language("english", 24) == "english_2026-04"


def test_resolve_maneko_language_is_identity_for_other_stems() -> None:
    assert language.resolve_maneko_language("german", 6) == "german"
    assert language.resolve_maneko_language("german", 24) == "german_24l"
    assert language.resolve_maneko_language("french", 24) == "french_24l"
    assert language.resolve_maneko_language("italian", 6) == "italian"
    assert language.resolve_maneko_language("spanish", 24) == "spanish_24l"
    assert language.resolve_maneko_language("pt-BR", 24) == "portuguese_24l"
