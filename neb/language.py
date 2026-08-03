"""Language mapping between EPUB dc:language tags and pocket-tts models."""

from __future__ import annotations

from typing import List, Optional, Tuple

POCKET_TTS_LANGUAGES = {
    "english",
    "french",
    "german",
    "italian",
    "portuguese",
    "spanish",
}

DEFAULT_LANGUAGE = "english"

_POCKET_TTS_MODEL_IDS: dict[Tuple[str, int], str] = {
    ("english", 6): "english",
    ("french", 24): "french_24l",
    ("german", 6): "german",
    ("german", 24): "german_24l",
    ("italian", 6): "italian",
    ("italian", 24): "italian_24l",
    ("portuguese", 6): "portuguese",
    ("portuguese", 24): "portuguese_24l",
    ("spanish", 6): "spanish",
    ("spanish", 24): "spanish_24l",
}

_DEFAULT_LAYERS_BY_LANGUAGE: dict[str, int] = {
    "english": 6,
    "french": 24,
    "german": 24,
    "italian": 6,
    "portuguese": 6,
    "spanish": 24,
}

_ISO_TO_POCKET_TTS = {
    "en": "english",
    "eng": "english",
    "fr": "french",
    "fra": "french",
    "fre": "french",
    "de": "german",
    "deu": "german",
    "ger": "german",
    "it": "italian",
    "ita": "italian",
    "pt": "portuguese",
    "por": "portuguese",
    "es": "spanish",
    "spa": "spanish",
}

_DISPLAY_NAMES = {
    "english": "English",
    "french": "French",
    "german": "German",
    "italian": "Italian",
    "portuguese": "Portuguese",
    "spanish": "Spanish",
}


class UnsupportedLanguageError(ValueError):
    """The EPUB declares a language pocket-tts cannot synthesize."""


def normalize_language_tag(tag: Optional[str]) -> str:
    """Map a raw dc:language tag to a pocket-tts language name.

    Empty/missing inputs fall back to DEFAULT_LANGUAGE.
    Raises UnsupportedLanguageError for codes outside pocket-tts' six languages.
    """
    if not tag:
        return DEFAULT_LANGUAGE

    raw = str(tag).strip().lower()
    if not raw:
        return DEFAULT_LANGUAGE

    if raw in POCKET_TTS_LANGUAGES:
        return raw

    primary = raw.replace("_", "-").split("-", 1)[0]
    mapped = _ISO_TO_POCKET_TTS.get(primary)
    if mapped:
        return mapped

    supported = ", ".join(sorted(POCKET_TTS_LANGUAGES))
    raise UnsupportedLanguageError(
        f"Language {tag!r} is not supported by pocket-tts. Supported: {supported}."
    )


def display_name(pocket_tts_language: Optional[str]) -> str:
    """Human-readable name for a pocket-tts language (e.g. 'german' -> 'German')."""
    if not pocket_tts_language:
        return ""
    return _DISPLAY_NAMES.get(
        pocket_tts_language, str(pocket_tts_language).capitalize()
    )


def resolve_language(tag: Optional[str]) -> str:
    """Normalize a raw language tag, falling back to DEFAULT_LANGUAGE if unsupported.

    Use this in display paths (e.g. player) where an unknown code should
    degrade gracefully rather than raise.
    """
    try:
        return normalize_language_tag(tag)
    except UnsupportedLanguageError:
        return DEFAULT_LANGUAGE


def available_layers(language: str) -> List[int]:
    """Layer variants actually bundled with pocket-tts for a given language."""
    lang = resolve_language(language)
    return sorted({layers for (lng, layers) in _POCKET_TTS_MODEL_IDS if lng == lang})


def default_layers(language: str) -> int:
    """Upstream-recommended layer count for a language."""
    lang = resolve_language(language)
    return _DEFAULT_LAYERS_BY_LANGUAGE.get(lang, 6)


def resolve_layers(language: str, layers: Optional[int]) -> int:
    """Coerce a requested layer count to one that pocket-tts supports.

    If `layers` is None or unavailable, falls back to the language's default.
    """
    lang = resolve_language(language)
    options = available_layers(lang)
    if layers in options:
        return int(layers)
    return default_layers(lang)


def resolve_model_id(language: str, layers: Optional[int] = None) -> str:
    """Return the pocket-tts YAML stem for a (language, layers) pair."""
    lang = resolve_language(language)
    lay = resolve_layers(lang, layers)
    model_id = _POCKET_TTS_MODEL_IDS.get((lang, lay))
    if model_id is None:
        model_id = _POCKET_TTS_MODEL_IDS[(lang, default_layers(lang))]
    return model_id


# pocket-tts model id -> maneko config stem. They are identical for every
# language/layer pair except English, where maneko's validated default checkpoint
# is the dated stem `english_2026-04`.
_POCKET_ID_TO_MANEKO_STEM: dict[str, str] = {
    "english": "english_2026-04",
}


def resolve_maneko_language(language: str, layers: Optional[int] = None) -> str:
    """Return the maneko config stem for a (language, layers) pair.

    Built on top of `resolve_model_id`, so it inherits the same ISO-tag handling
    and layer fallbacks. Identity for all stems except `english`.
    """
    model_id = resolve_model_id(language, layers)
    return _POCKET_ID_TO_MANEKO_STEM.get(model_id, model_id)
