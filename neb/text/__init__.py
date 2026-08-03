"""Text I/O and TTS-time normalization.

Layout:
  * `io`       — reading clean text and inferring chapter titles
  * `common`   — language-agnostic helpers (strip brackets/quotes,
                 transliteration, reading overrides, linebreak pauses)
  * `english`  — English lexical normalization
  * `german`   — German lexical normalization

`prepare_tts_text` dispatches to the per-language lexical normalizer after
applying the common pre-processing steps.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional, Sequence

from .common import (
    READING_OVERRIDES_FILENAME,
    _literal_override_pattern,
    _load_reading_overrides,
    _merge_reading_overrides,
    _normalize_linebreak_pauses,
    _normalize_reading_override_entry,
    _parse_reading_entries,
    _parse_reading_entry_line,
    _reading_overrides_path,
    _split_reading_overrides_data,
    _strip_brackets,
    _strip_double_quotes,
    _strip_single_quotes,
    _transliterate_pali_sanskrit,
    apply_reading_overrides,
)
from .english import (
    normalize_abbreviations,
    normalize_english_lexical,
    normalize_numbers_for_tts,
    normalize_urls,
)
from .german import normalize_german_lexical
from .io import guess_title_from_path, read_clean_text, title_from_filename

__all__ = [
    "READING_OVERRIDES_FILENAME",
    "apply_reading_overrides",
    "guess_title_from_path",
    "normalize_abbreviations",
    "normalize_english_lexical",
    "normalize_german_lexical",
    "normalize_numbers_for_tts",
    "normalize_urls",
    "prepare_tts_text",
    "read_clean_text",
    "title_from_filename",
]


_LEXICAL_NORMALIZERS: Dict[str, Callable[[str], str]] = {
    "english": normalize_english_lexical,
    "german": normalize_german_lexical,
}


def prepare_tts_text(
    text: str,
    reading_overrides: Optional[Sequence[Dict[str, Any]]] = None,
    language: str = "english",
) -> str:
    text = _strip_brackets(text)
    text = _strip_double_quotes(text)
    text = _strip_single_quotes(text)
    text = apply_reading_overrides(text, reading_overrides or [])
    text = _transliterate_pali_sanskrit(text)
    # Apply reading overrides twice so users can match either original
    # spellings (with diacritics) or transliterated forms.
    text = apply_reading_overrides(text, reading_overrides or [])
    normalizer = _LEXICAL_NORMALIZERS.get(language, normalize_english_lexical)
    text = normalizer(text)
    text = _normalize_linebreak_pauses(text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text
