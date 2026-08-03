"""English-specific lexical normalization for TTS.

Covers the English-only text transforms applied before synthesis:
URL spelling-out, dotted abbreviations, Roman numerals, currency symbols,
era labels (BC/AD/BCE/CE), and number-to-words conversion.

`normalize_english_lexical` is the orchestrator the dispatcher calls; the
module-level helpers and constants are exposed so tests and future locale
modules (e.g. German) can share the bits that happen to be universal.
"""
from __future__ import annotations

import re
from typing import List, Optional


# ---------------------------------------------------------------------------
# URL spelling-out
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"(?:https?|ftp)://[^\s<>\"')\]]+",
    re.IGNORECASE,
)
_URL_SPELL_OUT = {"http", "https", "ftp", "www"}
_URL_PUNCT: dict[str, str] = {
    ":": "colon",
    "/": "slash",
    ".": "dot",
    "?": "question mark",
    "=": "equals",
    "&": "ampersand",
    "#": "hash",
    "_": "underscore",
    "-": "dash",
    "@": "at",
    "%": "percent",
    "~": "tilde",
}


def _url_to_spoken(match: re.Match[str]) -> str:
    url = match.group(0)
    trailing = ""
    while url and url[-1] in ".,;:!?":
        trailing = url[-1] + trailing
        url = url[:-1]
    tokens = re.findall(r"[a-zA-Z0-9]+|.", url)
    parts: list[str] = []
    for token in tokens:
        if token in _URL_PUNCT:
            parts.append(_URL_PUNCT[token])
        elif token.lower() in _URL_SPELL_OUT:
            parts.append("-".join(token.lower()))
        else:
            parts.append(token)
    spoken = " ".join(parts)
    return re.sub(r"\s+", " ", spoken).strip() + trailing


def normalize_urls(text: str) -> str:
    if not text:
        return text
    return _URL_RE.sub(_url_to_spoken, text)


# ---------------------------------------------------------------------------
# Abbreviations and initialisms
# ---------------------------------------------------------------------------

_ABBREV_DOT_RE = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Fig|Figs)\.", re.IGNORECASE)
_INITIALS_WITH_NAME_RE = re.compile(
    r"\b(?P<seq>[A-Z]\.(?:\s+[A-Z]\.)+)\s+(?P<name>[A-Z][A-Za-z'’\-]+)\b"
)
_SPACED_DOTTED_INITIALISM_RE = re.compile(r"\b(?P<seq>[A-Z]\.(?:\s+[A-Z]\.)+)")
_COMPACT_DOTTED_INITIALISM_RE = re.compile(r"\b(?P<seq>(?:[A-Z][a-z]?\.){2,})")
_NO_NUMBER_ABBREV_RE = re.compile(r"\bNo\.(?=\s*\d)", re.IGNORECASE)
_PAGE_VERSE_ABBREV_RE = re.compile(
    r"\b(?P<token>p|pp|v|vv)\.(?=\s+(?:\d|[IVXLCDM]+\b))",
    re.IGNORECASE,
)
_ABBREV_EXPANSIONS = {
    "prof": "professor",
    "fig": "figure",
    "figs": "figures",
    "approx": "approximately",
    "ca": "circa",
    "cf": "compare",
    "i.e": "that is",
    "e.g": "for example",
    "etc": "et cetera",
    "et al": "and others",
    "et seq": "and the following",
    "et seqq": "and the following",
    "ibid": "in the same place",
    "loc. cit": "in the place cited",
    "n.b": "note well",
    "op. cit": "in the work cited",
    "q.v": "which see",
    "vs": "versus",
    "viz": "namely",
}
_ABBREV_EXPANSION_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, _ABBREV_EXPANSIONS)) + r")\.",
    re.IGNORECASE,
)


def _expand_abbreviations(text: str) -> str:
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        expansion = _ABBREV_EXPANSIONS.get(token.lower())
        if not expansion:
            return match.group(0)
        if token.isupper():
            return expansion.upper()
        if token[0].isupper():
            return expansion.capitalize()
        return expansion

    return _ABBREV_EXPANSION_RE.sub(replace, text)


def _hyphenate_dotted_letters(seq: str) -> str:
    segments = re.findall(r"[A-Za-z]+(?=\.)", seq)
    letters = [ch.upper() for seg in segments for ch in seg]
    if len(letters) < 2:
        return seq
    return "-".join(letters)


def _normalize_initials_with_name(text: str) -> str:
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        seq = match.group("seq")
        name = match.group("name")
        hyphenated = _hyphenate_dotted_letters(seq)
        if hyphenated == seq:
            return match.group(0)
        return f"{hyphenated}-{name}"

    return _INITIALS_WITH_NAME_RE.sub(replace, text)


def _normalize_dotted_initialisms(text: str) -> str:
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        seq = match.group("seq")
        return _hyphenate_dotted_letters(seq)

    text = _SPACED_DOTTED_INITIALISM_RE.sub(replace, text)
    return _COMPACT_DOTTED_INITIALISM_RE.sub(replace, text)


def _normalize_no_number_abbrev(text: str) -> str:
    if not text:
        return text
    return _NO_NUMBER_ABBREV_RE.sub("number", text)


def _normalize_page_verse_abbrev(text: str) -> str:
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        token = match.group("token").lower()
        if token == "p":
            return "page"
        if token == "pp":
            return "pages"
        if token == "v":
            return "verse"
        return "verses"

    return _PAGE_VERSE_ABBREV_RE.sub(replace, text)


def normalize_abbreviations(text: str) -> str:
    text = _expand_abbreviations(text)
    text = _normalize_initials_with_name(text)
    text = _normalize_dotted_initialisms(text)
    text = _normalize_no_number_abbrev(text)
    text = _normalize_page_verse_abbrev(text)
    return _ABBREV_DOT_RE.sub(r"\1", text)


# ---------------------------------------------------------------------------
# Number → words
# ---------------------------------------------------------------------------

_WORD_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_WORD_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_SCALE_WORDS = (
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
)


def _int_to_words(value: int) -> str:
    if value < 0:
        return f"minus {_int_to_words(abs(value))}"
    if value < 20:
        return _WORD_ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        if ones == 0:
            return _WORD_TENS[tens]
        return f"{_WORD_TENS[tens]} {_WORD_ONES[ones]}"
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        if rest == 0:
            return f"{_WORD_ONES[hundreds]} hundred"
        return f"{_WORD_ONES[hundreds]} hundred {_int_to_words(rest)}"
    for scale, label in _SCALE_WORDS:
        if value >= scale:
            major, rest = divmod(value, scale)
            if rest == 0:
                return f"{_int_to_words(major)} {label}"
            return f"{_int_to_words(major)} {label} {_int_to_words(rest)}"
    return str(value)


def _digits_to_words(value: str) -> str:
    parts: List[str] = []
    for ch in value:
        if ch.isdigit():
            parts.append(_WORD_ONES[int(ch)])
        else:
            parts.append(ch)
    return " ".join(parts)


def _int_to_ordinal_words(value: int) -> str:
    if value < 0:
        return f"minus {_int_to_ordinal_words(abs(value))}"
    if value == 0:
        return "zeroth"

    cardinal = _int_to_words(value)
    parts = cardinal.split()
    if not parts:
        return cardinal

    last = parts[-1]
    irregular = {
        "one": "first",
        "two": "second",
        "three": "third",
        "five": "fifth",
        "eight": "eighth",
        "nine": "ninth",
        "twelve": "twelfth",
    }
    if last in irregular:
        parts[-1] = irregular[last]
    elif last.endswith("y"):
        parts[-1] = f"{last[:-1]}ieth"
    elif last.endswith("e"):
        parts[-1] = f"{last}th"
    else:
        parts[-1] = f"{last}th"
    return " ".join(parts)


def _year_to_words(value: int) -> str:
    if value < 1000 or value > 2099:
        return _int_to_words(value)
    if value < 2000:
        century = value // 100
        suffix = value % 100
        prefix = _int_to_words(century)
        if suffix == 0:
            return f"{prefix} hundred"
        if suffix < 10:
            return f"{prefix} oh {_int_to_words(suffix)}"
        return f"{prefix} {_int_to_words(suffix)}"
    if value == 2000:
        return "two thousand"
    suffix = value - 2000
    if suffix < 10:
        return f"two thousand {_int_to_words(suffix)}"
    return f"twenty {_int_to_words(suffix)}"


def _expand_year_range_end(start: int, end_raw: str) -> Optional[int]:
    token = end_raw.strip()
    if not token.isdigit():
        return None
    if len(token) == 2:
        year = (start // 100) * 100 + int(token)
        if year < start:
            year += 100
        return year
    if len(token) == 4:
        return int(token)
    return None


def _number_run_to_words(token: str) -> str:
    if not token:
        return token
    if len(token) > 1 and token.startswith("0"):
        return _digits_to_words(token)
    try:
        value = int(token)
    except ValueError:
        return _digits_to_words(token)
    return _int_to_words(value)


def _pluralize_spoken_word(word: str) -> str:
    if not word:
        return word
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return f"{word[:-1]}ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return f"{word}es"
    return f"{word}s"


def _pluralize_spoken_phrase(phrase: str) -> str:
    parts = phrase.split()
    if not parts:
        return phrase
    parts[-1] = _pluralize_spoken_word(parts[-1])
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Number regex patterns and per-variant normalizers
# ---------------------------------------------------------------------------

_NUMBER_LABEL_RE = re.compile(
    r"\b(?P<label>(?:fig(?:ure)?|table|chapter|section|part|vol(?:ume)?|no|appendix|eq|equation))\.?"
    r"\s+(?P<num>\d+(?:\.\d+)+)\b",
    re.IGNORECASE,
)
_GROUPED_INT_COMMA_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_GROUPED_INT_DOT_RE = re.compile(r"\b\d{1,3}(?:\.\d{3})+\b")
_DECIMAL_RE = re.compile(r"\b\d+(?:\.\d+)+\b")
_PLAIN_INT_RE = re.compile(r"\b\d+\b")
_SIGNED_INT_RE = re.compile(r"(?<!\w)(?P<sign>[+-])(?P<num>\d+)\b")
_YEAR_RANGE_RE = re.compile(
    r"\b(?P<start>1\d{3}|20\d{2})\s*[–-]\s*(?P<end>\d{2,4})\b(?!\s*[–-]\s*\d)"
)
_YEAR_RE = re.compile(r"\b(?P<year>1\d{3}|20\d{2})\b")
_ORDINAL_RE = re.compile(r"\b(?P<num>\d+)(?P<suffix>st|nd|rd|th)\b", re.IGNORECASE)
_PLURAL_NUMBER_S_RE = re.compile(r"(?<!\w)(?:['’])?(?P<num>\d{1,4})(?:['’])?s\b")
_RESIDUAL_DIGITS_RE = re.compile(r"\d+")
_ROMAN_DECIMAL_RE = re.compile(r"\b(?P<roman>[IVXLCDM]+)\.(?P<num>\d+(?:\.\d+)*)\b")


def _normalize_label_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        label = match.group("label")
        num = match.group("num")
        parts = []
        for piece in num.split("."):
            try:
                value = int(piece)
            except ValueError:
                parts.append(piece)
                continue
            parts.append(_int_to_words(value))
        return f"{label} {' point '.join(parts)}"

    return _NUMBER_LABEL_RE.sub(replace, text)


def _normalize_grouped_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        stripped = token.replace(",", "").replace(".", "")
        try:
            value = int(stripped)
        except ValueError:
            return token
        return _int_to_words(value)

    text = _GROUPED_INT_COMMA_RE.sub(replace, text)
    text = _GROUPED_INT_DOT_RE.sub(replace, text)
    return text


def _normalize_decimal_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        parts = token.split(".")
        if len(parts) == 2:
            left, right = parts
            try:
                left_value = int(left)
            except ValueError:
                return token
            left_words = _int_to_words(left_value)
            right_words = _digits_to_words(right)
            return f"{left_words} point {right_words}"
        spoken = [_number_run_to_words(part) for part in parts]
        return " point ".join(spoken)

    return _DECIMAL_RE.sub(replace, text)


def _normalize_plain_large_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if len(token) < 7:
            return token
        if len(token) > 1 and token.startswith("0"):
            return token
        try:
            value = int(token)
        except ValueError:
            return token
        return _int_to_words(value)

    return _PLAIN_INT_RE.sub(replace, text)


def _normalize_signed_integers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        sign = match.group("sign")
        token = match.group("num")
        try:
            value = int(token)
        except ValueError:
            return match.group(0)
        words = _int_to_words(value)
        if sign == "-":
            return f"minus {words}"
        return f"plus {words}"

    return _SIGNED_INT_RE.sub(replace, text)


def _normalize_year_ranges(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        try:
            start = int(match.group("start"))
        except ValueError:
            return match.group(0)
        end = _expand_year_range_end(start, match.group("end"))
        if end is None or not (1000 <= end <= 2099):
            return match.group(0)
        return f"{_year_to_words(start)} to {_year_to_words(end)}"

    return _YEAR_RANGE_RE.sub(replace, text)


def _normalize_plain_years(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        try:
            year = int(match.group("year"))
        except ValueError:
            return match.group(0)
        return _year_to_words(year)

    return _YEAR_RE.sub(replace, text)


def _normalize_ordinals(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group("num")
        try:
            value = int(token)
        except ValueError:
            return match.group(0)
        return _int_to_ordinal_words(value)

    return _ORDINAL_RE.sub(replace, text)


def _normalize_plural_number_s(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group("num")
        try:
            value = int(token)
        except ValueError:
            return match.group(0)
        if len(token) == 4 and not token.startswith("0") and 1000 <= value <= 2099:
            spoken = _year_to_words(value)
        else:
            spoken = _number_run_to_words(token)
        return _pluralize_spoken_phrase(spoken)

    return _PLURAL_NUMBER_S_RE.sub(replace, text)


def _normalize_residual_digits(text: str) -> str:
    if not text:
        return text
    parts: List[str] = []
    cursor = 0
    text_len = len(text)
    for match in _RESIDUAL_DIGITS_RE.finditer(text):
        start, end = match.span()
        if cursor < start:
            parts.append(text[cursor:start])
        left = text[start - 1] if start > 0 else ""
        right = text[end] if end < text_len else ""
        replacement = _number_run_to_words(match.group(0))
        if left and left.isalpha():
            if not parts or not parts[-1].endswith(" "):
                parts.append(" ")
        parts.append(replacement)
        if right and right.isalpha():
            parts.append(" ")
        cursor = end
    if cursor < text_len:
        parts.append(text[cursor:])
    return "".join(parts)


def _normalize_roman_decimal_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        roman = match.group("roman")
        number = _roman_to_int(roman)
        if number is None:
            return match.group(0)
        parts = [_int_to_words(number)]
        for piece in match.group("num").split("."):
            try:
                value = int(piece)
            except ValueError:
                parts.append(piece)
                continue
            parts.append(_int_to_words(value))
        return " point ".join(parts)

    return _ROMAN_DECIMAL_RE.sub(replace, text)


def normalize_numbers_for_tts(text: str) -> str:
    text = _normalize_roman_decimal_numbers(text)
    text = _normalize_label_numbers(text)
    text = _normalize_grouped_numbers(text)
    text = _normalize_decimal_numbers(text)
    text = _normalize_plain_large_numbers(text)
    text = _normalize_signed_integers(text)
    text = _normalize_plural_number_s(text)
    text = _normalize_year_ranges(text)
    text = _normalize_plain_years(text)
    text = _normalize_ordinals(text)
    text = _normalize_residual_digits(text)
    return text


# ---------------------------------------------------------------------------
# Currency symbols
# ---------------------------------------------------------------------------

_CURRENCY_SYMBOL_UNITS = {
    "$": ("dollar", "dollars"),
    "€": ("euro", "euros"),
    "£": ("pound", "pounds"),
    "¥": ("yen", "yen"),
    "₹": ("rupee", "rupees"),
    "₽": ("ruble", "rubles"),
    "₩": ("won", "won"),
    "₪": ("shekel", "shekels"),
    "₫": ("dong", "dong"),
    "₴": ("hryvnia", "hryvnias"),
    "₦": ("naira", "naira"),
    "฿": ("baht", "baht"),
    "₺": ("lira", "lira"),
    "₱": ("peso", "pesos"),
}
_CURRENCY_SYMBOL_CLASS = "".join(re.escape(symbol) for symbol in _CURRENCY_SYMBOL_UNITS)
_CURRENCY_PREFIX_RE = re.compile(
    rf"(?<!\w)(?P<sym>[{_CURRENCY_SYMBOL_CLASS}])\s*"
    r"(?P<amount>[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))"
)
_CURRENCY_SUFFIX_RE = re.compile(
    r"(?P<amount>[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*"
    rf"(?P<sym>[{_CURRENCY_SYMBOL_CLASS}])(?!\w)"
)


def _is_singular_currency_amount(amount: str) -> bool:
    value = amount.replace(",", "").lstrip("+-")
    if not value:
        return False
    if "." not in value:
        return value == "1"
    left, right = value.split(".", 1)
    if not left:
        left = "0"
    return left == "1" and (not right or all(ch == "0" for ch in right))


def _normalize_currency_amount(amount: str) -> str:
    value = amount.replace(",", "")
    sign = ""
    if value.startswith(("+", "-")):
        sign, value = value[0], value[1:]
    if "." not in value:
        return sign + value
    left, right = value.split(".", 1)
    if right and all(ch == "0" for ch in right):
        return sign + left
    return sign + value


def _normalize_currency_symbols(text: str) -> str:
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        symbol = match.group("sym")
        amount = match.group("amount")
        unit = _CURRENCY_SYMBOL_UNITS.get(symbol)
        if not unit:
            return match.group(0)
        normalized_amount = _normalize_currency_amount(amount)
        singular, plural = unit
        noun = singular if _is_singular_currency_amount(amount) else plural
        return f"{normalized_amount} {noun}"

    text = _CURRENCY_PREFIX_RE.sub(replace, text)
    return _CURRENCY_SUFFIX_RE.sub(replace, text)


# ---------------------------------------------------------------------------
# Era abbreviations (BC/AD/BCE/CE)
# ---------------------------------------------------------------------------

_ERA_DOTTED_REPLACEMENTS = (
    (re.compile(r"\bB\s*\.\s*C\s*\.\s*E\s*\.?", re.IGNORECASE), "B-C-E"),
    (re.compile(r"\bA\s*\.\s*D\s*\.?", re.IGNORECASE), "A-D"),
    (re.compile(r"\bC\s*\.\s*E\s*\.?", re.IGNORECASE), "C-E"),
    (re.compile(r"\bB\s*\.\s*C\s*\.(?!\s*E\s*\.?)", re.IGNORECASE), "B-C"),
)
_ERA_PLAIN_WITH_YEAR_RE = re.compile(
    r"(?P<year>\b\d{1,4}(?:\s*[–-]\s*\d{1,4})?)\s+(?P<era>BCE|CE|BC|AD)\b"
)
_ERA_PLAIN_BEFORE_YEAR_RE = re.compile(
    r"\b(?P<era>BCE|CE|BC|AD)\s+(?P<year>\d{1,4}(?:\s*[–-]\s*\d{1,4})?)\b"
)


def _normalize_era_abbreviations(text: str) -> str:
    if not text:
        return text

    for pattern, replacement in _ERA_DOTTED_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    def hyphenate_era_letters(era: str) -> str:
        return "-".join(ch for ch in era if ch.isalpha())

    def replace_plain_with_year(match: re.Match[str]) -> str:
        year = match.group("year")
        era = match.group("era")
        return f"{year} {hyphenate_era_letters(era)}"

    text = _ERA_PLAIN_WITH_YEAR_RE.sub(replace_plain_with_year, text)

    def replace_plain_before_year(match: re.Match[str]) -> str:
        era = match.group("era")
        year = match.group("year")
        return f"{hyphenate_era_letters(era)} {year}"

    return _ERA_PLAIN_BEFORE_YEAR_RE.sub(replace_plain_before_year, text)


# ---------------------------------------------------------------------------
# Roman numerals
# ---------------------------------------------------------------------------

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_CANONICAL_RE = re.compile(
    r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)
_ROMAN_HEADING_RE = re.compile(
    r"\b(?P<label>(?:chapter(?:s)?|book(?:s)?|part(?:s)?|volume(?:s)?|vol(?:s)?|section(?:s)?|act(?:s)?|scene(?:s)?|appendix(?:es)?|appendices)\.?)"
    r"\s+(?P<num>[IVXLCDM]+(?:\s*(?:,\s*(?:and|or)\s+|,\s*|\s+(?:and|or|&)\s+)[IVXLCDM]+)*)\b",
    re.IGNORECASE,
)
_ROMAN_LEADING_TITLE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<num>[IVXLCDM]+)(?P<trail>[^A-Za-z0-9\n]*)[ \t]*\n"
    r"(?=[ \t]*[A-Z])",
    re.IGNORECASE,
)
_ROMAN_STANDALONE_RE = re.compile(r"^(?P<num>[IVXLCDM]+)(?P<trail>[^A-Za-z0-9]*)$", re.IGNORECASE)
_ROMAN_COLON_HEADING_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<num>[IVXLCDM]+)(?P<sep>\s*:\s*)",
    re.IGNORECASE | re.MULTILINE,
)
_ROMAN_TOKEN_RE = re.compile(r"\b[IVXLCDM]+\b", re.IGNORECASE)
_ROMAN_I_DETERMINERS = {
    "a",
    "an",
    "another",
    "any",
    "each",
    "every",
    "his",
    "her",
    "its",
    "my",
    "no",
    "our",
    "some",
    "that",
    "the",
    "their",
    "this",
    "your",
}
# Mirrors the punctuation-set definitions in common.py. Duplicated here so
# the Roman-numeral trail test stays a local concern.
_ROMAN_HEADING_TRAIL_PUNCT = (
    {".", "!", "?"}
    | {",", ";", ":"}
    | set("\"')]}" + "”’")
    | {"-", "–", "—"}
)


def _roman_to_int(value: str) -> Optional[int]:
    roman = value.upper()
    if not roman or not _ROMAN_CANONICAL_RE.fullmatch(roman):
        return None
    total = 0
    prev = 0
    for ch in reversed(roman):
        number = _ROMAN_VALUES.get(ch)
        if number is None:
            return None
        if number < prev:
            total -= number
        else:
            total += number
            prev = number
    return total or None


def _normalize_roman_numerals(text: str) -> str:
    def prev_word(start: int) -> Optional[str]:
        match = re.search(r"([A-Za-z]+)\s*$", text[:start])
        if not match:
            return None
        return match.group(1)

    def next_word(start: int) -> Optional[str]:
        match = re.search(r"\b([A-Za-z]+)", text[start:])
        if not match:
            return None
        return match.group(1)

    def next_non_space(start: int) -> str:
        match = re.search(r"\S", text[start:])
        if not match:
            return ""
        return match.group(0)

    def should_convert_roman_i(match: re.Match[str]) -> bool:
        label = match.group("label")
        if label and label[0].isupper():
            return True
        prev = prev_word(match.start())
        if prev and prev.lower() in _ROMAN_I_DETERMINERS:
            return False
        next_char = next_non_space(match.end())
        if not next_char:
            return True
        if next_char in _ROMAN_HEADING_TRAIL_PUNCT:
            return True
        next_token = next_word(match.end())
        if next_token and next_token[0].isupper():
            return True
        return False

    def replace_heading(match: re.Match[str]) -> str:
        numbers_text = match.group("num")
        romans = list(_ROMAN_TOKEN_RE.finditer(numbers_text))
        if not romans:
            return match.group(0)
        if len(romans) == 1:
            roman = romans[0].group(0)
            number = _roman_to_int(roman)
            if number is None:
                return match.group(0)
            if roman.upper() == "I" and not should_convert_roman_i(match):
                return match.group(0)
            return f"{match.group('label')} {_int_to_words(number)}"

        parts: List[str] = []
        cursor = 0
        for roman_match in romans:
            start, end = roman_match.span()
            number = _roman_to_int(roman_match.group(0))
            if number is None:
                return match.group(0)
            parts.append(numbers_text[cursor:start])
            parts.append(_int_to_words(number))
            cursor = end
        parts.append(numbers_text[cursor:])
        return f"{match.group('label')} {''.join(parts)}"

    def replace_leading_title(match: re.Match[str]) -> str:
        number = _roman_to_int(match.group("num"))
        if number is None:
            return match.group(0)
        trail = match.group("trail") or ""
        return f"{match.group('indent')}{_int_to_words(number)}{trail}\n"

    def replace_colon_heading(match: re.Match[str]) -> str:
        number = _roman_to_int(match.group("num"))
        if number is None:
            return match.group(0)
        return f"{match.group('indent')}{_int_to_words(number)}{match.group('sep')}"

    text = _ROMAN_LEADING_TITLE_RE.sub(replace_leading_title, text, count=1)
    text = _ROMAN_COLON_HEADING_RE.sub(replace_colon_heading, text)
    text = _ROMAN_HEADING_RE.sub(replace_heading, text)
    stripped = text.strip()
    match = _ROMAN_STANDALONE_RE.fullmatch(stripped)
    if not match:
        return text
    number = _roman_to_int(match.group("num"))
    if number is None:
        return text
    suffix = match.group("trail") or ""
    return f"{_int_to_words(number)}{suffix}"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def normalize_english_lexical(text: str) -> str:
    text = normalize_urls(text)
    text = normalize_abbreviations(text)
    text = _normalize_era_abbreviations(text)
    text = _normalize_roman_numerals(text)
    text = _normalize_currency_symbols(text)
    text = normalize_numbers_for_tts(text)
    return text
