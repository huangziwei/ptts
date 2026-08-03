"""German-specific lexical normalization for TTS.

Covers the German-only text transforms applied before synthesis:
dotted abbreviations (Dr./Prof./z. B./d. h.), currency (Dollar/Euro/Pfund
without plural -s), era labels (v. Chr./n. Chr./v. u. Z./n. u. Z.),
Roman heading numerals with German labels (Kapitel/Buch/Teil/...),
and number-to-words conversion including years and the tens/ones
reversal of German cardinals (e.g. 41 → einundvierzig).

`normalize_german_lexical` is the orchestrator the dispatcher calls.
"""
from __future__ import annotations

import re
from typing import List, Optional


# ---------------------------------------------------------------------------
# Number → words
# ---------------------------------------------------------------------------

# 0-19 standalone form (1 = "eins").
_WORD_ONES = (
    "null",
    "eins",
    "zwei",
    "drei",
    "vier",
    "fünf",
    "sechs",
    "sieben",
    "acht",
    "neun",
    "zehn",
    "elf",
    "zwölf",
    "dreizehn",
    "vierzehn",
    "fünfzehn",
    "sechzehn",
    "siebzehn",
    "achtzehn",
    "neunzehn",
)

# 0-9 compound form — used as prefix inside a larger compound (1 = "ein",
# as in "einundzwanzig", "einhundert", "eintausend").
_WORD_ONES_COMPOUND = (
    "null",
    "ein",
    "zwei",
    "drei",
    "vier",
    "fünf",
    "sechs",
    "sieben",
    "acht",
    "neun",
)

_WORD_TENS = (
    "",
    "",
    "zwanzig",
    "dreißig",
    "vierzig",
    "fünfzig",
    "sechzig",
    "siebzig",
    "achtzig",
    "neunzig",
)


def _under_100(value: int) -> str:
    """German word for 0 <= value <= 99."""
    if value < 20:
        return _WORD_ONES[value]
    tens, ones = divmod(value, 10)
    if ones == 0:
        return _WORD_TENS[tens]
    ones_word = _WORD_ONES_COMPOUND[ones]
    return f"{ones_word}und{_WORD_TENS[tens]}"


def _under_1000(value: int) -> str:
    """Single compound word for 0 < value < 1000."""
    if value == 0:
        return ""
    hundreds, rest = divmod(value, 100)
    out = ""
    if hundreds:
        out = _WORD_ONES_COMPOUND[hundreds] + "hundert"
    if rest:
        out += _under_100(rest)
    return out


def _compound_prefix(word: str) -> str:
    """Trailing 'eins' becomes 'ein' when followed by another element."""
    if word.endswith("eins"):
        return word[:-4] + "ein"
    return word


def _under_million(value: int) -> str:
    """Single compound word for 0 < value < 1_000_000."""
    if value == 0:
        return ""
    thousands, rest = divmod(value, 1000)
    out = ""
    if thousands:
        out = _compound_prefix(_under_1000(thousands)) + "tausend"
    if rest:
        out += _under_1000(rest)
    return out


def _int_to_words(value: int) -> str:
    """Convert an integer to its German word form."""
    if value < 0:
        return f"minus {_int_to_words(abs(value))}"
    if value == 0:
        return "null"

    parts: List[str] = []
    billions, value = divmod(value, 10**12)
    if billions:
        if billions == 1:
            parts.append("eine Billion")
        else:
            parts.append(f"{_int_to_words(billions)} Billionen")
    milliards, value = divmod(value, 10**9)
    if milliards:
        if milliards == 1:
            parts.append("eine Milliarde")
        else:
            parts.append(f"{_int_to_words(milliards)} Milliarden")
    millions, value = divmod(value, 10**6)
    if millions:
        if millions == 1:
            parts.append("eine Million")
        else:
            parts.append(f"{_int_to_words(millions)} Millionen")
    below = _under_million(value)
    if below:
        parts.append(below)
    return " ".join(parts)


def _digits_to_words(value: str) -> str:
    parts: List[str] = []
    for ch in value:
        if ch.isdigit():
            parts.append(_WORD_ONES[int(ch)])
        else:
            parts.append(ch)
    return " ".join(parts)


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


# ---------------------------------------------------------------------------
# Years
# ---------------------------------------------------------------------------

def _year_to_words(value: int) -> str:
    """Pronounce a year. 1985 → neunzehnhundertfünfundachtzig."""
    if value < 1100 or value > 2099:
        return _int_to_words(value)
    if value < 2000:
        century = value // 100  # 11-19
        suffix = value % 100
        prefix = _under_100(century) + "hundert"
        if suffix == 0:
            return prefix
        return prefix + _under_100(suffix)
    if value == 2000:
        return "zweitausend"
    suffix = value - 2000
    return "zweitausend" + _under_100(suffix)


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


# ---------------------------------------------------------------------------
# Ordinals
# ---------------------------------------------------------------------------

# Irregular single-digit ordinal stems (values 1, 3, 7, 8).
# The rest use "<cardinal>te" (n < 20) or "<cardinal>ste" (n >= 20).
_ORDINAL_IRREGULAR_TRAILING = {
    "eins": "erste",
    "drei": "dritte",
    "sieben": "siebte",
    "acht": "achte",
}


def _int_to_ordinal_words(value: int) -> str:
    if value < 0:
        return f"minus {_int_to_ordinal_words(abs(value))}"
    if value == 0:
        return "nullte"
    cardinal = _int_to_words(value)

    last_unit = value % 100
    if last_unit in (1, 3, 7, 8):
        trailing_cardinal = _WORD_ONES[last_unit]
        ordinal_form = _ORDINAL_IRREGULAR_TRAILING[trailing_cardinal]
        if cardinal.endswith(trailing_cardinal):
            return cardinal[: -len(trailing_cardinal)] + ordinal_form
        # Fall through if no match (shouldn't happen for standard numbers)

    if value < 20:
        return cardinal + "te"
    return cardinal + "ste"


# ---------------------------------------------------------------------------
# Abbreviations
# ---------------------------------------------------------------------------

# Multi-token dotted abbreviations (e.g. "z. B.", "d. h.").
# Processed before single-dot expansions so "z. B." doesn't trip on "B."
# Keys include both spaced and unspaced variants.
_MULTI_DOT_ABBREVS = {
    "z. B.": "zum Beispiel",
    "z.B.": "zum Beispiel",
    "d. h.": "das heißt",
    "d.h.": "das heißt",
    "s. o.": "siehe oben",
    "s.o.": "siehe oben",
    "s. u.": "siehe unten",
    "s.u.": "siehe unten",
    "u. a.": "unter anderem",
    "u.a.": "unter anderem",
    "z. T.": "zum Teil",
    "z.T.": "zum Teil",
    "u. U.": "unter Umständen",
    "u.U.": "unter Umständen",
    "v. a.": "vor allem",
    "v.a.": "vor allem",
    "n. d.": "nach dem",
    "n.d.": "nach dem",
}

# Single-dot abbreviations. Matched case-insensitively, case preserved on
# expansion. Keys are lowercase stems without the trailing period.
_ABBREV_EXPANSIONS = {
    "dr": "Doktor",
    "prof": "Professor",
    "hr": "Herr",
    "hrn": "Herrn",
    "fr": "Frau",
    "nr": "Nummer",
    "bd": "Band",
    "abb": "Abbildung",
    "tab": "Tabelle",
    "kap": "Kapitel",
    "jh": "Jahrhundert",
    "jhdt": "Jahrhundert",
    "bzw": "beziehungsweise",
    "usw": "und so weiter",
    "etc": "et cetera",
    "ca": "circa",
    "ggf": "gegebenenfalls",
    "vgl": "vergleiche",
    "evtl": "eventuell",
    "sog": "sogenannt",
    "inkl": "inklusive",
    "exkl": "exklusive",
    "geb": "geboren",
    "gest": "gestorben",
    "geg": "gegeben",
    "St": "Sankt",
}
_ABBREV_EXPANSION_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, _ABBREV_EXPANSIONS)) + r")\.",
    re.IGNORECASE,
)

# Title-like abbreviations whose dot we drop (so chunker does not split on them).
# These are not expanded — just the period is removed.
_ABBREV_DOT_RE = re.compile(r"\b(Dr|Prof|Hr|Hrn|Fr|St)\.", re.IGNORECASE)

# "Nr." followed by a number → "Nummer"
_NUMBER_ABBREV_RE = re.compile(r"\bNr\.(?=\s*\d)", re.IGNORECASE)

# "S." followed by a number → "Seite" (page reference)
_SEITE_ABBREV_RE = re.compile(r"\bS\.(?=\s+\d)")


def _expand_multi_dot_abbrevs(text: str) -> str:
    # Length-sorted so longer keys ("u. U.") match before shorter ones.
    for abbrev in sorted(_MULTI_DOT_ABBREVS, key=len, reverse=True):
        text = text.replace(abbrev, _MULTI_DOT_ABBREVS[abbrev])
    return text


def _expand_single_abbrevs(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        expansion = _ABBREV_EXPANSIONS.get(token.lower())
        if not expansion:
            return match.group(0)
        if token[0].isupper():
            return expansion[:1].upper() + expansion[1:]
        return expansion.lower()

    return _ABBREV_EXPANSION_RE.sub(replace, text)


def _expand_number_abbrev(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        # Preserve case of the leading N
        if token[0].isupper():
            return "Nummer"
        return "nummer"

    return _NUMBER_ABBREV_RE.sub(replace, text)


def _expand_seite_abbrev(text: str) -> str:
    return _SEITE_ABBREV_RE.sub("Seite", text)


def normalize_abbreviations(text: str) -> str:
    if not text:
        return text
    text = _expand_multi_dot_abbrevs(text)
    text = _expand_number_abbrev(text)
    text = _expand_seite_abbrev(text)
    text = _expand_single_abbrevs(text)
    return _ABBREV_DOT_RE.sub(r"\1", text)


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

# German uses the same noun for singular and plural for most currencies
# (e.g. "1 Euro", "10 Euro" — no final -s).
_CURRENCY_SYMBOL_UNITS = {
    "$": "Dollar",
    "€": "Euro",
    "£": "Pfund",
    "¥": "Yen",
    "₹": "Rupie",
    "₽": "Rubel",
    "₩": "Won",
    "₪": "Schekel",
    "฿": "Baht",
    "₺": "Lira",
    "₱": "Peso",
}
_CURRENCY_SYMBOL_CLASS = "".join(re.escape(symbol) for symbol in _CURRENCY_SYMBOL_UNITS)
# Amount: optional sign, then either "digits(.###)*(,###)?" (so thousands
# groups are always exactly 3 digits) or ",###" (bare decimal). Crucially
# this never ends on a '.', so it won't swallow a sentence-final period.
_CURRENCY_AMOUNT = r"[+-]?(?:\d+(?:\.\d{3})*(?:,\d+)?|,\d+)"
_CURRENCY_PREFIX_RE = re.compile(
    rf"(?<!\w)(?P<sym>[{_CURRENCY_SYMBOL_CLASS}])\s*"
    rf"(?P<amount>{_CURRENCY_AMOUNT})"
)
_CURRENCY_SUFFIX_RE = re.compile(
    rf"(?P<amount>{_CURRENCY_AMOUNT})\s*"
    rf"(?P<sym>[{_CURRENCY_SYMBOL_CLASS}])(?!\w)"
)


def _normalize_currency_amount(amount: str) -> str:
    # German decimal separator is comma, thousands separator is dot.
    # Strip thousands dots, keep/normalize the comma as decimal marker.
    value = amount
    sign = ""
    if value.startswith(("+", "-")):
        sign, value = value[0], value[1:]
    # Remove thousands dots (they come before the comma).
    if "," in value:
        int_part, dec_part = value.split(",", 1)
        int_part = int_part.replace(".", "")
        if dec_part and all(ch == "0" for ch in dec_part):
            return sign + int_part
        return sign + f"{int_part},{dec_part}"
    return sign + value.replace(".", "")


def _normalize_currency_symbols(text: str) -> str:
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        symbol = match.group("sym")
        amount = match.group("amount")
        noun = _CURRENCY_SYMBOL_UNITS.get(symbol)
        if not noun:
            return match.group(0)
        normalized_amount = _normalize_currency_amount(amount)
        return f"{normalized_amount} {noun}"

    text = _CURRENCY_PREFIX_RE.sub(replace, text)
    return _CURRENCY_SUFFIX_RE.sub(replace, text)


# ---------------------------------------------------------------------------
# Numbers: final digit-run normalization
# ---------------------------------------------------------------------------

_NUMBER_LABEL_RE = re.compile(
    r"\b(?P<label>(?:Abb(?:ildung)?|Tab(?:elle)?|Kap(?:itel)?|Teil|Band|Bd|Abschnitt|Nr|Nummer|S(?:eite)?|Gl(?:eichung)?))\.?"
    r"\s+(?P<num>\d+(?:\.\d+)+)\b",
    re.IGNORECASE,
)
_DECIMAL_COMMA_RE = re.compile(r"\b\d+,\d+\b")
_DECIMAL_DOT_RE = re.compile(r"\b\d+(?:\.\d+)+\b")
_GROUPED_INT_DOT_RE = re.compile(r"\b\d{1,3}(?:\.\d{3})+\b")
_PLAIN_INT_RE = re.compile(r"\b\d+\b")
_SIGNED_INT_RE = re.compile(r"(?<!\w)(?P<sign>[+-])(?P<num>\d+)\b")
# Non-year nouns that suppress year pronunciation (e.g. "1500 Euro" should
# stay cardinal, not become "fünfzehnhundert"). These are matched as a full
# word immediately after the number.
_NON_YEAR_UNIT_RE = (
    r"(?:Euro|Dollar|Pfund|Yen|Rupien?|Rubel|Won|Schekel|Baht|Lira|Peso|"
    r"Cent|Prozent|Promille|Millionen?|Milliarden?|Billionen?|"
    r"Stück|Meter|Kilometer|Kilogramm|Gramm|Sekunden?|Minuten?|Stunden?|Tage?)"
)
_YEAR_RANGE_RE = re.compile(
    r"\b(?P<start>1\d{3}|20\d{2})\s*[–-]\s*(?P<end>\d{2,4})\b(?!\s*[–-]\s*\d)"
)
_YEAR_RE = re.compile(
    r"\b(?P<year>1\d{3}|20\d{2})\b(?!\s*%)(?!\s+" + _NON_YEAR_UNIT_RE + r"\b)"
)
_RESIDUAL_DIGITS_RE = re.compile(r"\d+")


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
        return f"{label} {' Punkt '.join(parts)}"

    return _NUMBER_LABEL_RE.sub(replace, text)


def _normalize_grouped_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        stripped = token.replace(".", "")
        try:
            value = int(stripped)
        except ValueError:
            return token
        return _int_to_words(value)

    return _GROUPED_INT_DOT_RE.sub(replace, text)


def _normalize_decimal_commas(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        left, right = token.split(",", 1)
        try:
            left_value = int(left)
        except ValueError:
            return token
        return f"{_int_to_words(left_value)} Komma {_digits_to_words(right)}"

    return _DECIMAL_COMMA_RE.sub(replace, text)


def _normalize_decimal_dots(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        parts = token.split(".")
        if len(parts) == 2:
            left, right = parts
            try:
                left_value = int(left)
            except ValueError:
                return token
            return f"{_int_to_words(left_value)} Punkt {_digits_to_words(right)}"
        spoken = [_number_run_to_words(part) for part in parts]
        return " Punkt ".join(spoken)

    return _DECIMAL_DOT_RE.sub(replace, text)


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
        return f"{_year_to_words(start)} bis {_year_to_words(end)}"

    return _YEAR_RANGE_RE.sub(replace, text)


def _normalize_plain_years(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        try:
            year = int(match.group("year"))
        except ValueError:
            return match.group(0)
        return _year_to_words(year)

    return _YEAR_RE.sub(replace, text)


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


# Ordinal "N." immediately before a known heading noun → ordinal word.
_ORDINAL_HEADING_WORDS = (
    "Kapitel",
    "Kapitels",
    "Buch",
    "Buches",
    "Teil",
    "Teils",
    "Teiles",
    "Band",
    "Bandes",
    "Abschnitt",
    "Abschnitts",
    "Akt",
    "Aktes",
    "Aktes",
    "Szene",
    "Anhang",
    "Anhangs",
    "Auflage",
)
_ORDINAL_BEFORE_HEADING_RE = re.compile(
    r"\b(?P<num>\d{1,4})\.\s+(?=(?:"
    + "|".join(map(re.escape, _ORDINAL_HEADING_WORDS))
    + r")\b)"
)


def _normalize_ordinals(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        try:
            value = int(match.group("num"))
        except ValueError:
            return match.group(0)
        return f"{_int_to_ordinal_words(value)} "

    return _ORDINAL_BEFORE_HEADING_RE.sub(replace, text)


def normalize_numbers_for_tts(text: str) -> str:
    text = _normalize_label_numbers(text)
    text = _normalize_grouped_numbers(text)
    text = _normalize_decimal_commas(text)
    text = _normalize_decimal_dots(text)
    text = _normalize_signed_integers(text)
    text = _normalize_year_ranges(text)
    text = _normalize_plain_years(text)
    text = _normalize_ordinals(text)
    text = _normalize_residual_digits(text)
    return text


# ---------------------------------------------------------------------------
# Era markers
# ---------------------------------------------------------------------------

# Expand before-year and after-year forms. Handles spaced and unspaced dot
# variants ("v. Chr.", "v.Chr.").
_ERA_REPLACEMENTS = (
    (re.compile(r"\bv\s*\.\s*u\s*\.\s*Z\s*\.?", re.IGNORECASE), "vor unserer Zeitrechnung"),
    (re.compile(r"\bn\s*\.\s*u\s*\.\s*Z\s*\.?", re.IGNORECASE), "nach unserer Zeitrechnung"),
    (re.compile(r"\bv\s*\.\s*Chr\s*\.?", re.IGNORECASE), "vor Christus"),
    (re.compile(r"\bn\s*\.\s*Chr\s*\.?", re.IGNORECASE), "nach Christus"),
)


def _normalize_era_abbreviations(text: str) -> str:
    if not text:
        return text
    for pattern, replacement in _ERA_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Roman numerals
# ---------------------------------------------------------------------------

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_CANONICAL_RE = re.compile(
    r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)
_ROMAN_HEADING_RE = re.compile(
    r"\b(?P<label>(?:Kapitel|Buch|Bücher|Teil|Teile|Band|Bände|Abschnitt|Abschnitte|Akt|Akte|Szene|Szenen|Anhang|Anhänge))"
    r"\s+(?P<num>[IVXLCDM]+(?:\s*(?:,\s*(?:und|oder)\s+|,\s*|\s+(?:und|oder|&)\s+)[IVXLCDM]+)*)\b",
    re.IGNORECASE,
)
_ROMAN_LEADING_TITLE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<num>[IVXLCDM]+)(?P<trail>[^A-Za-z0-9\n]*)[ \t]*\n"
    r"(?=[ \t]*[A-ZÄÖÜ])",
    re.IGNORECASE,
)
_ROMAN_STANDALONE_RE = re.compile(
    r"^(?P<num>[IVXLCDM]+)(?P<trail>[^A-Za-z0-9]*)$",
    re.IGNORECASE,
)
_ROMAN_COLON_HEADING_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<num>[IVXLCDM]+)(?P<sep>\s*:\s*)",
    re.IGNORECASE | re.MULTILINE,
)
_ROMAN_TOKEN_RE = re.compile(r"\b[IVXLCDM]+\b", re.IGNORECASE)


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
    def replace_heading(match: re.Match[str]) -> str:
        numbers_text = match.group("num")
        romans = list(_ROMAN_TOKEN_RE.finditer(numbers_text))
        if not romans:
            return match.group(0)
        if len(romans) == 1:
            number = _roman_to_int(romans[0].group(0))
            if number is None:
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

def normalize_german_lexical(text: str) -> str:
    text = normalize_abbreviations(text)
    text = _normalize_era_abbreviations(text)
    text = _normalize_roman_numerals(text)
    text = _normalize_currency_symbols(text)
    text = normalize_numbers_for_tts(text)
    return text
