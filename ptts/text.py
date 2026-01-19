from __future__ import annotations

import re
from pathlib import Path


def read_clean_text(path: Path) -> str:
    s = path.read_text(encoding="utf-8", errors="strict")
    s = s.replace("\u02bc", "'").replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip() + "\n"
