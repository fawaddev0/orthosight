"""Small, source-aware text boundary rules for PDF reconstruction."""

from __future__ import annotations

import re
from collections.abc import Iterable

_SPLIT_WORD_END = re.compile(r"[A-Za-z]-$")
_CONNECTOR_END = re.compile(r"\b(?:and|or|the|of|to|with|for|in|that|this|a|an)$", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?][)\u201d\u2019\"]*(?=\s|$)")
_BOX_LABEL = re.compile(r"BOX\s+\d+(?:\.\d+)?", re.IGNORECASE)


def join_lines(lines: list[str], compound_prefixes: Iterable[str] | None = None) -> str:
    """Join wrapped lines, retaining hyphens in configured compounds."""
    prefixes = {"self", "well", "ill"} if compound_prefixes is None else set(compound_prefixes)
    out = ""
    for line in lines:
        if not out:
            out = line
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and line[:1].islower():
            stem = re.search(r"([A-Za-z]+)-$", out)
            out = (out if stem and stem[1].lower() in prefixes else out[:-1]) + line
        else:
            out += " " + line
    return out


def has_dangling_prose_end(text: str) -> bool:
    """Find strong evidence of a split word or unfinished prose sentence."""
    return bool(
        _SPLIT_WORD_END.search(text)
        or (len(text.split()) >= 12 and _CONNECTOR_END.search(text))
    )


def begins_with_continuation(text: str) -> bool:
    """A lowercase start can continue text from the preceding page."""
    return bool(text) and text[0].islower()


def is_boundary_label(text: str) -> bool:
    """Box number labels do not interrupt narrative continuation."""
    return bool(_BOX_LABEL.fullmatch(text))


def completed_sentence_prefix(text: str) -> str:
    """Keep only complete sentences when a PDF ends mid-sentence."""
    endings = list(_SENTENCE_END.finditer(text))
    return text[: endings[-1].end()].strip() if endings else ""
