"""The four mask variants, as character ranges over the rollout text.

    v0    delimited interiors u all M tokens outside them
    v1    v0 minus the first atomic unit of every span
    v0ms  v0 with spans of < 2 atomic units dropped
    v2    M_EXT runs immediately following a relational operator,
          up to the first non-M_EXT token

v0ms exists because v1 differs from v0 along two axes at once: it drops entry
tokens *and* it deletes length-1 spans (lone numerals in prose, "we have 3
cases"). v1 vs v0ms isolates the entry-token effect.
"""

import re

from .charclass import M_EXT
from .spans import Span, find_spans


VARIANTS = ("v0", "v1", "v0ms", "v2")

#: Not just bare `=`: restricting to it drives v2 retention below the gate.
TRIGGERS = (
    "=",
    "≈",
    "≡",
    "≤",
    "≥",
    "<",
    ">",
    r"\approx",
    r"\equiv",
    r"\le",
    r"\ge",
    r"\to",
    r"\Rightarrow",
    r"\boxed{",
)


def _trigger_pattern(triggers=TRIGGERS) -> re.Pattern:
    """Longest-first alternation; `\\le` must not fire inside `\\lemma`."""
    parts = []
    for t in sorted(triggers, key=len, reverse=True):
        esc = re.escape(t)
        if t[-1].isalpha():
            esc += "(?![a-zA-Z])"
        parts.append(esc)
    return re.compile("|".join(parts))


TRIGGER_RE = _trigger_pattern()


def _subtract(ranges: list[tuple[int, int]], holes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove `holes` from `ranges`, keeping the remaining pieces."""
    out = []
    for s, e in ranges:
        pieces = [(s, e)]
        for hs, he in holes:
            nxt = []
            for ps, pe in pieces:
                if he <= ps or hs >= pe:
                    nxt.append((ps, pe))
                    continue
                if ps < hs:
                    nxt.append((ps, hs))
                if he < pe:
                    nxt.append((he, pe))
            pieces = nxt
        out.extend(p for p in pieces if p[1] > p[0])
    return out


def v2_spans(text: str) -> list[Span]:
    """M_EXT runs immediately following a relational operator.

    From the end of each trigger: skip whitespace, then consume consecutive
    M_EXT matches separated only by whitespace, stopping at the first
    non-M_EXT token. `= 47` yields `47`; `= 3x + 2` stops at `x`.
    """
    spans: list[Span] = []
    for tm in TRIGGER_RE.finditer(text):
        pos = tm.end()
        units: list[tuple[int, int]] = []
        while pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            m = M_EXT.match(text, pos)
            if m is None or m.end() == m.start():
                break
            units.append((m.start(), m.end()))
            pos = m.end()
        if units:
            spans.append(Span(units[0][0], units[-1][1], "v2run", units=units, kept=list(units)))
    return spans


def variant_spans(text: str, variant: str) -> tuple[list[Span], dict]:
    """Spans contributing to `variant`, plus the delimiter stats for logging."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if variant == "v2":
        return v2_spans(text), {"openers": 0, "unbalanced": 0, "unbalanced_rate": 0.0, "covered": []}
    spans, stats = find_spans(text)
    if variant == "v0ms":
        spans = [s for s in spans if len(s.units) >= 2]
    return spans, stats


def variant_char_ranges(text: str, variant: str) -> list[tuple[int, int]]:
    """Character ranges entering the mask for `variant`."""
    if variant == "v2":
        spans = v2_spans(text)
        return [r for s in spans for r in s.kept]

    spans, _ = find_spans(text)
    if variant == "v0":
        return [r for s in spans for r in s.kept]
    if variant == "v0ms":
        return [r for s in spans if len(s.units) >= 2 for r in s.kept]
    if variant == "v1":
        out = []
        for s in spans:
            if not s.units:
                continue
            out.extend(_subtract(s.kept, [s.units[0]]))
        return out
    raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
