"""Span detection: delimited interiors, and M-runs in the text outside them.

Delimited spans take precedence. Run detection sees only the text *outside*
them -- otherwise every symbol inside `$...$` would be counted twice.
"""

import re
from dataclasses import dataclass, field

from .charclass import M, atomic_units, decompose


#: Max interior length. An opener with no partner inside this window is treated
#: as unbalanced: it is discarded and the region falls through to run detection.
MAX_INTERIOR = 512

_N = f"{{0,{MAX_INTERIOR}}}?"

DELIM = re.compile(
    rf"\$\$(.{_N})\$\$"  # display  $$..$$
    rf"|\$([^$\n]{_N})\$"  # inline   $..$
    rf"|\\\[(.{_N})\\\]"  # display  \[..\]
    rf"|\\\((.{_N})\\\)",  # inline   \(..\)
    re.S,
)

OPENER = re.compile(r"\$\$|\$|\\\[|\\\(")


@dataclass
class Span:
    """A contiguous region of interest.

    `units` are the atomic units used by v1 (drop the first) and v0ms (drop
    spans with fewer than two). `kept` are the char ranges that actually enter
    the mask -- for a delimited span the interior wholesale, for a run only the
    M matches, never the non-M text bridged by the gap tolerance.
    """

    start: int
    end: int
    kind: str  # "delim" | "run"
    units: list[tuple[int, int]] = field(default_factory=list)
    kept: list[tuple[int, int]] = field(default_factory=list)


def find_delimited_spans(text: str) -> tuple[list[Span], dict]:
    """Interior-only spans for `$$..$$`, `$..$`, `\\[..\\]`, `\\(..\\)`.

    Both opening and closing delimiters are excluded. The interior is kept
    wholesale -- variables, `\\text{}`, `\\frac` all included.

    Returns the spans plus `{"openers", "unbalanced", "unbalanced_rate",
    "covered"}`; `covered` is the list of full-match ranges (delimiters
    included) that run detection must skip.
    """
    spans: list[Span] = []
    covered: list[tuple[int, int]] = []
    for m in DELIM.finditer(text):
        gi = next(i for i in range(1, 5) if m.group(i) is not None)
        s, e = m.span(gi)
        covered.append(m.span())
        if e > s:
            spans.append(Span(s, e, "delim", units=decompose(text, M, s, e), kept=[(s, e)]))

    # An opener that never started a matched pair had no partner within
    # MAX_INTERIOR. Log the rate; above ~5% something is wrong with the regex.
    n_unbalanced = sum(1 for om in OPENER.finditer(text) if not any(cs <= om.start() < ce for cs, ce in covered))
    n_openers = n_unbalanced + len(covered)

    return spans, {
        "openers": n_openers,
        "unbalanced": n_unbalanced,
        "unbalanced_rate": n_unbalanced / n_openers if n_openers else 0.0,
        "covered": covered,
    }


def _gap_tokens(text: str, a: int, b: int) -> int:
    """Number of whitespace-separated chunks strictly between two M matches."""
    return len(text[a:b].split())


def find_run_spans(text: str, covered: list[tuple[int, int]], pattern: re.Pattern = M, k: int = 2) -> list[Span]:
    """Maximal sequences of `pattern` matches with gap tolerance `k`.

    Up to `k` consecutive non-M chunks may sit inside a run without breaking
    it, so `f(x) = 3x + 2` stays one span. Bare letters are never span
    *starters* -- only the M matches enter `kept`.
    """
    matches = [(s, e) for s, e in atomic_units(text, pattern) if not any(cs <= s < ce for cs, ce in covered)]
    if not matches:
        return []

    spans: list[Span] = []
    current = [matches[0]]
    for prev, cur in zip(matches, matches[1:], strict=False):  # offset pairing: lengths differ by one
        crosses = any(prev[1] <= cs and ce <= cur[0] for cs, ce in covered)
        if crosses or _gap_tokens(text, prev[1], cur[0]) > k:
            spans.append(Span(current[0][0], current[-1][1], "run", units=list(current), kept=list(current)))
            current = [cur]
        else:
            current.append(cur)
    spans.append(Span(current[0][0], current[-1][1], "run", units=list(current), kept=list(current)))
    return spans


def find_spans(text: str) -> tuple[list[Span], dict]:
    """All spans for the base variants: delimited interiors + outside runs."""
    delim, stats = find_delimited_spans(text)
    runs = find_run_spans(text, stats["covered"])
    spans = sorted(delim + runs, key=lambda s: s.start)
    return spans, stats
