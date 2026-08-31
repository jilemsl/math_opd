"""Character classes and atomic-unit decomposition.

Masks are built from *character spans* only. Nothing here touches token ids:
Qwen3 splits numbers digit-by-digit and every symbol has leading-space
variants, so a token-id mask would not survive a tokenizer change. See
`align.py` for the char-span -> token-index step.
"""

import re


DIGITS = r"\d[\d,]*(?:\.\d+)?%?"
OPERATORS = r"[+\-*/×÷·±^=<>≤≥≈≡~()\[\]{}|_]"

#: Base class: digits and operators, no LaTeX commands.
M = re.compile(f"{DIGITS}|{OPERATORS}")

LATEX_CMD = r"\\[a-zA-Z]+"
GREEK = r"[αβγδεθλμπρστφωΔΣΩ]"

#: Extended class, used by v2 only. Alternation order matters: OPERATORS has no
#: backslash, so a leading `\` falls through to LATEX_CMD.
M_EXT = re.compile(f"{DIGITS}|{OPERATORS}|{LATEX_CMD}|{GREEK}")


def atomic_units(text: str, pattern: re.Pattern, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Char ranges of every `pattern` match in ``text[start:end]``.

    One match is one atomic unit: `12` is a single unit even though the
    tokenizer emits `1`,`2`. v1 drops units, never tokens -- dropping one digit
    and supervising the other is incoherent.
    """
    end = len(text) if end is None else end
    return [(m.start(), m.end()) for m in pattern.finditer(text, start, end)]


def decompose(text: str, pattern: re.Pattern, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Atomic units of a region kept *wholesale* (a delimited interior).

    Unlike `atomic_units`, the non-matching material between matches is also
    emitted as units (maximal whitespace-free chunks), so that "the first atomic
    unit" of `$x = 12$` is `x` and not `=`.
    """
    end = len(text) if end is None else end
    units: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        if text[pos].isspace():
            pos += 1
            continue
        m = pattern.match(text, pos, end)
        if m is not None and m.end() > m.start():
            units.append((m.start(), m.end()))
            pos = m.end()
            continue
        # Maximal chunk of non-matching, non-space text (a variable, a word).
        chunk_start = pos
        while pos < end and not text[pos].isspace() and pattern.match(text, pos, end) is None:
            pos += 1
        if pos == chunk_start:  # defensive: never stall
            pos += 1
        units.append((chunk_start, pos))
    return units
