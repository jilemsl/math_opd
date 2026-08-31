"""Character spans -> token indices, via `offset_mapping` from a HF fast tokenizer.

A token is in the mask if its character span *overlaps* a mask span. Going
through characters is what makes the whitespace-variant problem disappear:
Qwen3 emits ` $` as one token whose span covers the space and the delimiter, and
since the delimiter is excluded from the interior that token is correctly left
out without any special-casing.
"""

import numpy as np

from .variants import variant_char_ranges


def char_ranges_to_token_mask(
    offsets: list[tuple[int, int]], ranges: list[tuple[int, int]], text_len: int | None = None
) -> np.ndarray:
    """Boolean mask over token indices.

    `offsets` is the tokenizer's `offset_mapping` for the completion text.
    Zero-width offsets (special tokens) are never selected.
    """
    n = len(offsets)
    mask = np.zeros(n, dtype=bool)
    if not ranges or n == 0:
        return mask

    if text_len is None:
        text_len = max((e for _, e in offsets), default=0)
        text_len = max(text_len, max((e for _, e in ranges), default=0))

    # Char-level indicator, then per-token overlap test. O(text + tokens).
    flag = np.zeros(text_len + 1, dtype=bool)
    for s, e in ranges:
        if e > s:
            flag[max(s, 0) : min(e, text_len)] = True

    csum = np.concatenate([[0], np.cumsum(flag)])
    for i, (s, e) in enumerate(offsets):
        if e <= s:  # special token / empty span
            continue
        s, e = max(s, 0), min(e, text_len)
        if e > s and csum[e] > csum[s]:
            mask[i] = True
    return mask


def build_mask(text: str, offsets: list[tuple[int, int]], variant: str) -> np.ndarray:
    """Boolean token mask for `variant` over the completion `text`."""
    return char_ranges_to_token_mask(offsets, variant_char_ranges(text, variant), len(text))


def completion_offsets(tokenizer, text: str) -> list[tuple[int, int]]:
    """`offset_mapping` for `text`, without special tokens.

    Use the *completion* text only. Offsets must be relative to the same string
    the masks were built from.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [tuple(o) for o in enc["offset_mapping"]]
