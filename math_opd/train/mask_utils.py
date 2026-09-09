"""Batch-level mask construction over generated completion ids.

The masks are defined on characters, so the batch path needs char offsets for
the *generated* token ids. It builds them by decoding token-by-token and
concatenating, never by re-tokenizing a decoded string: re-tokenization can
segment differently from what the model actually emitted, which would silently
shift every offset.
"""

import numpy as np
import torch

from ..masks.align import char_ranges_to_token_mask
from ..masks.baselines import match_retention, random_stratified_mask
from ..masks.variants import VARIANTS, variant_char_ranges


LEXICAL_ARMS = VARIANTS
#: Arms that must project every position to the vocabulary before they can
#: select. `rkl_min`/`rkl_max` score on the trainer's own per-token loss, so
#: they bound the gradient-mass axis from below and above at a fixed budget.
SCORED_ARMS = ("entropy", "kl", "rkl_min", "rkl_max")
ARMS = ("vanilla", "random", *SCORED_ARMS, *LEXICAL_ARMS)


def offsets_from_ids(tokenizer, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """Decoded text and per-token char offsets, exact by construction.

    Byte-level BPE can split one multi-byte character across two tokens; those
    decode to replacement characters here. The offsets still partition the
    string, so alignment stays valid -- such positions simply never match a
    mask span. Rare in math text; measured by the Phase 0 retention log.
    """
    pieces = tokenizer.batch_decode([[i] for i in ids], skip_special_tokens=False)
    offsets, pos = [], 0
    for p in pieces:
        offsets.append((pos, pos + len(p)))
        pos += len(p)
    return "".join(pieces), offsets


def lexical_batch_mask(
    tokenizer, completion_ids: torch.Tensor, completion_mask: torch.Tensor, variant: str
) -> torch.Tensor:
    """`(B, T)` float mask selecting the `variant`'s tokens inside the completion."""
    out = torch.zeros_like(completion_mask, dtype=torch.float32)
    for b in range(completion_ids.size(0)):
        keep = completion_mask[b].bool()
        n = int(keep.sum())
        if n == 0:
            continue
        ids = completion_ids[b][keep].tolist()
        text, offsets = offsets_from_ids(tokenizer, ids)
        sel = char_ranges_to_token_mask(offsets, variant_char_ranges(text, variant), len(text))
        out[b, : completion_mask.size(1)][keep] = torch.from_numpy(sel.astype(np.float32)).to(out.device)
    return out


def baseline_batch_mask(
    target: torch.Tensor,
    completion_mask: torch.Tensor,
    arm: str,
    scores: torch.Tensor | None = None,
    seed: int = 0,
    step: int = 0,
) -> torch.Tensor:
    """Budget-matched baseline mask.

    `target` is the v0 mask for the same batch. Its per-sequence masked count is
    the budget every baseline must hit -- retuned each batch, because a fixed
    global threshold drifts out of match as the student trains.
    """
    out = torch.zeros_like(completion_mask, dtype=torch.float32)
    rng = np.random.default_rng(seed + step)
    tgt = target.detach().cpu().numpy().astype(bool)
    valid = completion_mask.detach().cpu().numpy().astype(bool)
    sc = None if scores is None else scores.detach().float().cpu().numpy()

    for b in range(out.size(0)):
        if not valid[b].any():
            continue
        if arm == "random":
            sel = random_stratified_mask(tgt[b], valid[b], rng=rng)
        elif arm in SCORED_ARMS:
            if sc is None:
                raise ValueError(f"arm {arm!r} needs `scores`")
            sel = match_retention(sc[b], tgt[b], valid[b])
        else:
            raise ValueError(f"unknown baseline arm {arm!r}")
        out[b] = torch.from_numpy(sel.astype(np.float32)).to(out.device)
    return out


def retention(mask: torch.Tensor, completion_mask: torch.Tensor) -> float:
    """Fraction of completion tokens the mask keeps. Log this every step."""
    denom = completion_mask.sum()
    return float((mask.sum() / denom).item()) if denom > 0 else 0.0
