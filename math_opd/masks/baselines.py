"""Budget-matched baseline masks.

Budget matching is the whole ballgame. Every baseline's threshold is tuned
*per batch* to v0's retention on that batch -- a fixed global threshold drifts
out of match as the student trains and the comparison quietly stops being
like-for-like.
"""

import numpy as np


def topk_mask(scores: np.ndarray, budget: int, valid: np.ndarray | None = None) -> np.ndarray:
    """Top-`budget` positions by `scores`, restricted to `valid` positions.

    Used for both the entropy baseline (TIP-style, student entropy) and the KL
    baseline (teacher-student KL). Neither can select without first scoring
    everything -- which is exactly the compute asymmetry the paper measures.
    """
    mask = np.zeros(scores.shape, dtype=bool)
    if budget <= 0:
        return mask
    valid = np.ones(scores.shape, dtype=bool) if valid is None else valid.astype(bool)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return mask
    budget = min(budget, idx.size)
    order = idx[np.argsort(-scores[idx], kind="stable")]
    mask[order[:budget]] = True
    return mask


def random_stratified_mask(
    target: np.ndarray, valid: np.ndarray | None = None, n_bins: int = 20, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Random mask reproducing `target`'s positional and count profile.

    Bins normalized position into `n_bins`, then samples uniformly *within each
    bin* exactly as many positions as `target` has there. Uniform random would
    not do: a positive result against it reads as "sparse gating helps", which
    DOPD 2606.30626 and Rock Tokens 2605.09253 already showed is roughly free.
    """
    rng = np.random.default_rng() if rng is None else rng
    target = target.astype(bool)
    n = target.shape[-1]
    mask = np.zeros(n, dtype=bool)
    valid = np.ones(n, dtype=bool) if valid is None else valid.astype(bool)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return mask

    edges = np.linspace(0, n_valid, n_bins + 1)
    valid_idx = np.flatnonzero(valid)
    # Rank each valid position within the completion, so bins are over the real
    # response length rather than over padding.
    rank = np.empty(n, dtype=float)
    rank[valid_idx] = np.arange(n_valid)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = valid_idx[(rank[valid_idx] >= lo) & (rank[valid_idx] < hi)]
        if in_bin.size == 0:
            continue
        k = int(target[in_bin].sum())
        if k <= 0:
            continue
        mask[rng.choice(in_bin, size=min(k, in_bin.size), replace=False)] = True
    return mask


def match_retention(scores: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Top-k baseline whose budget equals `target`'s masked count."""
    return topk_mask(scores, int(np.asarray(target).astype(bool).sum()), valid)


def token_entropy(logprobs: np.ndarray) -> np.ndarray:
    """Shannon entropy per position, in nats, from full log-probs `(T, V)`."""
    p = np.exp(logprobs)
    return -(p * logprobs).sum(-1)


def token_kl(teacher_logprobs: np.ndarray, student_logprobs: np.ndarray) -> np.ndarray:
    """KL(teacher || student) per position, from full log-probs `(T, V)`."""
    pt = np.exp(teacher_logprobs)
    return (pt * (teacher_logprobs - student_logprobs)).sum(-1)
