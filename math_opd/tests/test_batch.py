"""Batch-path invariants: offsets must come from the generated ids, not a re-tokenization."""

import numpy as np
import pytest
import torch

from math_opd.masks.variants import VARIANTS
from math_opd.train.mask_utils import ARMS, baseline_batch_mask, lexical_batch_mask, offsets_from_ids, retention


class FakeTokenizer:
    """Character-level stand-in: token id == codepoint."""

    def batch_decode(self, batch, skip_special_tokens=False):
        return ["".join(chr(i) for i in ids) for ids in batch]


def ids_of(text):
    return [ord(c) for c in text]


def test_offsets_partition_the_decoded_text():
    text = "x = 12.5"
    t, offsets = offsets_from_ids(FakeTokenizer(), ids_of(text))
    assert t == text
    assert offsets[0][0] == 0
    assert offsets[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in zip(offsets, offsets[1:], strict=False)), "offsets must be contiguous"


def test_offsets_from_real_tokenizer_reconstruct_text_exactly():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    except Exception as exc:  # offline / not cached
        pytest.skip(f"tokenizer unavailable: {exc}")
    src = r"We get $x = 12.5$ so \boxed{47}."
    ids = tok(src, add_special_tokens=False)["input_ids"]
    text, offsets = offsets_from_ids(tok, ids)
    assert text == src
    assert len(offsets) == len(ids)
    assert offsets[-1][1] == len(text)


@pytest.mark.parametrize("variant", VARIANTS)
def test_lexical_batch_mask_stays_inside_completion(variant):
    text = "$x = 12$ and f(y) = 3 so 4"
    ids = torch.tensor([ids_of(text) + [0, 0]])
    cmask = torch.tensor([[1] * len(text) + [0, 0]])
    m = lexical_batch_mask(FakeTokenizer(), ids, cmask, variant)
    assert m.shape == cmask.shape
    assert m[0, -2:].sum() == 0, "padding must never be selected"
    assert (m <= cmask).all()


def test_lexical_batch_mask_matches_single_sequence_masks():
    text = "$x = 12$ and 3 + 4"
    ids = torch.tensor([ids_of(text)])
    cmask = torch.ones(1, len(text), dtype=torch.long)
    m = lexical_batch_mask(FakeTokenizer(), ids, cmask, "v0")
    selected = "".join(c for c, k in zip(text, m[0].bool().tolist(), strict=True) if k)
    assert "x = 12" in selected
    assert "$" not in selected


def test_random_baseline_matches_v0_budget_per_sequence():
    text = "$x = 12$ and 3 + 4 with 5"
    ids = torch.tensor([ids_of(text)])
    cmask = torch.ones(1, len(text), dtype=torch.long)
    target = lexical_batch_mask(FakeTokenizer(), ids, cmask, "v0")
    rnd = baseline_batch_mask(target, cmask, "random", seed=0, step=0)
    assert rnd.sum() == target.sum(), "budget must match v0 exactly"


def test_scored_baseline_requires_scores():
    cmask = torch.ones(1, 8, dtype=torch.long)
    target = torch.zeros(1, 8)
    target[0, :3] = 1
    with pytest.raises(ValueError, match="needs `scores`"):
        baseline_batch_mask(target, cmask, "entropy")


def test_scored_baseline_matches_budget_when_given_scores():
    cmask = torch.ones(1, 8, dtype=torch.long)
    target = torch.zeros(1, 8)
    target[0, :3] = 1
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]])
    m = baseline_batch_mask(target, cmask, "entropy", scores=scores)
    assert m.sum() == 3
    assert m[0].bool().tolist() == [False, True, False, True, False, True, False, False]


def test_unknown_baseline_arm_raises():
    cmask = torch.ones(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="unknown baseline arm"):
        baseline_batch_mask(torch.zeros(1, 4), cmask, "nope")


def test_retention_ignores_padding():
    cmask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    assert retention(mask, cmask) == pytest.approx(0.5)


def test_arm_list_covers_every_variant_and_baseline():
    assert set(VARIANTS) <= set(ARMS)
    assert {"vanilla", "random", "entropy", "kl"} <= set(ARMS)


def test_empty_completion_produces_empty_mask():
    cmask = torch.zeros(1, 5, dtype=torch.long)
    ids = torch.zeros(1, 5, dtype=torch.long)
    m = lexical_batch_mask(FakeTokenizer(), ids, cmask, "v0")
    assert m.sum() == 0
    assert np.isclose(retention(m, cmask), 0.0)
