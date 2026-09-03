"""Invariants from MATH_MASK_EXPERIMENT.md, especially its "Gotchas" list."""

import numpy as np
import pytest

from math_opd.masks.align import char_ranges_to_token_mask
from math_opd.masks.baselines import match_retention, random_stratified_mask, topk_mask
from math_opd.masks.charclass import M_EXT, M, atomic_units, decompose
from math_opd.masks.spans import find_delimited_spans, find_run_spans, find_spans
from math_opd.masks.variants import VARIANTS, v2_spans, variant_char_ranges


def covered_text(text, ranges):
    return "".join(text[s:e] for s, e in sorted(ranges))


def char_set(ranges):
    return {i for s, e in ranges for i in range(s, e)}


# --- character classes -------------------------------------------------------


def test_digits_are_one_atomic_unit():
    text = "12.5 and 1,000 and 40%"
    assert [text[s:e] for s, e in atomic_units(text, M)] == ["12.5", "1,000", "40%"]


def test_m_ext_matches_latex_and_greek_but_m_does_not():
    text = r"\alpha \frac 3 + π"
    assert r"\alpha" in [text[s:e] for s, e in atomic_units(text, M_EXT)]
    assert r"\alpha" not in [text[s:e] for s, e in atomic_units(text, M)]


def test_decompose_emits_variables_as_units():
    text = "x = 12"
    assert [text[s:e] for s, e in decompose(text, M)] == ["x", "=", "12"]


# --- delimiters --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "interior"),
    [
        ("a $x=1$ b", "x=1"),
        ("a $$x=1$$ b", "x=1"),
        (r"a \[x=1\] b", "x=1"),
        (r"a \(x=1\) b", "x=1"),
    ],
)
def test_delimiters_excluded_interior_kept(text, interior):
    spans, _ = find_delimited_spans(text)
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == interior


def test_interior_kept_wholesale_including_text_and_frac():
    text = r"$\frac{1}{2} \text{ apples}$"
    spans, _ = find_delimited_spans(text)
    assert text[spans[0].start : spans[0].end] == r"\frac{1}{2} \text{ apples}"


def test_no_double_counting_between_delimited_and_runs():
    text = "outside 3 + 4 then $x = 5$ done"
    spans, stats = find_spans(text)
    delim = [s for s in spans if s.kind == "delim"]
    runs = [s for s in spans if s.kind == "run"]
    assert delim and runs
    d_chars = char_set([(s.start, s.end) for s in delim])
    r_chars = char_set([r for s in runs for r in s.kept])
    assert not (d_chars & r_chars), "run detection must skip delimited regions"


def test_unbalanced_opener_is_counted_and_falls_back_to_runs():
    text = "costs $5 for 3 apples"
    spans, stats = find_spans(text)
    assert stats["unbalanced"] == 1
    assert stats["unbalanced_rate"] == 1.0
    assert all(s.kind == "run" for s in spans)
    assert "5" in covered_text(text, [r for s in spans for r in s.kept])


def test_overlong_interior_is_not_a_span():
    text = "$" + "a" * 600 + "$"
    spans, stats = find_delimited_spans(text)
    assert spans == []
    assert stats["unbalanced"] == 2


# --- runs and gap tolerance --------------------------------------------------


def test_gap_tolerance_keeps_expression_one_span():
    text = "f(x) = 3x + 2"
    spans = find_run_spans(text, [])
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "(x) = 3x + 2"


def test_bare_letters_are_not_span_starters():
    text = "f(x) = 3x + 2"
    spans = find_run_spans(text, [])
    assert text[spans[0].start] == "(", "span must not start at the bare `f`"


def test_gap_over_tolerance_breaks_the_run():
    text = "3 one two three 4"
    spans = find_run_spans(text, [])
    assert len(spans) == 2


def test_run_kept_excludes_bridged_non_m_text():
    text = "3x + 2"
    spans = find_run_spans(text, [])
    kept = covered_text(text, spans[0].kept)
    assert "x" not in kept and kept == "3+2"


# --- variants ----------------------------------------------------------------


def test_v1_drops_first_atomic_unit_whole_not_first_token():
    text = "$12 + 3$"
    v0 = char_set(variant_char_ranges(text, "v0"))
    v1 = char_set(variant_char_ranges(text, "v1"))
    dropped = v0 - v1
    assert covered_text(text, [(min(dropped), max(dropped) + 1)]) == "12", "both digits of `12` must go"


def test_v1_drops_padding_before_the_entry_unit():
    """A delimited interior keeps its whitespace, so removing only the unit's own
    range would leave the space in front of it in the mask -- and the tokenizer
    glues that space to the entry token, re-selecting what v1 exists to drop."""
    text = "so the value is $ x $ here"
    kept = char_set(variant_char_ranges(text, "v1"))
    assert not (kept & char_set([(16, 19)])), "neither `x` nor the space before it may survive v1"


def test_v1_and_v0ms_are_subsets_of_v0():
    text = r"We get $x = 12.5$ and f(x) = 3x + 2 with 7 cases and \[y=1\]."
    v0 = char_set(variant_char_ranges(text, "v0"))
    assert char_set(variant_char_ranges(text, "v1")) <= v0
    assert char_set(variant_char_ranges(text, "v0ms")) <= v0


def test_v0ms_drops_single_unit_spans():
    text = "we have 3 cases. entirely separate prose here. and 4 more"
    spans, _ = find_spans(text)
    assert all(len(s.units) == 1 for s in spans)
    assert variant_char_ranges(text, "v0") != []
    assert variant_char_ranges(text, "v0ms") == []


def test_v2_follows_trigger_and_stops_at_first_non_m_ext():
    text = "so x = 3x + 2"
    spans = v2_spans(text)
    assert len(spans) == 1
    assert covered_text(text, spans[0].kept) == "3", "must stop at the variable `x`"


def test_v2_captures_final_value_and_boxed():
    assert covered_text("total = 47.", v2_spans("total = 47.")[0].kept) == "47"
    text = r"\boxed{47}"
    assert covered_text(text, v2_spans(text)[0].kept) == "47}"


def test_v2_trigger_does_not_fire_inside_a_longer_command():
    assert v2_spans(r"\lemma 3") == []


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown variant"):
        variant_char_ranges("x = 1", "v9")


# --- alignment ---------------------------------------------------------------


def test_alignment_selects_every_token_overlapping_a_span():
    # `12.5` split digit-by-digit, as Qwen3 does.
    offsets = [(0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]  # "abc","1","2",".","5","z"
    mask = char_ranges_to_token_mask(offsets, [(3, 7)], text_len=8)
    assert mask.tolist() == [False, True, True, True, True, False]


def test_alignment_ignores_zero_width_special_tokens():
    mask = char_ranges_to_token_mask([(0, 0), (0, 2), (0, 0)], [(0, 2)], text_len=2)
    assert mask.tolist() == [False, True, False]


def test_leading_space_delimiter_token_is_not_selected():
    # Qwen3 emits " $" as one token spanning the space and the delimiter; the
    # interior starts after it, so char-level alignment excludes it for free.
    offsets = [(0, 2), (2, 3), (3, 4)]  # " $", "x", "$"
    mask = char_ranges_to_token_mask(offsets, [(2, 3)], text_len=4)
    assert mask.tolist() == [False, True, False]


def test_empty_ranges_give_empty_mask():
    assert not char_ranges_to_token_mask([(0, 1), (1, 2)], [], text_len=2).any()


# --- baselines ---------------------------------------------------------------


def test_topk_respects_budget_and_validity():
    scores = np.array([5.0, 1.0, 4.0, 3.0])
    valid = np.array([True, True, False, True])
    mask = topk_mask(scores, 2, valid)
    assert mask.tolist() == [True, False, False, True]


def test_match_retention_matches_target_count_exactly():
    rng = np.random.default_rng(0)
    target = rng.random(200) < 0.13
    mask = match_retention(rng.random(200), target)
    assert mask.sum() == target.sum()


def test_random_stratified_matches_count_and_positional_profile():
    target = np.zeros(200, dtype=bool)
    target[:20] = True  # all mass in the first bins
    mask = random_stratified_mask(target, n_bins=20, rng=np.random.default_rng(0))
    assert mask.sum() == target.sum()
    assert mask[:20].sum() == 20, "positional profile must be reproduced, not uniform"


def test_random_stratified_differs_lexically_from_target():
    rng = np.random.default_rng(1)
    target = np.zeros(400, dtype=bool)
    target[::4] = True
    mask = random_stratified_mask(target, n_bins=20, rng=rng)
    assert mask.sum() == target.sum()
    assert (mask != target).any(), "must not reproduce the exact selection"


# --- end-to-end --------------------------------------------------------------


def test_all_variants_produce_nonempty_masks_on_typical_math_text():
    text = r"First $x = 12$. Then f(x) = 3x + 2 so \[y = 47\] and \boxed{47}."
    offsets = [(i, i + 1) for i in range(len(text))]
    for v in VARIANTS:
        mask = char_ranges_to_token_mask(offsets, variant_char_ranges(text, v), len(text))
        assert mask.any(), f"{v} produced an empty mask"
        assert mask.sum() < len(text), f"{v} selected everything"
