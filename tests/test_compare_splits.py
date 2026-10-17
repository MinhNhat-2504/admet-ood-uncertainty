"""The secondary split-constructor comparison: alignment, pairing and the reading."""

import numpy as np
import pytest

from admetood import compare_splits as cs


def report(gaps_by_split: dict[str, list[float]], endpoints: list[str]) -> dict:
    return {
        "eligible_endpoints": endpoints,
        "comparisons": [
            {"split": split, "endpoint": endpoint,
             "methods": {cs.PRIMARY_METHOD: {"gap_pp": gap}}}
            for split, gaps in gaps_by_split.items()
            for endpoint, gap in zip(endpoints, gaps)
        ],
    }


def test_absolute_gaps_takes_magnitude_and_keeps_endpoint_order():
    gaps = cs.absolute_gaps(report({"random": [-4.0, 7.0]}, ["a", "b"]))
    assert gaps["random"].tolist() == [4.0, 7.0]


def test_a_split_missing_an_endpoint_is_dropped_rather_than_mispaired():
    """Pairing across endpoints is only defined when both splits cover all of them."""
    partial = report({"random": [1.0, 2.0]}, ["a", "b"])
    partial["comparisons"].append(
        {"split": "scaffold", "endpoint": "a",
         "methods": {cs.PRIMARY_METHOD: {"gap_pp": 5.0}}})
    gaps = cs.absolute_gaps(partial)
    assert set(gaps) == {"random"}


def test_paired_difference_recovers_a_known_offset():
    a = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    result = cs.paired_difference(a, a - 3.0)
    assert result["mean_difference_pp"] == pytest.approx(3.0)
    assert result["standard_error_pp"] == pytest.approx(0.0)
    assert result["ci95_pp"] == [pytest.approx(3.0), pytest.approx(3.0)]


def test_identical_splits_are_not_reported_as_different():
    """Wilcoxon is undefined on an all-zero difference; it must not read as separated."""
    a = np.array([3.0, 5.0, 7.0, 9.0])
    result = cs.paired_difference(a, a.copy())
    assert result["mean_difference_pp"] == 0.0
    assert result["wilcoxon_p"] == 1.0


def test_indistinguishable_splits_yield_the_not_shown_reading():
    """Gaps that cross back and forth must not be read as one split beating another."""
    endpoints = [f"e{i}" for i in range(8)]
    result = cs.compare(report(
        {"random":   [12.0, 5.0, 18.0, 9.0, 22.0, 7.0, 14.0, 11.0],
         "scaffold": [10.0, 8.0, 15.0, 13.0, 19.0, 6.0, 17.0, 9.0],
         "butina":   [11.0, 6.0, 17.0, 11.0, 20.0, 8.0, 15.0, 10.0]},
        endpoints))
    assert result["any_pair_separated_at_0.05"] == []
    assert "not shown" in result["reading"]
    assert result["friedman"]["p_value"] > 0.05


def test_a_real_separation_is_reported_as_such():
    endpoints = [f"e{i}" for i in range(8)]
    base = np.linspace(5.0, 20.0, 8)
    result = cs.compare(report(
        {"random": list(base + 15.0), "scaffold": list(base)}, endpoints))
    assert result["any_pair_separated_at_0.05"] == ["random_minus_scaffold"]
    assert "Separated pairs" in result["reading"]


def test_compare_refuses_a_single_split():
    result = cs.compare(report({"random": [1.0, 2.0]}, ["a", "b"]))
    assert "error" in result


def test_two_splits_are_compared_without_a_friedman_test():
    """Friedman needs three groups; two splits must still produce a paired comparison."""
    endpoints = [f"e{i}" for i in range(8)]
    base = np.linspace(4.0, 18.0, 8)
    result = cs.compare(report(
        {"random": list(base + 15.0), "scaffold": list(base)}, endpoints))
    assert result["friedman"] is None
    assert result["all_pairs"]["random_minus_scaffold"]["mean_difference_pp"] > 0
