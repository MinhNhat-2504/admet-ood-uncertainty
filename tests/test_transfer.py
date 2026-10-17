"""Verdict logic, multiplicity control and the prespecified outcome rule."""

import numpy as np
import pytest

from admetood import transfer as tr


def test_verdict_three_way_against_the_margin():
    assert tr.verdict(-3.0, 4.0, margin=5.0) == tr.VERDICT_TRANSFERS
    assert tr.verdict(6.0, 12.0, margin=5.0) == tr.VERDICT_FAILS
    assert tr.verdict(-20.0, -7.0, margin=5.0) == tr.VERDICT_FAILS
    assert tr.verdict(-6.0, 2.0, margin=5.0) == tr.VERDICT_INCONCLUSIVE
    assert tr.verdict(float("nan"), 1.0, margin=5.0) == tr.VERDICT_INCONCLUSIVE


def test_verdict_boundary_is_inclusive_for_transfer():
    """A gap exactly at the margin is treated as equivalent, as the plan states."""
    assert tr.verdict(-5.0, 5.0, margin=5.0) == tr.VERDICT_TRANSFERS


def test_bootstrap_recovers_a_known_difference():
    simulated = np.concatenate([np.ones(900), np.zeros(100)])   # 90% coverage
    real = np.concatenate([np.ones(700), np.zeros(300)])        # 70% coverage
    observed, low, high = tr.bootstrap_difference(simulated, real, 2000, seed=0)
    assert observed == pytest.approx(20.0, abs=1e-6)
    assert low < 20.0 < high
    assert high - low < 10.0, "interval implausibly wide for n=1000 per side"


def test_bootstrap_handles_empty_input():
    observed, low, high = tr.bootstrap_difference(np.array([]), np.ones(10), 100, 0)
    assert all(np.isnan(v) for v in (observed, low, high))


def test_benjamini_hochberg_controls_discoveries():
    assert tr.benjamini_hochberg([0.001, 0.002, 0.003], 0.05) == [True, True, True]
    assert tr.benjamini_hochberg([0.9, 0.8, 0.7], 0.05) == [False, False, False]
    # A clearly significant value survives alongside null ones.
    rejected = tr.benjamini_hochberg([0.0001] + [0.6] * 9, 0.05)
    assert rejected[0] and not any(rejected[1:])
    assert tr.benjamini_hochberg([], 0.05) == []


def test_equivalence_p_value_is_small_only_outside_the_margin():
    outside = np.random.default_rng(0).normal(loc=20.0, scale=1.0, size=10000)
    inside = np.random.default_rng(1).normal(loc=0.0, scale=1.0, size=10000)
    assert tr.equivalence_p_value(outside, margin=5.0) < 0.01
    assert tr.equivalence_p_value(inside, margin=5.0) > 0.5


def test_hypothesis_outcome_applies_the_prespecified_rule():
    structure = ("scaffold", "butina", "frontier", "size")
    def counts(transfers, fails, inconclusive):
        return {tr.VERDICT_TRANSFERS: transfers, tr.VERDICT_FAILS: fails,
                tr.VERDICT_INCONCLUSIVE: inconclusive}

    all_fail = {name: counts(0, 5, 3) for name in structure}
    assert tr.hypothesis_outcome(all_fail, structure, 8) == "supported"

    one_transfers = {**all_fail, "scaffold": counts(6, 1, 1)}
    assert tr.hypothesis_outcome(one_transfers, structure, 8) == "rejected"

    partial = {**all_fail, "scaffold": counts(2, 2, 4)}
    assert tr.hypothesis_outcome(partial, structure, 8) == "mixed"

    assert tr.hypothesis_outcome({}, structure, 8) == "not_evaluable"


def test_summarize_verdicts_counts_every_category():
    summary = tr.summarize_verdicts([tr.VERDICT_FAILS, tr.VERDICT_FAILS,
                                     tr.VERDICT_TRANSFERS])
    assert summary == {tr.VERDICT_TRANSFERS: 1, tr.VERDICT_FAILS: 2,
                       tr.VERDICT_INCONCLUSIVE: 0}
