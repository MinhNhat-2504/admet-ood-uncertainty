"""Conformal must attain its nominal level on exchangeable data and degrade honestly."""

import numpy as np
import pytest

from admetood import uncertainty as unc


def test_conformal_attains_nominal_coverage_on_exchangeable_data():
    """The guarantee the method actually makes; a failure here is an implementation bug."""
    rng = np.random.default_rng(0)
    coverages = []
    for _ in range(200):
        calibration = np.abs(rng.normal(size=500))
        evaluation = np.abs(rng.normal(size=500))
        q = unc.conformal_quantile(calibration, 0.90)
        coverages.append((evaluation <= q).mean())
    # Split conformal is valid in expectation over calibration draws.
    assert 0.89 <= np.mean(coverages) <= 0.92


def test_conformal_undercovers_when_evaluation_is_shifted():
    """The effect this study measures: exchangeability broken, guarantee lost."""
    rng = np.random.default_rng(1)
    q = unc.conformal_quantile(np.abs(rng.normal(size=1000)), 0.90)
    shifted = np.abs(rng.normal(loc=1.5, size=1000))
    assert (shifted <= q).mean() < 0.80


def test_small_calibration_set_yields_infinite_width_not_false_confidence():
    assert unc.conformal_quantile(np.arange(5.0), 0.90) == float("inf")
    assert np.isfinite(unc.conformal_quantile(np.arange(100.0), 0.90))


def test_conformal_quantile_rank_is_the_finite_sample_one():
    scores = np.arange(1.0, 101.0)           # n = 100
    # ceil((100+1) * 0.9) = 91 -> the 91st smallest score.
    assert unc.conformal_quantile(scores, 0.90) == 91.0


def test_mondrian_quantiles_are_computed_per_bin():
    scores = np.concatenate([np.full(100, 1.0), np.full(100, 10.0)])
    bins = np.concatenate([np.zeros(100, int), np.ones(100, int)])
    quantiles = unc.mondrian_quantiles(scores, bins, 0.90)
    assert quantiles[0] == pytest.approx(1.0)
    assert quantiles[1] == pytest.approx(10.0)


def test_similarity_bins_respect_prespecified_edges():
    binned = unc.similarity_bins(np.array([0.1, 0.45, 0.6, 0.8]))
    assert list(binned) == [0, 1, 2, 3]


def test_error_ranking_detects_useful_and_useless_uncertainty():
    y_true = np.arange(100.0)
    errors = np.linspace(0, 10, 100)
    y_pred = y_true + errors
    assert unc.error_ranking(y_true, y_pred, errors) > 0.99       # Perfectly informative.
    constant = unc.error_ranking(y_true, y_pred, np.ones(100))
    assert constant is None                                       # No ordering to score.


def test_risk_coverage_retains_more_when_uncertainty_is_informative():
    y_true = np.zeros(100)
    errors = np.linspace(0.0, 2.0, 100)
    informative = unc.risk_coverage(y_true, errors, errors, target_error=0.5)
    uninformative = unc.risk_coverage(y_true, errors, np.ones(100), target_error=0.5)
    assert informative["retained_at_target"] > uninformative["retained_at_target"]
    assert len(informative["curve"]) == 10


def test_constant_uncertainty_cannot_be_flattered_by_input_order():
    """Ties must not inherit the caller's ordering, which may be sorted by error."""
    y_true = np.zeros(100)
    errors_sorted = np.linspace(0.0, 2.0, 100)
    constant = np.ones(100)
    ascending = unc.risk_coverage(y_true, errors_sorted, constant, 0.5)
    descending = unc.risk_coverage(y_true, errors_sorted[::-1], constant, 0.5)
    assert ascending["retained_at_target"] == descending["retained_at_target"]
    # With no information, the retained fraction reflects the overall error rate only.
    assert ascending["retained_at_target"] < 0.75


def test_interval_metrics_count_infinite_intervals():
    y_true, y_pred = np.zeros(4), np.zeros(4)
    widths = np.array([1.0, 1.0, np.inf, np.inf])
    result = unc.interval_metrics(y_true, y_pred, widths)
    assert result["coverage"] == 1.0
    assert result["infinite_intervals"] == 2
    assert result["mean_width"] == pytest.approx(2.0)
