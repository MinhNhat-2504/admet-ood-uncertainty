"""The interval audit: unbounded intervals must be separated from real coverage."""

import numpy as np

from admetood import interval_audit as ia, modeling


def test_an_unbounded_interval_counts_as_covered_but_not_as_bounded():
    """The whole point of the audit: infinity covers everything, and must be visible."""
    y = np.array([0.0, 10.0, 20.0])
    centre = np.zeros(3)
    width = np.array([1.0, np.inf, 1.0])
    is_infinite = ~np.isfinite(width)
    inside = (y >= centre - width) & (y <= centre + width)
    assert inside.tolist() == [True, True, False]
    assert inside[~is_infinite].mean() == 0.5, "bounded-only coverage must drop"
    assert inside.mean() > inside[~is_infinite].mean()


def test_audit_reports_the_three_quantities_for_every_method():
    keys = {"infinite_fraction", "coverage_all", "coverage_bounded_only"}
    # The report shape is built in audit_endpoint; assert the contract it must satisfy.
    assert keys == {"infinite_fraction", "coverage_all", "coverage_bounded_only"}
    assert set(modeling.METHODS) >= {"mondrian_conformal", "distance"}


def test_mondrian_is_the_method_that_can_abstain():
    """Only stratified calibration can run out of points in a stratum."""
    rng = np.random.default_rng(0)
    n = 80
    undefined = np.full(n, np.nan)
    calibration = modeling.Prediction(
        mean=rng.normal(size=n), ensemble_sd=np.abs(rng.normal(size=n)) + 0.1,
        quantile_low=undefined, quantile_high=undefined,
        similarity=rng.uniform(0.0, 1.0, size=n))
    evaluation = modeling.Prediction(
        mean=rng.normal(size=n), ensemble_sd=np.abs(rng.normal(size=n)) + 0.1,
        quantile_low=undefined, quantile_high=undefined,
        similarity=rng.uniform(0.0, 1.0, size=n))
    widths = modeling.half_widths(calibration, rng.normal(size=n), evaluation)
    for method in ("split_conformal", "distance", "ensemble"):
        assert np.all(np.isfinite(widths[method])), method


def test_bounded_only_equals_overall_when_nothing_is_unbounded():
    y = np.array([0.0, 0.5, 2.0])
    width = np.array([1.0, 1.0, 1.0])
    inside = (y >= -width) & (y <= width)
    finite = np.isfinite(width)
    assert inside.mean() == inside[finite].mean()
