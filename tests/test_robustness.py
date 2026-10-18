"""The exploratory second-family check: comparability with the primary analysis."""

import numpy as np
import pytest

from admetood import modeling, robustness as rb


def test_quantile_is_excluded_from_the_reported_methods():
    """It would need a pinball-loss learner from the family under test."""
    assert "quantile" not in rb.METHODS
    assert set(rb.METHODS) < set(modeling.METHODS)


def test_the_ensemble_keeps_every_member_so_spread_stays_comparable():
    """Repeats are reduced for cost; ensemble size must not be, or 0.177 is not comparable."""
    seeds = [0, 1, 2, 3, 4]
    x_fit = np.random.default_rng(0).normal(size=(60, 8))
    y_fit = x_fit[:, 0] * 2.0
    rb.MLP_PARAMS["max_iter"] = 20
    out = rb.fit_predict_mlp(x_fit, y_fit, [x_fit[:10]], seeds)
    assert len(out) == 1
    assert out[0].mean.shape == (10,)
    assert out[0].ensemble_sd.shape == (10,)
    assert np.all(out[0].ensemble_sd >= 0)


def test_repeats_are_fewer_than_the_ensemble_members():
    assert rb.REPEATS < 5, "the whole point is that repeats, not members, were reduced"


def test_quantile_arms_are_undefined_and_must_not_be_read():
    seeds = [0, 1]
    x_fit = np.random.default_rng(1).normal(size=(40, 5))
    rb.MLP_PARAMS["max_iter"] = 20
    out = rb.fit_predict_mlp(x_fit, x_fit[:, 0], [x_fit[:5]], seeds)[0]
    assert np.all(np.isnan(out.quantile_low))
    assert np.all(np.isnan(out.quantile_high))


def test_half_widths_for_the_reported_methods_are_finite_despite_nan_quantiles():
    """A NaN quantile arm must not leak into the four methods this check reports."""
    rng = np.random.default_rng(2)
    n = 60
    undefined = np.full(n, np.nan)
    calibration = modeling.Prediction(
        mean=rng.normal(size=n), ensemble_sd=np.abs(rng.normal(size=n)) + 0.1,
        quantile_low=undefined, quantile_high=undefined,
        similarity=rng.uniform(0.2, 0.9, size=n))
    evaluation = modeling.Prediction(
        mean=rng.normal(size=n), ensemble_sd=np.abs(rng.normal(size=n)) + 0.1,
        quantile_low=undefined, quantile_high=undefined,
        similarity=rng.uniform(0.2, 0.9, size=n))
    widths = modeling.half_widths(calibration, rng.normal(size=n), evaluation)
    for method in rb.METHODS:
        # Infinity is a legitimate conformal output: a similarity bin too small for the
        # finite-sample quantile yields an unbounded interval, which is honest. NaN is not,
        # and would mean the undefined quantile arms had leaked into a reported method.
        assert not np.any(np.isnan(widths[method])), method
    assert np.all(np.isnan(widths["quantile"])), "quantile is the one that must stay unusable"

