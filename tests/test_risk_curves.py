"""The recovered risk-coverage curves: what they compare and what they must not claim."""

import numpy as np

from admetood import risk_curves as rc


def curves(mae_by_method: dict[str, list[float]]) -> dict:
    return {"coverage_grid": [0.2, 0.6, 1.0],
            "methods": {m: {"mae_at_coverage": v} for m, v in mae_by_method.items()}}


def test_discrimination_is_positive_when_the_method_keeps_less_error():
    value = rc.discrimination(
        curves({"quantile": [0.30, 0.40, 0.50], "ensemble": [0.35, 0.45, 0.55]}),
        "quantile", "ensemble")
    assert value == 0.05


def test_discrimination_is_negative_when_the_method_is_worse():
    value = rc.discrimination(
        curves({"distance": [0.40, 0.50, 0.60], "ensemble": [0.35, 0.45, 0.55]}),
        "distance", "ensemble")
    assert value == -0.05


def test_identical_signals_score_exactly_zero():
    """split_conformal and ensemble share a ranking signal; the zero must be an identity."""
    shared = [0.35, 0.45, 0.55]
    value = rc.discrimination(
        curves({"split_conformal": shared, "ensemble": list(shared)}),
        "split_conformal", "ensemble")
    assert value == 0.0


def test_a_missing_method_yields_none_rather_than_a_misleading_zero():
    assert rc.discrimination(curves({"ensemble": [0.3]}), "absent", "ensemble") is None
    assert rc.discrimination(curves({"quantile": [0.3]}), "quantile", "absent") is None


def test_the_ranking_signals_are_three_not_five():
    """Section 6.4 rests on this; if modeling adds a distinct signal the claim changes."""
    from admetood import modeling

    class Fake:
        ensemble_sd = np.array([1.0, 2.0, 3.0])
        quantile_high = np.array([2.0, 3.0, 4.0])
        quantile_low = np.array([0.0, 0.0, 0.0])
        similarity = np.array([0.1, 0.2, 0.3])

    signals = modeling.sigmas(Fake())
    distinct = {tuple(np.asarray(v, dtype=float).round(9)) for v in signals.values()}
    assert len(signals) == 5 and len(distinct) == 3
    assert np.array_equal(signals["split_conformal"], signals["ensemble"])
    assert np.array_equal(signals["distance"], signals["mondrian_conformal"])
