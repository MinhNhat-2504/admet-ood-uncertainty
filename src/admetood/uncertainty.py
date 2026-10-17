"""Conformal calibration and the metrics used to score uncertainty.

Split conformal attains its nominal coverage under exchangeability of calibration and
evaluation data. A prospective split violates that assumption by construction, so the
coverage it actually attains is the quantity this study measures.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

NOMINAL_COVERAGE = 0.90
MIN_CALIBRATION = 20  # Below this the conformal quantile is not defined meaningfully.


def conformal_quantile(scores: np.ndarray, coverage: float = NOMINAL_COVERAGE) -> float:
    """Finite-sample conformal quantile of nonconformity scores.

    Uses the rank ceil((n+1)(1-alpha))/n, which is what makes split conformal attain at
    least `coverage` on exchangeable data. Returns infinity when the calibration set is
    too small to guarantee the level, so intervals become uninformative rather than
    falsely narrow.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n < MIN_CALIBRATION:
        return float("inf")
    rank = int(np.ceil((n + 1) * coverage))
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


def absolute_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))


def normalized_scores(y_true: np.ndarray, y_pred: np.ndarray,
                      sigma: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Residuals scaled by predicted uncertainty, giving width that varies per molecule."""
    return absolute_scores(y_true, y_pred) / np.maximum(np.asarray(sigma, float), floor)


def mondrian_quantiles(scores: np.ndarray, bins: np.ndarray,
                       coverage: float = NOMINAL_COVERAGE) -> dict[int, float]:
    """A separate conformal quantile per bin.

    Bins are intended to group molecules by how far they sit from training, so that
    calibration is conditional on distance rather than pooled across it.
    """
    quantiles = {}
    for bin_id in np.unique(bins):
        selected = scores[bins == bin_id]
        quantiles[int(bin_id)] = conformal_quantile(selected, coverage)
    return quantiles


def similarity_bins(similarity: np.ndarray, edges: tuple[float, ...] = (0.4, 0.55, 0.7)) -> np.ndarray:
    """Discretize nearest-neighbour similarity into fixed, prespecified bins."""
    return np.digitize(np.asarray(similarity, float), edges)


def coverage_of(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    inside = (np.asarray(y_true, float) >= lower) & (np.asarray(y_true, float) <= upper)
    return float(inside.mean())


def interval_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     half_width: np.ndarray) -> dict:
    lower, upper = y_pred - half_width, y_pred + half_width
    finite = np.isfinite(half_width)
    return {
        "coverage": coverage_of(y_true, lower, upper),
        "mean_width": float(np.mean(2 * half_width[finite])) if finite.any() else None,
        "median_width": float(np.median(2 * half_width[finite])) if finite.any() else None,
        "infinite_intervals": int((~finite).sum()),
    }


def error_ranking(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray) -> float | None:
    """Spearman correlation between predicted uncertainty and realised absolute error."""
    errors = absolute_scores(y_true, y_pred)
    sigma = np.asarray(sigma, float)
    ok = np.isfinite(errors) & np.isfinite(sigma)
    if ok.sum() < 3 or np.unique(sigma[ok]).size < 2:
        return None
    return float(stats.spearmanr(sigma[ok], errors[ok]).statistic)


def risk_coverage(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray,
                  target_error: float, tie_seed: int = 0) -> dict:
    """Selective prediction: keep the most confident predictions, measure the error kept.

    `retained_at_target` is the decision-relevant quantity: the largest fraction of
    compounds that can be kept while mean absolute error stays at or below the target.

    Ties in `sigma` are broken by a seeded permutation rather than by input order. A
    method that returns constant uncertainty would otherwise inherit whatever ordering
    the caller happened to supply, which can flatter it arbitrarily.
    """
    errors = absolute_scores(y_true, y_pred)
    sigma = np.asarray(sigma, float)
    ok = np.isfinite(errors) & np.isfinite(sigma)
    errors, sigma = errors[ok], sigma[ok]
    if errors.size == 0:
        return {"retained_at_target": None, "curve": []}

    jitter = np.random.default_rng(tie_seed).permutation(errors.size)
    order = np.lexsort((jitter, sigma))  # Most confident first, ties broken at random.
    running = np.cumsum(errors[order]) / np.arange(1, errors.size + 1)
    fractions = np.arange(1, errors.size + 1) / errors.size

    within = np.flatnonzero(running <= target_error)
    retained = float(fractions[within[-1]]) if within.size else 0.0
    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    curve = [{"coverage": g,
              "mae": float(running[min(int(np.ceil(g * errors.size)) - 1, errors.size - 1)])}
             for g in grid]
    return {"retained_at_target": retained, "target_error": target_error, "curve": curve}


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    residual = y_true - y_pred
    spearman = (float(stats.spearmanr(y_true, y_pred).statistic)
                if y_true.size >= 3 and np.unique(y_pred).size > 1 else None)
    return {
        "n": int(y_true.size),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": spearman,
    }
