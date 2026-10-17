"""Primary analysis: does calibration on a simulated split transfer to the real split?

Quantities and rules are fixed by the analysis plan. The only thing decided here is how
to resample, and one point needs stating plainly: the plan speaks of a "paired"
bootstrap, but the two coverage values come from disjoint molecule sets (a simulated
held-out set and the real prospective test set), so pairing at the molecule level is not
defined. Each side is therefore resampled independently and the difference is taken,
which is the standard treatment for a difference of two proportions.
"""

from __future__ import annotations

import numpy as np

VERDICT_TRANSFERS = "transfers"
VERDICT_FAILS = "fails_to_transfer"
VERDICT_INCONCLUSIVE = "inconclusive"


def bootstrap_difference(simulated_hits: np.ndarray, real_hits: np.ndarray,
                         resamples: int, seed: int) -> tuple[float, float, float]:
    """Difference in coverage with a percentile interval, in percentage points.

    `*_hits` hold per-molecule coverage indicators; the real side may hold values in
    [0, 1] when several size-matched training subsets are averaged.
    """
    simulated_hits = np.asarray(simulated_hits, dtype=float)
    real_hits = np.asarray(real_hits, dtype=float)
    if simulated_hits.size == 0 or real_hits.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    simulated_draws = rng.integers(0, simulated_hits.size,
                                   size=(resamples, simulated_hits.size))
    real_draws = rng.integers(0, real_hits.size, size=(resamples, real_hits.size))
    differences = (simulated_hits[simulated_draws].mean(axis=1)
                   - real_hits[real_draws].mean(axis=1)) * 100.0
    observed = (simulated_hits.mean() - real_hits.mean()) * 100.0
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(observed), float(low), float(high)


def verdict(low: float, high: float, margin: float) -> str:
    """Three-way reading of an interval against the prespecified equivalence margin."""
    if not np.isfinite(low) or not np.isfinite(high):
        return VERDICT_INCONCLUSIVE
    if low >= -margin and high <= margin:
        return VERDICT_TRANSFERS          # Wholly inside the margin.
    if low > margin or high < -margin:
        return VERDICT_FAILS              # Wholly outside, on one side.
    return VERDICT_INCONCLUSIVE           # Straddles a boundary.


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Return which hypotheses are rejected at the given false discovery rate."""
    values = np.asarray(p_values, dtype=float)
    n = values.size
    if n == 0:
        return []
    order = np.argsort(values)
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passing = values[order] <= thresholds
    rejected = np.zeros(n, dtype=bool)
    if passing.any():
        # Everything up to the largest passing rank is rejected.
        rejected[order[:np.flatnonzero(passing)[-1] + 1]] = True
    return rejected.tolist()


def equivalence_p_value(differences: np.ndarray, margin: float) -> float:
    """Bootstrap p-value for the null that the gap lies inside the equivalence margin.

    Computed as twice the smaller tail mass beyond the nearer margin, so a small value
    means the gap is outside the margin rather than merely different from zero.
    """
    differences = np.asarray(differences, dtype=float)
    if differences.size == 0:
        return float("nan")
    beyond_upper = float((differences <= margin).mean())
    beyond_lower = float((differences >= -margin).mean())
    return float(min(1.0, 2.0 * min(beyond_upper, beyond_lower)))


def summarize_verdicts(verdicts: list[str]) -> dict[str, int]:
    return {name: sum(v == name for v in verdicts)
            for name in (VERDICT_TRANSFERS, VERDICT_FAILS, VERDICT_INCONCLUSIVE)}


def hypothesis_outcome(per_split: dict[str, dict[str, int]],
                       structure_splits: tuple[str, ...],
                       n_endpoints: int) -> str:
    """Apply the prespecified rule to the per-split verdict counts.

    Supported if every structure-based split fails on at least half the endpoints;
    rejected if any achieves transfer on at least three quarters; otherwise mixed.
    """
    fail_threshold = n_endpoints // 2
    transfer_threshold = int(np.ceil(n_endpoints * 0.75))
    counts = [per_split[name] for name in structure_splits if name in per_split]
    if not counts:
        return "not_evaluable"
    if any(c[VERDICT_TRANSFERS] >= transfer_threshold for c in counts):
        return "rejected"
    if all(c[VERDICT_FAILS] >= fail_threshold for c in counts):
        return "supported"
    return "mixed"
