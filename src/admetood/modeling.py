"""Fit the prespecified models and turn each uncertainty signal into a 90% interval.

Every method must deliver two things: a per-molecule uncertainty used for ranking and
selective prediction, and an interval whose empirical coverage can be compared against
the 90% nominal level. Methods that produce only one of the two are not comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from lightgbm import LGBMRegressor
from rdkit import DataStructs

from . import uncertainty as unc
from .splits import fingerprints

# Gaussian z for a central 90% interval, used by methods that report a standard deviation.
Z_90 = 1.6448536269514722
QUANTILES = (0.05, 0.95)
NEIGHBOURS = 5
LGBM_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
               "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
               "colsample_bytree": 0.6, "verbose": -1, "n_jobs": -1}


@dataclass
class Prediction:
    """Model output on one evaluation set."""

    mean: np.ndarray
    ensemble_sd: np.ndarray
    quantile_low: np.ndarray
    quantile_high: np.ndarray
    similarity: np.ndarray  # Mean Tanimoto to the k nearest training molecules.
    extras: dict = field(default_factory=dict)


def neighbour_similarity(reference_smiles: list[str], query_smiles: list[str],
                         k: int = NEIGHBOURS) -> np.ndarray:
    """Mean Tanimoto to the k most similar reference molecules; high means well covered."""
    reference_fps = [fp for fp in fingerprints(reference_smiles) if fp is not None]
    query_fps = fingerprints(query_smiles)
    scores = np.zeros(len(query_smiles))
    if not reference_fps:
        return scores
    for index, fp in enumerate(query_fps):
        if fp is None:
            continue  # Unfeaturizable molecules score 0: maximally far from training.
        similarity = np.sort(DataStructs.BulkTanimotoSimilarity(fp, reference_fps))[::-1]
        scores[index] = float(similarity[:k].mean())
    return scores


def fit_predict(x_fit: np.ndarray, y_fit: np.ndarray, x_eval: list[np.ndarray],
                seeds: list[int]) -> list[Prediction]:
    """Train the ensemble and quantile models once, predict on several evaluation sets."""
    members = []
    for seed in seeds:
        model = LGBMRegressor(random_state=seed, **LGBM_PARAMS)
        model.fit(x_fit, y_fit)
        members.append(model)

    quantile_models = {}
    for alpha in QUANTILES:
        model = LGBMRegressor(objective="quantile", alpha=alpha,
                              random_state=seeds[0], **LGBM_PARAMS)
        model.fit(x_fit, y_fit)
        quantile_models[alpha] = model

    outputs = []
    for x in x_eval:
        stacked = np.vstack([m.predict(x) for m in members])
        outputs.append(Prediction(
            mean=stacked.mean(axis=0),
            # ddof=1: an unbiased spread estimate from a small ensemble.
            ensemble_sd=stacked.std(axis=0, ddof=1),
            quantile_low=quantile_models[QUANTILES[0]].predict(x),
            quantile_high=quantile_models[QUANTILES[1]].predict(x),
            similarity=np.zeros(x.shape[0]),
        ))
    return outputs


def half_widths(calibration: Prediction, y_calibration: np.ndarray,
                evaluation: Prediction) -> dict[str, np.ndarray]:
    """Half-width of the 90% interval on the evaluation set, per uncertainty method.

    Conformal variants calibrate on held-out residuals; the Gaussian variants do not and
    are included because they are what most published pipelines actually report.
    """
    widths: dict[str, np.ndarray] = {}

    # Distribution-free but constant width: every molecule receives the same interval.
    absolute = unc.absolute_scores(y_calibration, calibration.mean)
    widths["split_conformal"] = np.full(evaluation.mean.shape,
                                        unc.conformal_quantile(absolute))

    # Conformal conditional on distance to training, the prespecified mitigation.
    bins_calibration = unc.similarity_bins(calibration.similarity)
    bins_evaluation = unc.similarity_bins(evaluation.similarity)
    per_bin = unc.mondrian_quantiles(absolute, bins_calibration)
    pooled = unc.conformal_quantile(absolute)
    widths["mondrian_conformal"] = np.array(
        # A bin unseen during calibration falls back to the pooled quantile.
        [per_bin.get(int(b), pooled) for b in bins_evaluation])

    # Uncalibrated Gaussian interval from the ensemble spread.
    widths["ensemble"] = Z_90 * evaluation.ensemble_sd

    # Direct quantile regression; asymmetric, so the half-width is the mean arm.
    widths["quantile"] = np.maximum(
        (evaluation.quantile_high - evaluation.quantile_low) / 2.0, 0.0)

    # Distance to training, conformalized so it becomes an interval rather than a score.
    distance_calibration = 1.0 - calibration.similarity
    distance_evaluation = 1.0 - evaluation.similarity
    scaled = unc.normalized_scores(y_calibration, calibration.mean, distance_calibration)
    widths["distance"] = unc.conformal_quantile(scaled) * distance_evaluation
    return widths


def sigmas(prediction: Prediction) -> dict[str, np.ndarray]:
    """The ranking signal behind each interval, for error-ranking and selective prediction."""
    spread = np.maximum(prediction.quantile_high - prediction.quantile_low, 0.0)
    return {
        "split_conformal": prediction.ensemble_sd,   # Constant width; rank by the model's own spread.
        "mondrian_conformal": 1.0 - prediction.similarity,
        "ensemble": prediction.ensemble_sd,
        "quantile": spread,
        "distance": 1.0 - prediction.similarity,
    }


METHODS = ("ensemble", "quantile", "split_conformal", "mondrian_conformal", "distance")
