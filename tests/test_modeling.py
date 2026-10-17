"""Every uncertainty method must yield both an interval and a usable ranking signal."""

import numpy as np
import pytest

from admetood import modeling
from admetood.features import FEATURE_NAMES, featurize

SMILES = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCO", "CC(=O)O",
          "c1ccncc1", "CCN", "CCCC", "c1ccc2ccccc2c1", "CC(C)O"]


def toy_prediction(n, seed=0):
    rng = np.random.default_rng(seed)
    return modeling.Prediction(
        mean=rng.normal(size=n),
        ensemble_sd=np.abs(rng.normal(size=n)) + 0.1,
        quantile_low=-np.abs(rng.normal(size=n)) - 1,
        quantile_high=np.abs(rng.normal(size=n)) + 1,
        similarity=rng.uniform(0.2, 0.9, size=n),
    )


def test_featurizer_shape_and_invalid_handling():
    matrix, valid = featurize(SMILES + ["not_a_molecule"])
    assert matrix.shape == (len(SMILES) + 1, len(FEATURE_NAMES))
    assert valid[:-1].all() and not valid[-1]
    assert np.isfinite(matrix).all(), "features must never contain NaN or inf"


def test_descriptor_block_is_not_all_zero():
    """A silent featurizer failure would show up as a constant descriptor block."""
    matrix, _ = featurize(SMILES)
    descriptors = matrix[:, -len(modeling.__dict__.get("DESCRIPTORS", {})) or None:]
    assert matrix[:, 2048:].std(axis=0).sum() > 0


def test_neighbour_similarity_is_one_for_identical_sets():
    scores = modeling.neighbour_similarity(SMILES, SMILES, k=1)
    assert np.allclose(scores, 1.0)


def test_neighbour_similarity_drops_for_unrelated_chemistry():
    aromatic = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1"]
    aliphatic = ["CCCCCCCCCC"]
    assert modeling.neighbour_similarity(aromatic, aliphatic, k=1)[0] < 0.3


def test_unparsable_query_scores_zero_similarity():
    assert modeling.neighbour_similarity(SMILES, ["not_a_molecule"])[0] == 0.0


def test_every_method_produces_finite_half_widths_and_sigmas():
    calibration, evaluation = toy_prediction(200, 1), toy_prediction(50, 2)
    y_calibration = calibration.mean + np.random.default_rng(3).normal(scale=0.3, size=200)
    widths = modeling.half_widths(calibration, y_calibration, evaluation)
    ranking = modeling.sigmas(evaluation)
    assert set(widths) == set(modeling.METHODS)
    assert set(ranking) == set(modeling.METHODS)
    for method in modeling.METHODS:
        assert widths[method].shape == (50,)
        assert np.all(widths[method] >= 0), f"{method} produced a negative width"
        assert np.isfinite(ranking[method]).all()


def test_split_conformal_width_is_constant_and_mondrian_is_not():
    calibration, evaluation = toy_prediction(300, 4), toy_prediction(80, 5)
    y_calibration = calibration.mean + np.random.default_rng(6).normal(scale=0.5, size=300)
    widths = modeling.half_widths(calibration, y_calibration, evaluation)
    assert np.unique(widths["split_conformal"]).size == 1
    # Mondrian conditions on distance, so widths must vary across similarity bins.
    assert np.unique(widths["mondrian_conformal"]).size > 1


def test_fit_predict_returns_one_output_per_evaluation_set():
    matrix, _ = featurize(SMILES * 12)
    y = np.arange(len(SMILES) * 12, dtype=float)
    outputs = modeling.fit_predict(matrix, y, [matrix[:10], matrix[10:20]], seeds=[0, 1, 2])
    assert len(outputs) == 2
    for output in outputs:
        assert output.mean.shape == (10,)
        assert np.isfinite(output.ensemble_sd).all()
        assert np.all(output.quantile_high >= output.quantile_low - 1e-6)
