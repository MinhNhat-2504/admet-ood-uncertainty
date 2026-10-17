"""Recover the full risk-coverage curves on the real split.

The primary run computed these curves and kept only `retained_at_target`, a scalar at one
prespecified threshold. That threshold saturated on six of eight endpoints, so the scalar
cannot discriminate between methods and Section 6.4 of the write-up had nothing to say
about decision value. The curve does not depend on the threshold and answers the question
the scalar could not.

This re-runs the same size-matched fits as the primary analysis to recover a quantity
that run was already computing. It is not a new hypothesis test: no verdict, margin or
outcome rule is involved, and it writes a separate file so the primary analysis is
untouched.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import modeling, uncertainty as unc
from .characterize_splits import endpoints, scaled
from .features import featurize

PRIMARY_METHOD = "split_conformal"


def real_curves(train: pd.DataFrame, test: pd.DataFrame, endpoint: str,
                n_fit: int, n_calibration: int, seeds: list[int],
                target_error: float) -> dict | None:
    """Risk-coverage curves on the real test set, averaged over size-matched subsets."""
    train_labels = scaled(train[endpoint], endpoint).to_numpy(dtype=float)
    test_labels = scaled(test[endpoint], endpoint).to_numpy(dtype=float)
    train_features, train_ok = featurize(list(train["SMILES"]))
    test_features, test_ok = featurize(list(test["SMILES"]))

    train_pool = np.flatnonzero(np.isfinite(train_labels) & train_ok)
    test_index = np.flatnonzero(np.isfinite(test_labels) & test_ok)
    needed = n_fit + n_calibration
    if train_pool.size < needed or test_index.size < unc.MIN_CALIBRATION:
        return None

    y_test = test_labels[test_index]
    per_method: dict[str, list[list[float]]] = {m: [] for m in modeling.METHODS}
    grid: list[float] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(train_pool, size=needed, replace=False)
        calibration_index, fit_index = chosen[:n_calibration], chosen[n_calibration:]

        calibration, evaluation = modeling.fit_predict(
            train_features[fit_index], train_labels[fit_index],
            [train_features[calibration_index], test_features[test_index]], seeds)

        fit_smiles = train["SMILES"].to_numpy()[fit_index].tolist()
        calibration.similarity = modeling.neighbour_similarity(
            fit_smiles, train["SMILES"].to_numpy()[calibration_index].tolist())
        evaluation.similarity = modeling.neighbour_similarity(
            fit_smiles, test["SMILES"].to_numpy()[test_index].tolist())

        ranking = modeling.sigmas(evaluation)
        for method in modeling.METHODS:
            curve = unc.risk_coverage(y_test, evaluation.mean, ranking[method],
                                      target_error)["curve"]
            per_method[method].append([point["mae"] for point in curve])
            grid = [point["coverage"] for point in curve]

    return {
        "n_eval": int(test_index.size),
        "n_fit": int(n_fit),
        "coverage_grid": grid,
        "methods": {
            method: {
                # Averaged over the size-matched subsets, as the coverage numbers are.
                "mae_at_coverage": [round(float(v), 4)
                                    for v in np.mean(per_method[method], axis=0)],
            }
            for method in modeling.METHODS
        },
    }


def discrimination(curves: dict, method: str, reference: str) -> float | None:
    """How much lower the method's error is than the reference's, averaged over the grid.

    Positive means the method ranks errors better: keeping the same fraction of compounds
    leaves less error. This is the comparison the saturated scalar could not make.
    """
    a = curves["methods"].get(method, {}).get("mae_at_coverage")
    b = curves["methods"].get(reference, {}).get("mae_at_coverage")
    if not a or not b:
        return None
    return round(float(np.mean(np.asarray(b) - np.asarray(a))), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path,
                        default=Path("data/raw/expansionrx/expansion_data_train.csv"))
    parser.add_argument("--test", type=Path,
                        default=Path("data/raw/expansionrx/expansion_data_test.csv"))
    parser.add_argument("--transfer", type=Path,
                        default=Path("results/transfer_analysis.json"))
    parser.add_argument("--simulated", type=Path,
                        default=Path("results/simulated_split_results.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/risk_coverage_curves.json"))
    args = parser.parse_args()
    for required in (args.transfer, args.simulated):
        if not required.is_file():
            parser.error(f"Missing {required}; run the primary analysis first")

    from protocol.lock import SETTINGS, verify
    locked = verify()
    transfer = json.loads(args.transfer.read_text(encoding="utf-8"))
    simulated = json.loads(args.simulated.read_text(encoding="utf-8"))
    train, test = pd.read_csv(args.train), pd.read_csv(args.test)

    # Take the fit and calibration sizes from the same place the primary run took them,
    # so these curves describe those models and not differently sized ones.
    sizes: dict[str, tuple[int, int]] = {}
    for per_endpoint in simulated["results"].values():
        for endpoint, outcome in per_endpoint.items():
            if outcome and "skipped" not in outcome and endpoint not in sizes:
                sizes[endpoint] = (outcome["n_fit"], outcome["n_calibration"])

    report = {
        "status": "secondary_risk_coverage_curves_real_split",
        "note": ("Recovers curves the primary run computed and discarded. No verdict, "
                 "margin or outcome rule is applied here."),
        "lock_created_utc": locked["created_utc"],
        "inputs": {str(p): sha256(p.read_bytes()).hexdigest()
                   for p in (args.train, args.test, args.transfer)},
        "target_error": SETTINGS["target_absolute_error"],
        "endpoints": {},
    }

    for endpoint in transfer["eligible_endpoints"]:
        if endpoint not in sizes:
            continue
        n_fit, n_calibration = sizes[endpoint]
        curves = real_curves(train, test, endpoint, n_fit, int(n_calibration),
                             SETTINGS["model_seeds"], SETTINGS["target_absolute_error"])
        if curves is None:
            continue
        curves["versus_ensemble"] = {
            method: discrimination(curves, method, "ensemble")
            for method in modeling.METHODS if method != "ensemble"}
        report["endpoints"][endpoint] = curves
        primary = curves["methods"][PRIMARY_METHOD]["mae_at_coverage"]
        print(f"{endpoint:32s} MAE at 20% kept={primary[1]:.3f}  "
              f"at 50%={primary[4]:.3f}  at 100%={primary[-1]:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
