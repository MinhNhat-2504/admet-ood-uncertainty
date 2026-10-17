"""Step 3 of the analysis plan: fit and score on the simulated splits only.

This script reads the training file and nothing else. The real prospective test set stays
closed until step 4, so nothing here can be tuned against it.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import modeling, splits as split_module, uncertainty as unc
from .characterize_splits import ID_COLUMNS, endpoints, scaled
from .features import featurize

CALIBRATION_FRACTION = 0.25  # Of the simulated training part, held out to calibrate.


def evaluate_once(frame: pd.DataFrame, endpoint: str, held_out: np.ndarray,
                  seeds: list[int], target_error: float, rng_seed: int) -> dict:
    """Fit on the simulated training part and score on the simulated held-out part."""
    labels = scaled(frame[endpoint], endpoint).to_numpy(dtype=float)
    features, featurizable = featurize(list(frame["SMILES"]))
    usable = np.isfinite(labels) & featurizable

    train_mask = usable & ~held_out
    eval_mask = usable & held_out
    if train_mask.sum() < 50 or eval_mask.sum() < unc.MIN_CALIBRATION:
        return {"skipped": "insufficient_rows",
                "n_train": int(train_mask.sum()), "n_eval": int(eval_mask.sum())}

    # Calibration is a random holdout of the training part: what a practitioner has.
    train_index = np.flatnonzero(train_mask)
    shuffled = np.random.default_rng(rng_seed).permutation(train_index)
    n_calibration = max(int(round(len(shuffled) * CALIBRATION_FRACTION)),
                        unc.MIN_CALIBRATION)
    calibration_index, fit_index = shuffled[:n_calibration], shuffled[n_calibration:]
    eval_index = np.flatnonzero(eval_mask)
    if len(fit_index) < 50:
        return {"skipped": "insufficient_fit_rows", "n_train": int(train_mask.sum())}

    started = time.perf_counter()
    calibration, evaluation = modeling.fit_predict(
        features[fit_index], labels[fit_index],
        [features[calibration_index], features[eval_index]], seeds)

    fit_smiles = frame["SMILES"].to_numpy()[fit_index].tolist()
    calibration.similarity = modeling.neighbour_similarity(
        fit_smiles, frame["SMILES"].to_numpy()[calibration_index].tolist())
    evaluation.similarity = modeling.neighbour_similarity(
        fit_smiles, frame["SMILES"].to_numpy()[eval_index].tolist())

    y_calibration, y_eval = labels[calibration_index], labels[eval_index]
    widths = modeling.half_widths(calibration, y_calibration, evaluation)
    ranking = modeling.sigmas(evaluation)

    result = {
        "n_fit": int(len(fit_index)),
        "n_calibration": int(len(calibration_index)),
        "n_eval": int(len(eval_index)),
        "seconds": round(time.perf_counter() - started, 1),
        "point": unc.point_metrics(y_eval, evaluation.mean),
        "mean_similarity_to_training": round(float(evaluation.similarity.mean()), 4),
        "methods": {},
    }
    for method in modeling.METHODS:
        result["methods"][method] = {
            **unc.interval_metrics(y_eval, evaluation.mean, widths[method]),
            "error_ranking_spearman": unc.error_ranking(y_eval, evaluation.mean,
                                                        ranking[method]),
            "risk_coverage": unc.risk_coverage(y_eval, evaluation.mean, ranking[method],
                                               target_error),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path,
                        default=Path("data/raw/expansionrx/expansion_data_train.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/simulated_split_results.json"))
    parser.add_argument("--splits", nargs="*", default=None)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing simulated-split results")

    from protocol.lock import SETTINGS, verify
    locked = verify()

    train = pd.read_csv(args.train)
    smiles = list(train["SMILES"])
    chosen = args.splits or SETTINGS["simulated_splits"]

    report = {
        "status": "simulated_splits_only_real_test_set_not_opened",
        "lock_created_utc": locked["created_utc"],
        "input_sha256": sha256(args.train.read_bytes()).hexdigest(),
        "settings": {k: SETTINGS[k] for k in
                     ("model_seeds", "split_seed", "test_fraction",
                      "nominal_coverage", "target_absolute_error")},
        "results": {},
    }

    for split_name in chosen:
        mask = split_module.build(split_name, smiles, SETTINGS["test_fraction"],
                                  SETTINGS["split_seed"])
        report["results"][split_name] = {}
        for endpoint in endpoints(train):
            outcome = evaluate_once(train, endpoint, mask, SETTINGS["model_seeds"],
                                    SETTINGS["target_absolute_error"],
                                    SETTINGS["split_seed"])
            report["results"][split_name][endpoint] = outcome
            if "skipped" in outcome:
                print(f"{split_name:10s} {endpoint:30s} skipped ({outcome['skipped']})")
                continue
            conformal = outcome["methods"]["split_conformal"]["coverage"]
            print(f"{split_name:10s} {endpoint:30s} n={outcome['n_eval']:5d} "
                  f"rmse={outcome['point']['rmse']:.3f} "
                  f"conformal_coverage={conformal:.3f} ({outcome['seconds']:.0f}s)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
