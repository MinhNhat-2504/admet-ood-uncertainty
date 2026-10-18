"""Audit how much of each method's coverage comes from unbounded intervals.

A conformal interval is infinite when its calibration stratum holds too few points for the
finite-sample quantile. That is honest behaviour — the method is declining to make a
claim — but an infinite interval covers every value, so a coverage figure that includes
them overstates what the method delivers. The primary run recorded this count on the
simulated side and not on the real side; this fills the gap for the real split.

Descriptive only: no verdict, margin or outcome rule applies.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import modeling, uncertainty as unc
from .characterize_splits import scaled
from .features import featurize


def audit_endpoint(train: pd.DataFrame, test: pd.DataFrame, endpoint: str, n_fit: int,
                   n_calibration: int, seeds: list[int],
                   features: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                   ) -> dict | None:
    """Fraction of infinite intervals, and coverage with and without them, per method."""
    train_features, train_ok, test_features, test_ok = features
    train_labels = scaled(train[endpoint], endpoint).to_numpy(dtype=float)
    test_labels = scaled(test[endpoint], endpoint).to_numpy(dtype=float)

    train_pool = np.flatnonzero(np.isfinite(train_labels) & train_ok)
    test_index = np.flatnonzero(np.isfinite(test_labels) & test_ok)
    needed = n_fit + n_calibration
    if train_pool.size < needed or test_index.size < unc.MIN_CALIBRATION:
        return None

    y_test = test_labels[test_index]
    infinite: dict[str, list[float]] = {m: [] for m in modeling.METHODS}
    covered: dict[str, list[float]] = {m: [] for m in modeling.METHODS}
    covered_finite: dict[str, list[float]] = {m: [] for m in modeling.METHODS}

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

        widths = modeling.half_widths(calibration, train_labels[calibration_index],
                                      evaluation)
        for method, width in widths.items():
            width = np.asarray(width, dtype=float)
            is_infinite = ~np.isfinite(width)
            inside = ((y_test >= evaluation.mean - width)
                      & (y_test <= evaluation.mean + width))
            infinite[method].append(float(is_infinite.mean()))
            covered[method].append(float(inside.mean()))
            # Coverage over the molecules the method actually bounded.
            finite = ~is_infinite
            covered_finite[method].append(
                float(inside[finite].mean()) if finite.any() else float("nan"))

    def mean(values: list[float]) -> float | None:
        clean = [v for v in values if np.isfinite(v)]
        return round(float(np.mean(clean)), 4) if clean else None

    return {
        "n_eval": int(test_index.size),
        "methods": {
            method: {
                "infinite_fraction": mean(infinite[method]),
                "coverage_all": mean(covered[method]),
                "coverage_bounded_only": mean(covered_finite[method]),
            }
            for method in modeling.METHODS
        },
    }


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
                        default=Path("results/interval_audit.json"))
    args = parser.parse_args()
    for required in (args.transfer, args.simulated):
        if not required.is_file():
            parser.error(f"Missing {required}; run the primary analysis first")

    from protocol.lock import SETTINGS, verify
    locked = verify()
    transfer = json.loads(args.transfer.read_text(encoding="utf-8"))
    simulated = json.loads(args.simulated.read_text(encoding="utf-8"))
    train, test = pd.read_csv(args.train), pd.read_csv(args.test)

    sizes: dict[str, tuple[int, int]] = {}
    for per_endpoint in simulated["results"].values():
        for endpoint, outcome in per_endpoint.items():
            if outcome and "skipped" not in outcome and endpoint not in sizes:
                sizes[endpoint] = (outcome["n_fit"], outcome["n_calibration"])

    report = {
        "status": "descriptive_interval_audit_real_split",
        "note": ("Infinite conformal intervals cover every value. Coverage that includes "
                 "them overstates what a method delivers, so both are reported."),
        "lock_created_utc": locked["created_utc"],
        "inputs": {str(p): sha256(p.read_bytes()).hexdigest()
                   for p in (args.train, args.test)},
        "endpoints": {},
    }

    features = (*featurize(list(train["SMILES"])), *featurize(list(test["SMILES"])))
    for endpoint in transfer["eligible_endpoints"]:
        if endpoint not in sizes:
            continue
        result = audit_endpoint(train, test, endpoint, *sizes[endpoint],
                                SETTINGS["model_seeds"], features)
        if result is None:
            continue
        report["endpoints"][endpoint] = result
        row = result["methods"]["mondrian_conformal"]
        print(f"{endpoint:32s} mondrian infinite={row['infinite_fraction']:.3f} "
              f"coverage_all={row['coverage_all']:.3f} "
              f"bounded_only={row['coverage_bounded_only']}", flush=True)

    if report["endpoints"]:
        report["summary"] = {
            method: {
                key: round(float(np.mean([
                    e["methods"][method][key] for e in report["endpoints"].values()
                    if e["methods"][method][key] is not None])), 4)
                for key in ("infinite_fraction", "coverage_all", "coverage_bounded_only")
            }
            for method in modeling.METHODS
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
