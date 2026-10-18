"""Exploratory robustness check: does the coverage failure survive a second model family?

The protocol fixed one learner, gradient-boosted trees over Morgan counts and RDKit
descriptors, so that differences between uncertainty methods could not be attributed to
features. That leaves open whether the coverage collapse under prospective shift is a
property of the shift or of that learner. This refits the real-split comparison with a
neural network instead and reports the two side by side.

This is post hoc. It was not preregistered, no verdict, margin or outcome rule attaches to
it, and it writes a separate file so the primary analysis is untouched. It is reported as
an exploratory robustness check and must be labelled that way wherever it is quoted.

Quantile regression is excluded: it needs a pinball-loss learner, and substituting a
gradient-boosted one would reintroduce the family under test. The four remaining methods
cover every headline number in Sections 5.1 and 5.2.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from . import modeling, uncertainty as unc
from .characterize_splits import endpoints, scaled
from .features import featurize

# Methods needing only a mean, an ensemble spread and a distance to training.
METHODS = ("ensemble", "split_conformal", "mondrian_conformal", "distance")
PRIMARY_METHOD = "split_conformal"
MLP_PARAMS = {
    "hidden_layer_sizes": (128, 32),
    "activation": "relu",
    "alpha": 1e-3,
    "learning_rate_init": 1e-3,
    "max_iter": 400,   # Early stopping ends training well before this; 120 gave identical results.
    "early_stopping": True,
    "n_iter_no_change": 8,
    "tol": 1e-3,
}
# The primary analysis averages over five size-matched training subsets. A network is far
# more expensive to fit than a boosted tree, so this check uses three. Fewer repeats makes
# each coverage estimate noisier; it does not bias it, and the ensemble keeps all five
# members so the ensemble-spread comparison stays like for like.
REPEATS = 3


def fit_predict_mlp(x_fit: np.ndarray, y_fit: np.ndarray, x_eval: list[np.ndarray],
                    seeds: list[int]) -> list[modeling.Prediction]:
    """A neural-network ensemble shaped like `modeling.fit_predict`.

    Features are standardized on the fit set, which trees do not need and networks do.
    The quantile arms are left undefined; the caller must not read them.
    """
    scaler = StandardScaler().fit(x_fit)
    x_fit_scaled = scaler.transform(x_fit)
    members = [MLPRegressor(random_state=seed, **MLP_PARAMS).fit(x_fit_scaled, y_fit)
               for seed in seeds]

    outputs = []
    for x in x_eval:
        x_scaled = scaler.transform(x)
        stacked = np.vstack([m.predict(x_scaled) for m in members])
        undefined = np.full(x.shape[0], np.nan)
        outputs.append(modeling.Prediction(
            mean=stacked.mean(axis=0),
            ensemble_sd=stacked.std(axis=0, ddof=1),
            quantile_low=undefined,
            quantile_high=undefined,
            similarity=np.zeros(x.shape[0]),
        ))
    return outputs


def evaluate(train: pd.DataFrame, test: pd.DataFrame, endpoint: str, n_fit: int,
             n_calibration: int, seeds: list[int],
             features: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
             ) -> dict | None:
    """Real-split coverage under the second model family, size-matched as before.

    Featurization is passed in: it depends only on the SMILES, so recomputing it per
    endpoint would repeat the same work over all 7,608 molecules eight times.
    """
    train_features, train_ok, test_features, test_ok = features
    train_labels = scaled(train[endpoint], endpoint).to_numpy(dtype=float)
    test_labels = scaled(test[endpoint], endpoint).to_numpy(dtype=float)

    train_pool = np.flatnonzero(np.isfinite(train_labels) & train_ok)
    test_index = np.flatnonzero(np.isfinite(test_labels) & test_ok)
    needed = n_fit + n_calibration
    if train_pool.size < needed or test_index.size < unc.MIN_CALIBRATION:
        return None

    y_test = test_labels[test_index]
    hits: dict[str, list[np.ndarray]] = {m: [] for m in METHODS}
    point: list[dict] = []

    for seed in seeds[:REPEATS]:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(train_pool, size=needed, replace=False)
        calibration_index, fit_index = chosen[:n_calibration], chosen[n_calibration:]

        calibration, evaluation = fit_predict_mlp(
            train_features[fit_index], train_labels[fit_index],
            [train_features[calibration_index], test_features[test_index]], seeds)

        fit_smiles = train["SMILES"].to_numpy()[fit_index].tolist()
        calibration.similarity = modeling.neighbour_similarity(
            fit_smiles, train["SMILES"].to_numpy()[calibration_index].tolist())
        evaluation.similarity = modeling.neighbour_similarity(
            fit_smiles, test["SMILES"].to_numpy()[test_index].tolist())

        widths = modeling.half_widths(calibration, train_labels[calibration_index],
                                      evaluation)
        point.append(unc.point_metrics(y_test, evaluation.mean))
        for method in METHODS:
            inside = ((y_test >= evaluation.mean - widths[method])
                      & (y_test <= evaluation.mean + widths[method]))
            hits[method].append(inside.astype(float))

    return {
        "n_fit": int(n_fit),
        "n_eval": int(test_index.size),
        "repeats": len(seeds[:REPEATS]),
        "ensemble_members": len(seeds),
        "point": {k: round(float(np.mean([r[k] for r in point if r[k] is not None])), 4)
                  for k in ("rmse", "mae")},
        "coverage": {m: round(float(np.mean(hits[m])), 4) for m in METHODS},
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
                        default=Path("results/model_family_robustness.json"))
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

    # The primary learner's numbers, averaged over the size-matched splits as elsewhere.
    reference = {
        endpoint: {
            method: round(float(np.mean(
                [r["methods"][method]["coverage_real"]
                 for r in transfer["comparisons"] if r["endpoint"] == endpoint])), 4)
            for method in METHODS}
        for endpoint in transfer["eligible_endpoints"]}

    report = {
        "status": "exploratory_second_model_family_not_preregistered",
        "note": ("Post hoc robustness check. No verdict, margin or outcome rule applies. "
                 "Quantile regression is excluded because it needs a pinball-loss "
                 "learner from the family under test."),
        "lock_created_utc": locked["created_utc"],
        "inputs": {str(p): sha256(p.read_bytes()).hexdigest()
                   for p in (args.train, args.test, args.transfer)},
        "primary_family": "lightgbm",
        "second_family": "sklearn_mlp",
        "second_family_params": MLP_PARAMS,
        "endpoints": {},
    }

    print("Featurizing once for all endpoints...", flush=True)
    features = (*featurize(list(train["SMILES"])), *featurize(list(test["SMILES"])))

    for endpoint in transfer["eligible_endpoints"]:
        if endpoint not in sizes:
            continue
        n_fit, n_calibration = sizes[endpoint]
        print(f"  fitting {endpoint}...", flush=True)
        result = evaluate(train, test, endpoint, n_fit, n_calibration,
                          SETTINGS["model_seeds"], features)
        if result is None:
            continue
        result["coverage_primary_family"] = reference[endpoint]
        result["difference_pp"] = {
            m: round((result["coverage"][m] - reference[endpoint][m]) * 100, 2)
            for m in METHODS}
        report["endpoints"][endpoint] = result
        print(f"{endpoint:32s} conformal mlp={result['coverage'][PRIMARY_METHOD]:.3f} "
              f"lgbm={reference[endpoint][PRIMARY_METHOD]:.3f} "
              f"diff={result['difference_pp'][PRIMARY_METHOD]:+6.1f}pp  "
              f"ensemble mlp={result['coverage']['ensemble']:.3f}")

    if report["endpoints"]:
        report["summary"] = {
            method: {
                "mean_coverage_second_family": round(float(np.mean(
                    [e["coverage"][method] for e in report["endpoints"].values()])), 4),
                "mean_coverage_primary_family": round(float(np.mean(
                    [e["coverage_primary_family"][method]
                     for e in report["endpoints"].values()])), 4),
            }
            for method in METHODS}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
