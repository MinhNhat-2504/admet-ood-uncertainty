"""Step 4: open the real prospective test set and run the prespecified comparisons.

This is the irreversible step. It requires the step-3 results to exist, so the simulated
side is fixed before the real side is seen, and it refuses to overwrite its own output.
Training-set size is matched to each simulated split, so a gap reflects shift rather
than how much data the model had.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import modeling, transfer as tr, uncertainty as unc
from .characterize_splits import endpoints, scaled
from .features import featurize

STRUCTURE_SPLITS = ("scaffold", "butina", "frontier", "size")
PRIMARY_METHOD = "split_conformal"


def coverage_indicators(y_true: np.ndarray, centre: np.ndarray,
                        half_width: np.ndarray) -> np.ndarray:
    return ((y_true >= centre - half_width) & (y_true <= centre + half_width)).astype(float)


def evaluate_real(train: pd.DataFrame, test: pd.DataFrame, endpoint: str,
                  n_fit: int, n_calibration: int, seeds: list[int],
                  target_error: float) -> dict | None:
    """Fit on size-matched random subsets of training and score on the real test set."""
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
    per_method_hits: dict[str, list[np.ndarray]] = {m: [] for m in modeling.METHODS}
    per_method_extra: dict[str, list[dict]] = {m: [] for m in modeling.METHODS}
    point_records, repeats = [], []

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
        ranking = modeling.sigmas(evaluation)
        point_records.append(unc.point_metrics(y_test, evaluation.mean))
        repeats.append(float(evaluation.similarity.mean()))

        for method in modeling.METHODS:
            per_method_hits[method].append(
                coverage_indicators(y_test, evaluation.mean, widths[method]))
            per_method_extra[method].append({
                "error_ranking_spearman": unc.error_ranking(y_test, evaluation.mean,
                                                            ranking[method]),
                "retained_at_target": unc.risk_coverage(
                    y_test, evaluation.mean, ranking[method],
                    target_error)["retained_at_target"],
            })

    def average(values: list[float | None]) -> float | None:
        present = [v for v in values if v is not None]
        return round(float(np.mean(present)), 4) if present else None

    return {
        "n_fit": int(n_fit),
        "n_calibration": int(n_calibration),
        "n_eval": int(test_index.size),
        "repeats": len(seeds),
        "mean_similarity_to_training": round(float(np.mean(repeats)), 4),
        "point": {key: average([r[key] for r in point_records])
                  for key in ("rmse", "mae", "spearman")},
        "methods": {
            method: {
                # Averaged per-molecule indicators: subset noise is averaged out while
                # molecule-level variation is kept for the bootstrap.
                "coverage_indicators": np.mean(per_method_hits[method], axis=0),
                "coverage": round(float(np.mean(per_method_hits[method])), 4),
                "error_ranking_spearman": average(
                    [e["error_ranking_spearman"] for e in per_method_extra[method]]),
                "retained_at_target": average(
                    [e["retained_at_target"] for e in per_method_extra[method]]),
            }
            for method in modeling.METHODS
        },
    }


def simulated_indicators(coverage: float, n: int) -> np.ndarray:
    """Reconstruct per-molecule indicators from a stored coverage and sample size."""
    hits = int(round(coverage * n))
    return np.concatenate([np.ones(hits), np.zeros(n - hits)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path,
                        default=Path("data/raw/expansionrx/expansion_data_train.csv"))
    parser.add_argument("--test", type=Path,
                        default=Path("data/raw/expansionrx/expansion_data_test.csv"))
    parser.add_argument("--simulated", type=Path,
                        default=Path("results/simulated_split_results.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/transfer_analysis.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite the primary analysis; amend the plan instead")
    if not args.simulated.is_file():
        parser.error("Run the simulated splits first; the real set must be opened last")

    from protocol.lock import SETTINGS, verify
    locked = verify()
    simulated = json.loads(args.simulated.read_text(encoding="utf-8"))
    train, test = pd.read_csv(args.train), pd.read_csv(args.test)

    eligible = [e for e in endpoints(train)
                if train[e].notna().sum() >= SETTINGS["eligibility"]["min_train_rows"]
                and test[e].notna().sum() >= SETTINGS["eligibility"]["min_test_rows"]]
    margin = SETTINGS["equivalence_margin_pp"]

    report = {
        "status": "primary_transfer_analysis_real_test_set_opened_once",
        "lock_created_utc": locked["created_utc"],
        "inputs": {str(p): sha256(p.read_bytes()).hexdigest()
                   for p in (args.train, args.test, args.simulated)},
        "eligible_endpoints": eligible,
        "equivalence_margin_pp": margin,
        "bootstrap_note": ("The two coverage values come from disjoint molecule sets, so "
                           "each side is resampled independently and the difference taken."),
        "comparisons": [],
    }

    real_cache: dict[tuple[str, int, int], dict] = {}
    for split_name, per_endpoint in simulated["results"].items():
        for endpoint in eligible:
            outcome = per_endpoint.get(endpoint)
            if not outcome or "skipped" in outcome:
                continue
            key = (endpoint, outcome["n_fit"], outcome["n_calibration"])
            if key not in real_cache:
                real_cache[key] = evaluate_real(
                    train, test, endpoint, outcome["n_fit"], outcome["n_calibration"],
                    SETTINGS["model_seeds"], SETTINGS["target_absolute_error"])
            real = real_cache[key]
            if real is None:
                continue

            row = {"split": split_name, "endpoint": endpoint,
                   "n_simulated_eval": outcome["n_eval"], "n_real_eval": real["n_eval"],
                   "n_fit": outcome["n_fit"],
                   "point_rmse_simulated": round(outcome["point"]["rmse"], 4),
                   "point_rmse_real": real["point"]["rmse"],
                   "methods": {}}

            for method in modeling.METHODS:
                simulated_coverage = outcome["methods"][method]["coverage"]
                simulated_hits = simulated_indicators(simulated_coverage,
                                                      outcome["n_eval"])
                real_hits = real["methods"][method]["coverage_indicators"]
                observed, low, high = tr.bootstrap_difference(
                    simulated_hits, real_hits, SETTINGS["bootstrap_resamples"],
                    SETTINGS["bootstrap_seed"])

                rng = np.random.default_rng(SETTINGS["bootstrap_seed"])
                draws = ((simulated_hits[rng.integers(0, simulated_hits.size,
                                                      (2000, simulated_hits.size))].mean(axis=1)
                          - real_hits[rng.integers(0, real_hits.size,
                                                   (2000, real_hits.size))].mean(axis=1)) * 100)
                row["methods"][method] = {
                    "coverage_simulated": round(simulated_coverage, 4),
                    "coverage_real": real["methods"][method]["coverage"],
                    "gap_pp": round(observed, 3),
                    "ci95_pp": [round(low, 3), round(high, 3)],
                    "verdict_nominal": tr.verdict(low, high, margin),
                    "equivalence_p": round(tr.equivalence_p_value(draws, margin), 6),
                    "error_ranking_simulated":
                        outcome["methods"][method]["error_ranking_spearman"],
                    "error_ranking_real":
                        real["methods"][method]["error_ranking_spearman"],
                    "retained_at_target_simulated":
                        outcome["methods"][method]["risk_coverage"]["retained_at_target"],
                    "retained_at_target_real":
                        real["methods"][method]["retained_at_target"],
                }
            report["comparisons"].append(row)
            primary = row["methods"][PRIMARY_METHOD]
            print(f"{split_name:10s} {endpoint:30s} sim={primary['coverage_simulated']:.3f} "
                  f"real={primary['coverage_real']:.3f} gap={primary['gap_pp']:+6.1f}pp "
                  f"{primary['verdict_nominal']}")

    # Multiplicity control across the primary family only.
    primary_rows = [r["methods"][PRIMARY_METHOD] for r in report["comparisons"]]
    rejected = tr.benjamini_hochberg([r["equivalence_p"] for r in primary_rows],
                                     alpha=0.05)
    for row, is_rejected in zip(primary_rows, rejected):
        nominal = row["verdict_nominal"]
        row["verdict_adjusted"] = (
            nominal if nominal != tr.VERDICT_FAILS
            else (tr.VERDICT_FAILS if is_rejected else tr.VERDICT_INCONCLUSIVE))

    per_split: dict[str, dict[str, int]] = {}
    for split_name in simulated["results"]:
        verdicts = [r["methods"][PRIMARY_METHOD]["verdict_adjusted"]
                    for r in report["comparisons"] if r["split"] == split_name]
        if verdicts:
            per_split[split_name] = tr.summarize_verdicts(verdicts)

    report["primary_summary"] = per_split
    report["hypothesis_outcome"] = tr.hypothesis_outcome(
        per_split, STRUCTURE_SPLITS, len(eligible))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=float) + "\n",
                           encoding="utf-8")
    print("\nPrimary verdicts after multiplicity control:")
    for split_name, counts in per_split.items():
        print(f"  {split_name:10s} {counts}")
    print(f"\nPrespecified outcome: {report['hypothesis_outcome'].upper()}")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
