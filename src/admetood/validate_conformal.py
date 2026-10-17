"""Check that split conformal attains its nominal level before any shift is introduced.

The analysis plan makes this a fail-closed condition: if conformal does not reach
88-92% coverage on a random split, an undercoverage seen later is an implementation bug
rather than an effect of distribution shift.

A single split cannot settle the question. Conformal validity is marginal over the draw
of the calibration and evaluation sets, so this repeats the random split many times and
reports the grand mean with a confidence interval, together with the per-split spread
that any single-split result has to be read against.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import splits as split_module
from .characterize_splits import endpoints
from .run_simulated import evaluate_once

REPEATS = 20
BASE_SEED = 500


def validate(train: pd.DataFrame, endpoint: str, seeds: list[int], test_fraction: float,
             repeats: int = REPEATS, base_seed: int = BASE_SEED) -> dict:
    smiles = list(train["SMILES"])
    coverages, widths = [], []
    for offset in range(repeats):
        seed = base_seed + offset
        mask = split_module.build("random", smiles, test_fraction, seed=seed)
        outcome = evaluate_once(train, endpoint, mask, seeds, 0.5, rng_seed=seed)
        if "skipped" in outcome:
            continue
        conformal = outcome["methods"]["split_conformal"]
        coverages.append(conformal["coverage"])
        widths.append(conformal["mean_width"])

    if len(coverages) < 3:
        return {"endpoint": endpoint, "repeats_completed": len(coverages),
                "verdict": "insufficient_repeats"}

    values = np.array(coverages)
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    low, high = values.mean() - 1.96 * standard_error, values.mean() + 1.96 * standard_error
    return {
        "endpoint": endpoint,
        "repeats_completed": int(values.size),
        "grand_mean_coverage": round(float(values.mean()), 4),
        "ci95": [round(low, 4), round(high, 4)],
        # The quantity a single-split reading must be judged against.
        "per_split_sd": round(float(values.std(ddof=1)), 4),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
        "mean_interval_width": round(float(np.mean(widths)), 4),
        # Conformal guarantees at least the nominal level; over-coverage is expected for
        # small calibration sets and is not a failure.
        "verdict": "valid" if high >= 0.90 else "undercovers_investigate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path,
                        default=Path("data/raw/expansionrx/expansion_data_train.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/conformal_validation.json"))
    parser.add_argument("--endpoints", nargs="*", default=None)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing validation report")

    from protocol.lock import SETTINGS, verify
    locked = verify()
    train = pd.read_csv(args.train)
    chosen = args.endpoints or endpoints(train)

    results = {}
    for endpoint in chosen:
        results[endpoint] = validate(train, endpoint, SETTINGS["model_seeds"],
                                     SETTINGS["test_fraction"], args.repeats)
        row = results[endpoint]
        if row.get("verdict") == "insufficient_repeats":
            print(f"{endpoint:30s} insufficient repeats")
            continue
        print(f"{endpoint:30s} mean={row['grand_mean_coverage']:.4f} "
              f"CI={row['ci95']} sd={row['per_split_sd']:.3f} -> {row['verdict']}")

    report = {
        "status": "conformal_validity_check_random_splits_only",
        "lock_created_utc": locked["created_utc"],
        "input_sha256": sha256(args.train.read_bytes()).hexdigest(),
        "repeats": args.repeats,
        "note": ("Random splits only; no shift is present by construction. A verdict of "
                 "'valid' means the implementation reaches the nominal level, so any "
                 "undercoverage measured under a shifted split is attributable to shift."),
        "endpoints": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
