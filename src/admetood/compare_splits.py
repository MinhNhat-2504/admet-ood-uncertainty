"""Secondary analysis: do the split constructors differ from each other at all?

The analysis plan asks whether the structure-based constructors beat a random partition
at anticipating prospective calibration. Section 6.1 of the write-up answers it from the
verdict counts, which are coarse; this script tests the underlying gaps directly, so the
claim rests on a saved artifact rather than an arithmetic aside.

Low power is the point to keep in view: eight endpoints across five constructors cannot
resolve a modest difference, so a null result here is "not shown to differ", not "equal".
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy import stats

PRIMARY_METHOD = "split_conformal"
REFERENCE = "random"


def absolute_gaps(report: dict, method: str = PRIMARY_METHOD) -> dict[str, np.ndarray]:
    """Per-endpoint |gap| for each split, aligned on the eligible endpoint order."""
    endpoints = report["eligible_endpoints"]
    by_split: dict[str, np.ndarray] = {}
    for row in report["comparisons"]:
        by_split.setdefault(row["split"], {})[row["endpoint"]] = abs(
            row["methods"][method]["gap_pp"])
    aligned = {}
    for split, values in by_split.items():
        if all(e in values for e in endpoints):   # Only fully observed splits are paired.
            aligned[split] = np.array([values[e] for e in endpoints], dtype=float)
    return aligned


def paired_difference(a: np.ndarray, b: np.ndarray) -> dict:
    """Mean paired difference with a normal interval and a distribution-free p-value."""
    difference = a - b
    n = difference.size
    standard_error = float(difference.std(ddof=1) / np.sqrt(n))
    p_value = float(stats.wilcoxon(a, b).pvalue) if np.any(difference != 0) else 1.0
    return {
        "mean_difference_pp": round(float(difference.mean()), 3),
        "standard_error_pp": round(standard_error, 3),
        "ci95_pp": [round(float(difference.mean() - 1.96 * standard_error), 3),
                    round(float(difference.mean() + 1.96 * standard_error), 3)],
        "wilcoxon_p": round(p_value, 4),
    }


def compare(report: dict, method: str = PRIMARY_METHOD) -> dict:
    gaps = absolute_gaps(report, method)
    names = sorted(gaps)
    if len(names) < 2:
        return {"method": method, "error": "fewer than two fully observed splits"}

    # Friedman needs at least three groups; with two, the paired tests below carry it.
    friedman = (stats.friedmanchisquare(*[gaps[name] for name in names])
                if len(names) >= 3 else None)
    result = {
        "method": method,
        "n_endpoints": int(len(report["eligible_endpoints"])),
        "splits": names,
        "mean_absolute_gap_pp": {
            name: round(float(gaps[name].mean()), 2) for name in names},
        "standard_error_pp": {
            name: round(float(gaps[name].std(ddof=1) / np.sqrt(gaps[name].size)), 2)
            for name in names},
        "friedman": ({"statistic": round(float(friedman.statistic), 3),
                      "p_value": round(float(friedman.pvalue), 4)}
                     if friedman is not None else None),
        "versus_reference": {},
        "all_pairs": {},
    }
    for name in names:
        if name != REFERENCE and REFERENCE in gaps:
            result["versus_reference"][f"{REFERENCE}_minus_{name}"] = paired_difference(
                gaps[REFERENCE], gaps[name])
    for first, second in combinations(names, 2):
        result["all_pairs"][f"{first}_minus_{second}"] = paired_difference(
            gaps[first], gaps[second])

    separated = [key for key, value in result["all_pairs"].items()
                 if value["wilcoxon_p"] < 0.05]
    result["any_pair_separated_at_0.05"] = separated
    result["reading"] = (
        "No pair of split constructors is separated at the 0.05 level, random assignment "
        "included, so the structure-based constructors are not shown to anticipate "
        "prospective calibration better than a random partition. With eight endpoints "
        "this does not exclude a modest advantage; the point estimates mildly favour the "
        "distance-based constructors."
        if not separated else
        f"Separated pairs: {separated}.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path,
                        default=Path("results/transfer_analysis.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/split_comparison.json"))
    args = parser.parse_args()
    if not args.transfer.is_file():
        parser.error("Run the primary transfer analysis first")

    report = json.loads(args.transfer.read_text(encoding="utf-8"))
    comparison = {"status": "secondary_comparison_of_split_constructors",
                  "source": str(args.transfer),
                  "primary": compare(report, PRIMARY_METHOD)}

    primary = comparison["primary"]
    print(f"Mean |gap| over {primary['n_endpoints']} endpoints, {PRIMARY_METHOD}:")
    for name in primary["splits"]:
        print(f"  {name:10s} {primary['mean_absolute_gap_pp'][name]:5.1f} "
              f"+/- {primary['standard_error_pp'][name]:4.1f} pp")
    if primary["friedman"] is not None:
        print(f"\nFriedman chi2={primary['friedman']['statistic']:.2f}, "
              f"p={primary['friedman']['p_value']:.3f}")
    for key, value in primary["versus_reference"].items():
        print(f"  {key:24s} {value['mean_difference_pp']:+5.1f} pp "
              f"CI95 [{value['ci95_pp'][0]:+5.1f},{value['ci95_pp'][1]:+5.1f}] "
              f"Wilcoxon p={value['wilcoxon_p']:.3f}")
    print(f"\n{primary['reading']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
