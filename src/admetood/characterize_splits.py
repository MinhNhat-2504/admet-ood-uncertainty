"""Characterize simulated splits and compare their shift profile to the real split.

Descriptive only: no model is fitted here. The question is whether a split defined on
structure can reproduce the shift the real prospective split actually produces, and in
particular whether it can reproduce shift that differs between endpoints measured on the
same molecules.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import splits as split_module
from .audit_leakage import nearest_neighbour_similarity
from .standardize import keys_for

# Endpoints modelled on a log10 scale, fixed in the protocol and not revisited.
LOG_ENDPOINTS = {"KSOL", "HLM CLint", "MLM CLint", "Caco-2 Permeability Papp A>B",
                 "Caco-2 Permeability Efflux", "MPPB", "MBPB", "MGMB"}
ID_COLUMNS = ("Molecule Name", "SMILES")
MIN_ROWS_FOR_SHIFT = 30  # Below this a shift estimate is too noisy to report.


def endpoints(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ID_COLUMNS]


def scaled(values: pd.Series, endpoint: str) -> pd.Series:
    """Apply the prespecified transform; clip guards against zero and negative readings."""
    if endpoint in LOG_ENDPOINTS:
        return np.log10(values.clip(lower=1e-3))
    return values


def label_shift(reference: pd.DataFrame, held_out: pd.DataFrame) -> dict:
    """Shift of each endpoint, in reference standard deviations, plus a KS statistic."""
    report = {}
    for endpoint in endpoints(reference):
        a = scaled(reference[endpoint].dropna(), endpoint)
        b = scaled(held_out[endpoint].dropna(), endpoint)
        if len(a) < MIN_ROWS_FOR_SHIFT or len(b) < MIN_ROWS_FOR_SHIFT:
            report[endpoint] = {"n_reference": int(len(a)), "n_held_out": int(len(b)),
                                "shift_sd": None, "ks": None}
            continue
        spread = float(a.std())
        statistic, p_value = stats.ks_2samp(a, b)
        report[endpoint] = {
            "n_reference": int(len(a)),
            "n_held_out": int(len(b)),
            # Signed effect size; p-values are omitted from the headline because at these
            # sample sizes they are significant for negligible differences.
            "shift_sd": round(float((b.mean() - a.mean()) / spread), 4) if spread else None,
            "ks": round(float(statistic), 4),
            "ks_p": float(p_value),
        }
    return report


def scaffold_overlap(reference_smiles: list[str], held_out_smiles: list[str]) -> dict:
    reference = {keys_for(s).scaffold for s in reference_smiles} - {None, ""}
    held_out = [keys_for(s).scaffold for s in held_out_smiles]
    shared = [s for s in held_out if s in reference and s]
    return {"held_out_rows": len(held_out), "rows_sharing_scaffold": len(shared),
            "fraction": round(len(shared) / max(len(held_out), 1), 4)}


def characterize(reference: pd.DataFrame, held_out: pd.DataFrame, name: str) -> dict:
    similarity = nearest_neighbour_similarity(list(reference["SMILES"]),
                                              list(held_out["SMILES"]))
    return {
        "split": name,
        "n_reference": int(len(reference)),
        "n_held_out": int(len(held_out)),
        "held_out_fraction": round(len(held_out) / (len(reference) + len(held_out)), 4),
        "nearest_neighbour_tanimoto": {
            "mean": round(float(similarity.mean()), 4),
            "median": round(float(np.median(similarity)), 4),
            **{f"fraction_above_{t}": round(float((similarity > t).mean()), 4)
               for t in (0.4, 0.7, 0.9)},
        },
        "scaffold_overlap": scaffold_overlap(list(reference["SMILES"]),
                                             list(held_out["SMILES"])),
        "label_shift": label_shift(reference, held_out),
    }


def shift_spread(characterization: dict) -> dict:
    """How much the shift varies across endpoints within one split.

    A structure-based split applies one partition to all endpoints. If real shift is
    endpoint-specific, a split that produces a narrow spread cannot reproduce it.
    """
    values = [e["shift_sd"] for e in characterization["label_shift"].values()
              if e["shift_sd"] is not None]
    if not values:
        return {"n_endpoints": 0}
    absolute = [abs(v) for v in values]
    return {
        "n_endpoints": len(values),
        "min_abs_shift_sd": round(min(absolute), 4),
        "max_abs_shift_sd": round(max(absolute), 4),
        "range_abs_shift_sd": round(max(absolute) - min(absolute), 4),
        "mean_abs_shift_sd": round(float(np.mean(absolute)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path,
                        default=Path("data/raw/expansionrx/expansion_data_train.csv"))
    parser.add_argument("--test", type=Path,
                        default=Path("data/raw/expansionrx/expansion_data_test.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/split_characterization.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing characterization")

    from protocol.lock import SETTINGS, verify
    locked = verify()

    train = pd.read_csv(args.train)
    real_test = pd.read_csv(args.test)
    smiles = list(train["SMILES"])

    # The real split is the reference standard: full training set versus the real test.
    report = {
        "status": "descriptive_split_characterization_no_model_fitted",
        "lock_created_utc": locked["created_utc"],
        "inputs": {str(p): sha256(p.read_bytes()).hexdigest() for p in (args.train, args.test)},
        "real": characterize(train, real_test, "real_prospective"),
        "simulated": {},
    }
    report["real"]["shift_spread"] = shift_spread(report["real"])

    for name in SETTINGS["simulated_splits"]:
        mask = split_module.build(name, smiles, SETTINGS["test_fraction"],
                                  SETTINGS["split_seed"])
        characterization = characterize(train.loc[~mask].reset_index(drop=True),
                                        train.loc[mask].reset_index(drop=True), name)
        characterization["shift_spread"] = shift_spread(characterization)
        report["simulated"][name] = characterization

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    rows = [("real", report["real"])] + list(report["simulated"].items())
    print(f"{'split':12s} {'held':>6s} {'NN sim':>7s} {'scaf%':>6s} "
          f"{'|shift| min':>11s} {'max':>6s} {'range':>6s}")
    for name, value in rows:
        spread = value["shift_spread"]
        print(f"{name:12s} {value['n_held_out']:6d} "
              f"{value['nearest_neighbour_tanimoto']['mean']:7.3f} "
              f"{value['scaffold_overlap']['fraction'] * 100:6.1f} "
              f"{spread.get('min_abs_shift_sd', float('nan')):11.3f} "
              f"{spread.get('max_abs_shift_sd', float('nan')):6.3f} "
              f"{spread.get('range_abs_shift_sd', float('nan')):6.3f}")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
