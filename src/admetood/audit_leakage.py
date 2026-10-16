"""Audit train/test leakage at several molecular identity levels.

Reports how many test molecules reappear in training under progressively more
permissive notions of identity, plus nearest-neighbour similarity. The output is a
"leakage card" that a benchmark can publish alongside its splits.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .standardize import LEVELS, Keys, keys_for

SMILES_COLUMN = "SMILES"


def key_table(smiles: Iterable[str]) -> pd.DataFrame:
    """One row per input SMILES with every identity key."""
    rows = [keys_for(s) for s in smiles]
    return pd.DataFrame([vars(row) for row in rows])


def _fingerprints(smiles: Iterable[str]) -> tuple[list, list[int]]:
    """Morgan count fingerprints; returns vectors and the indices that parsed."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    vectors, kept = [], []
    for index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is not None:
            vectors.append(generator.GetFingerprint(mol))
            kept.append(index)
    return vectors, kept


def nearest_neighbour_similarity(train_smiles: list[str],
                                 test_smiles: list[str]) -> np.ndarray:
    """Max Tanimoto similarity of each test molecule to any training molecule."""
    train_fps, _ = _fingerprints(train_smiles)
    test_fps, _ = _fingerprints(test_smiles)
    if not train_fps or not test_fps:
        return np.array([])
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
                     for fp in test_fps])


def leakage_by_level(train_keys: pd.DataFrame, test_keys: pd.DataFrame) -> dict:
    """Count test rows whose key also occurs in training, for each identity level."""
    report = {}
    for level in LEVELS:
        train_values = set(train_keys[level].dropna()) - {""}
        test_values = test_keys[level]
        matched = test_values.isin(train_values) & test_values.notna() & (test_values != "")
        report[level] = {
            "test_rows_matched": int(matched.sum()),
            "test_rows_comparable": int((test_values.notna() & (test_values != "")).sum()),
            "fraction_of_test": round(float(matched.mean()), 6),
            "distinct_shared_values": len(train_values & set(test_values.dropna()) - {""}),
        }
    return report


def duplicate_report(keys: pd.DataFrame, level: str = "parent_inchikey") -> dict:
    """Duplicates inside one split, which inflate apparent dataset size."""
    values = keys[level].dropna()
    counts = Counter(values)
    repeated = {value: count for value, count in counts.items() if count > 1}
    return {"level": level, "rows": int(len(values)), "distinct": len(counts),
            "duplicate_rows": int(sum(repeated.values()) - len(repeated)),
            "duplicated_values": len(repeated)}


def audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    train_keys = key_table(train[SMILES_COLUMN])
    test_keys = key_table(test[SMILES_COLUMN])
    similarity = nearest_neighbour_similarity(list(train[SMILES_COLUMN]),
                                              list(test[SMILES_COLUMN]))
    if similarity.size:
        neighbour = {"mean": round(float(similarity.mean()), 4)}
        neighbour.update({f"p{p}": round(float(np.percentile(similarity, p)), 4)
                          for p in (50, 75, 90, 95, 99)})
        neighbour.update({f"fraction_above_{t}": round(float((similarity > t).mean()), 4)
                          for t in (0.4, 0.7, 0.9)})
    else:
        neighbour = {"mean": None}
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "parse_failures": {
            "train": int(train_keys["error"].notna().sum()),
            "test": int(test_keys["error"].notna().sum()),
            "reasons": Counter(train_keys["error"].dropna()).most_common(5),
        },
        "leakage_by_level": leakage_by_level(train_keys, test_keys),
        "within_split_duplicates": {
            "train": duplicate_report(train_keys),
            "test": duplicate_report(test_keys),
        },
        "nearest_neighbour_tanimoto": neighbour,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing audit report")

    train, test = pd.read_csv(args.train), pd.read_csv(args.test)
    report = {
        "status": "train_test_leakage_audit_identity_levels",
        "inputs": {str(path): sha256(path.read_bytes()).hexdigest()
                   for path in (args.train, args.test)},
        **audit(train, test),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "leakage_by_level": {level: value["test_rows_matched"]
                                           for level, value in report["leakage_by_level"].items()},
                      "nn_tanimoto": report["nearest_neighbour_tanimoto"]}, indent=2))


if __name__ == "__main__":
    main()
