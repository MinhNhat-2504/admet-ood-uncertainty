"""The audit must find leaks that exact-SMILES matching misses, and count them once."""

import pandas as pd

from admetood.audit_leakage import audit, leakage_by_level, key_table

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_SODIUM = "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"
ALANINE_R = "C[C@H](N)C(=O)O"
ALANINE_S = "C[C@@H](N)C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
BENZENE = "c1ccccc1"


def frame(smiles):
    return pd.DataFrame({"SMILES": smiles})


def test_salt_leak_is_invisible_to_exact_smiles_but_caught_by_parent_key():
    report = leakage_by_level(key_table([ASPIRIN]), key_table([ASPIRIN_SODIUM]))
    assert report["canonical_smiles"]["test_rows_matched"] == 0
    assert report["inchikey"]["test_rows_matched"] == 0
    assert report["parent_inchikey"]["test_rows_matched"] == 1
    assert report["parent_inchikey"]["fraction_of_test"] == 1.0


def test_stereo_leak_is_caught_only_at_the_stereo_level():
    report = leakage_by_level(key_table([ALANINE_R]), key_table([ALANINE_S]))
    assert report["parent_inchikey"]["test_rows_matched"] == 0
    assert report["stereo_insensitive"]["test_rows_matched"] == 1


def test_clean_split_reports_no_leakage():
    report = leakage_by_level(key_table([CAFFEINE]), key_table([ASPIRIN]))
    assert all(level["test_rows_matched"] == 0 for level in report.values())


def test_audit_reports_counts_parse_failures_and_similarity():
    result = audit(frame([ASPIRIN, CAFFEINE]), frame([ASPIRIN_SODIUM, "junk", BENZENE]))
    assert result["n_train"] == 2 and result["n_test"] == 3
    assert result["parse_failures"]["test"] == 1
    assert result["leakage_by_level"]["parent_inchikey"]["test_rows_matched"] == 1
    # An unparsable row must not be counted as comparable.
    assert result["leakage_by_level"]["parent_inchikey"]["test_rows_comparable"] == 2
    assert 0.0 <= result["nearest_neighbour_tanimoto"]["mean"] <= 1.0


def test_within_split_duplicates_are_counted_at_parent_level():
    result = audit(frame([ASPIRIN, ASPIRIN_SODIUM, CAFFEINE]), frame([BENZENE]))
    duplicates = result["within_split_duplicates"]["train"]
    assert duplicates["rows"] == 3 and duplicates["distinct"] == 2
    assert duplicates["duplicate_rows"] == 1
