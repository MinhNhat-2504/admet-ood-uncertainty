"""Figure building: the values plotted must come from the saved results, unchanged."""

import json

import numpy as np
import pytest

from admetood import make_figures as mf


def transfer_report() -> dict:
    endpoints = ["LogD", "KSOL"]
    splits = ["random", "scaffold"]
    coverage = {("LogD", "random"): 0.70, ("LogD", "scaffold"): 0.80,
                ("KSOL", "random"): 0.50, ("KSOL", "scaffold"): 0.60}
    return {
        "eligible_endpoints": endpoints,
        "comparisons": [
            {"split": split, "endpoint": endpoint,
             "methods": {name: {"coverage_real": coverage[(endpoint, split)]}
                         for name in mf.METHOD_LABELS}}
            for endpoint in endpoints for split in splits
        ],
    }


def test_real_coverage_collects_every_size_matched_split():
    values = mf.real_coverage(transfer_report()["comparisons"], "LogD")
    assert sorted(values.tolist()) == [0.70, 0.80]


def test_real_coverage_does_not_mix_endpoints():
    """A mis-filtered endpoint would silently average unrelated numbers into a bar."""
    report = transfer_report()
    assert mf.real_coverage(report["comparisons"], "KSOL").tolist() == [0.50, 0.60]
    assert np.isclose(mf.real_coverage(report["comparisons"], "KSOL").mean(), 0.55)


def test_unknown_endpoint_yields_nothing_rather_than_a_wrong_bar():
    assert mf.real_coverage(transfer_report()["comparisons"], "absent").size == 0


def test_label_shortens_only_the_endpoints_it_knows():
    assert mf.label("Caco-2 Permeability Efflux") == "Caco-2 efflux ratio"
    assert mf.label("LogD") == "LogD"


def test_every_method_in_the_legend_is_one_the_analysis_produces():
    from admetood import modeling
    assert set(mf.METHOD_LABELS) == set(modeling.METHODS)


@pytest.mark.parametrize("name", ["figure1_real_coverage.png", "figure2_method_coverage.png",
                                  "figure3_shift_dissociation.png",
                                  "figure4_split_comparison.png"])
def test_each_figure_is_written(tmp_path, name):
    transfer = transfer_report()
    characterization = {"real": {
        "label_shift": {"LogD": {"shift_sd": -0.5}, "KSOL": {"shift_sd": 1.1}},
        "nearest_neighbour_tanimoto": {"mean": 0.613}}}
    comparison = {"primary": {
        "splits": ["random", "scaffold"], "n_endpoints": 2,
        "mean_absolute_gap_pp": {"random": 12.6, "scaffold": 10.8},
        "standard_error_pp": {"random": 2.9, "scaffold": 2.1},
        "friedman": None}}
    builders = {
        "figure1_real_coverage.png": lambda p: mf.figure_real_coverage(transfer, p),
        "figure2_method_coverage.png": lambda p: mf.figure_method_coverage(transfer, p),
        "figure3_shift_dissociation.png":
            lambda p: mf.figure_shift_dissociation(transfer, characterization, p),
        "figure4_split_comparison.png":
            lambda p: mf.figure_split_comparison(comparison, p),
    }
    path = tmp_path / name
    builders[name](path)
    assert path.is_file() and path.stat().st_size > 1000


def test_a_missing_friedman_result_does_not_break_the_caption(tmp_path):
    """Two splits produce no Friedman test; the figure must still build."""
    comparison = {"primary": {
        "splits": ["random", "scaffold"], "n_endpoints": 2,
        "mean_absolute_gap_pp": {"random": 12.6, "scaffold": 10.8},
        "standard_error_pp": {"random": 2.9, "scaffold": 2.1},
        "friedman": None}}
    path = tmp_path / "f4.png"
    mf.figure_split_comparison(comparison, path)
    assert path.is_file()


def test_figures_match_the_saved_results_when_they_exist():
    """Guards the claim that plotted values are read from results, never recomputed."""
    from pathlib import Path
    results = Path("results/transfer_analysis.json")
    if not results.is_file():
        pytest.skip("results not built in this environment")
    transfer = json.loads(results.read_text(encoding="utf-8"))
    endpoint = transfer["eligible_endpoints"][0]
    expected = [r["methods"][mf.PRIMARY_METHOD]["coverage_real"]
                for r in transfer["comparisons"] if r["endpoint"] == endpoint]
    assert mf.real_coverage(transfer["comparisons"], endpoint).tolist() == expected
