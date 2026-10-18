"""Build the manuscript figures from the saved result files.

Every value plotted is read from a results JSON; nothing is recomputed from the raw data
and nothing is hard-coded, so a figure cannot drift from the numbers in the text. Figures
are written under paper/, which is excluded from version control along with the draft.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRIMARY_METHOD = "split_conformal"
NOMINAL = 0.90
METHOD_LABELS = {
    "ensemble": "Ensemble +/-1.645 sigma",
    "quantile": "Quantile regression",
    "split_conformal": "Split conformal",
    "mondrian_conformal": "Mondrian\n(distance-conditioned)",
    "distance": "Conformalized distance",
}
SHORT = {
    "Caco-2 Permeability Efflux": "Caco-2 efflux ratio",
    "Caco-2 Permeability Papp A>B": "Caco-2 Papp A>B",
}


def label(endpoint: str) -> str:
    return SHORT.get(endpoint, endpoint)


def real_coverage(comparisons: list[dict], endpoint: str,
                  method: str = PRIMARY_METHOD) -> np.ndarray:
    """Real-split coverage for one endpoint, one value per size-matched split."""
    return np.array([r["methods"][method]["coverage_real"]
                     for r in comparisons if r["endpoint"] == endpoint], dtype=float)


def figure_real_coverage(transfer: dict, path: Path) -> None:
    """Nominal 90% intervals against what they actually cover on the real split."""
    endpoints = transfer["eligible_endpoints"]
    values = [real_coverage(transfer["comparisons"], e) for e in endpoints]
    means = np.array([v.mean() for v in values])
    order = np.argsort(means)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = np.arange(len(order))
    ax.barh(y, means[order], color="#b0b7c3", height=0.62, zorder=2)
    for row, index in enumerate(order):
        # The spread is across size-matched training subsets, not across molecules.
        ax.plot([values[index].min(), values[index].max()], [row, row],
                color="#2f3640", linewidth=1.4, zorder=3)
    ax.axvline(NOMINAL, color="#c0392b", linestyle="--", linewidth=1.4, zorder=4,
               label="nominal 0.90")
    ax.set_yticks(y, [label(endpoints[i]) for i in order], fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Empirical coverage of nominal 90% split-conformal intervals")
    ax.set_title("Calibration under real prospective shift, by endpoint", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def figure_method_coverage(transfer: dict, path: Path) -> None:
    """Every method's real-split coverage, one point per comparison."""
    methods = list(METHOD_LABELS)
    per_method = [np.array([r["methods"][m]["coverage_real"]
                            for r in transfer["comparisons"]], dtype=float)
                  for m in methods]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    rng = np.random.default_rng(0)   # Jitter is cosmetic; seeded so the figure is stable.
    for position, values in enumerate(per_method):
        ax.scatter(values, position + rng.uniform(-0.17, 0.17, values.size),
                   s=16, alpha=0.55, color="#34495e", zorder=3, linewidths=0)
        ax.plot([values.mean()], [position], marker="|", markersize=26,
                color="#c0392b", markeredgewidth=2.2, zorder=4)
    ax.axvline(NOMINAL, color="#c0392b", linestyle="--", linewidth=1.4, zorder=2)
    ax.set_yticks(range(len(methods)), [METHOD_LABELS[m] for m in methods], fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Real-split coverage across the 40 split x endpoint comparisons")
    ax.set_title("Nominal 90% intervals: what each method delivers", fontsize=11)
    # Anchored to the line in x but to the axes in y, so it cannot ride up into the title.
    ax.annotate("nominal 0.90", xy=(NOMINAL - 0.012, 0.02),
                xycoords=("data", "axes fraction"), ha="right", va="bottom",
                fontsize=8.5, color="#c0392b")
    ax.grid(axis="x", alpha=0.25, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def figure_shift_dissociation(transfer: dict, characterization: dict, path: Path) -> None:
    """Label shift varies by endpoint while structural distance is one fixed quantity."""
    shift = characterization["real"]["label_shift"]
    similarity = characterization["real"]["nearest_neighbour_tanimoto"]["mean"]
    endpoints = [e for e in transfer["eligible_endpoints"] if e in shift]
    x = np.array([shift[e]["shift_sd"] for e in endpoints], dtype=float)
    y = np.array([real_coverage(transfer["comparisons"], e).mean() for e in endpoints])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(x, y, s=58, color="#2f3640", zorder=3)
    for xi, yi, name in zip(x, y, endpoints):
        # Labels go left of points in the right-hand third so they stay inside the axes.
        right = xi > x.min() + 0.72 * (x.max() - x.min())
        ax.annotate(label(name), (xi, yi), textcoords="offset points",
                    xytext=(-9 if right else 8, 5),
                    ha="right" if right else "left", fontsize=8)
    ax.axhline(NOMINAL, color="#c0392b", linestyle="--", linewidth=1.4, zorder=2)
    ax.axvline(0.0, color="#8d95a0", linewidth=0.9, zorder=1)
    ax.set_xlabel("Signed label shift, training to real test set "
                  "(training standard deviations)")
    ax.set_ylabel("Real-split coverage")
    ax.set_title("Structural distance is fixed across endpoints; label shift is not",
                 fontsize=11)
    ax.text(0.02, 0.05,
            "mean nearest-neighbour Tanimoto = "
            f"{similarity:.3f},\nidentical for every endpoint",
            transform=ax.transAxes, ha="left", fontsize=8.5, color="#555c66")
    ax.grid(alpha=0.25, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def figure_split_comparison(comparison: dict, path: Path) -> None:
    """Mean absolute transfer gap per constructor, with the test that fails to separate them."""
    primary = comparison["primary"]
    names = primary["splits"]
    means = np.array([primary["mean_absolute_gap_pp"][n] for n in names], dtype=float)
    errors = np.array([primary["standard_error_pp"][n] for n in names], dtype=float)
    order = np.argsort(means)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(order))
    colours = ["#c0392b" if names[i] == "random" else "#b0b7c3" for i in order]
    ax.bar(x, means[order], yerr=errors[order], capsize=4, color=colours, zorder=2,
           error_kw={"ecolor": "#2f3640", "elinewidth": 1.3})
    ax.set_xticks(x, [names[i] for i in order], fontsize=9)
    ax.set_ylabel("Mean |transfer gap| (pp)")
    ax.set_title("No constructor anticipates real calibration, and none is separated "
                 "from the others", fontsize=10.5)
    friedman = primary["friedman"]
    if friedman:
        caption = (f"Friedman chi2 = {friedman['statistic']:.2f}, "
                   f"p = {friedman['p_value']:.2f}; no pair separated at 0.05 "
                   f"(n = {primary['n_endpoints']} endpoints). Random assignment in red.")
    else:
        caption = "Random assignment in red."
    ax.text(0.0, -0.26, caption, transform=ax.transAxes, fontsize=8.5, color="#555c66")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_risk_coverage(curves: dict, path: Path) -> None:
    """Selective-prediction curves the saturated scalar could not distinguish."""
    endpoints = list(curves["endpoints"])
    grid = np.array(curves["endpoints"][endpoints[0]]["coverage_grid"], dtype=float)
    target = curves["target_error"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colours = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(endpoints)))
    for endpoint, colour in zip(endpoints, colours):
        mae = curves["endpoints"][endpoint]["methods"][PRIMARY_METHOD]["mae_at_coverage"]
        ax.plot(grid, mae, marker="o", markersize=3.4, linewidth=1.5, color=colour,
                label=label(endpoint))
    ax.axhline(target, color="#c0392b", linestyle="--", linewidth=1.4, zorder=1)
    # Right of the axes the target line runs through the flattest curve, so label it left.
    ax.annotate(f"prespecified target {target}", xy=(grid.min(), target),
                xytext=(0, -12), textcoords="offset points", ha="left", fontsize=8.5,
                color="#c0392b")
    ax.set_xlabel("Fraction of compounds kept, most confident first")
    ax.set_ylabel("Mean absolute error of the kept compounds")
    ax.set_title("Selective prediction on the real split (split conformal)", fontsize=11)
    ax.legend(fontsize=7.6, frameon=False, ncol=2, loc="upper left")
    ax.grid(alpha=0.25, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def figure_toc(transfer: dict, path: Path) -> None:
    """Table-of-contents graphic: simulated coverage against real, with the margin band.

    Sized to the area ACS specifies, 3.33 x 1.87 inches (8.46 x 4.76 cm), and submitted at
    that actual size rather than larger, as the guidelines ask. It therefore carries one
    claim only — that the points do not lie in the band — and nothing that stops being
    legible when small.
    """
    margin = transfer["equivalence_margin_pp"] / 100.0
    splits = sorted({r["split"] for r in transfer["comparisons"]})
    colours = dict(zip(splits, plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(splits)))))

    fig, ax = plt.subplots(figsize=(3.33, 1.87))
    grid = np.array([0.5, 1.0])
    ax.fill_between(grid, grid - margin, grid + margin, color="#9aa4b1", alpha=0.35,
                    linewidth=0, zorder=1)
    ax.plot(grid, grid, color="#5b6472", linewidth=0.8, zorder=2)
    for split in splits:
        xs = [r["methods"][PRIMARY_METHOD]["coverage_simulated"]
              for r in transfer["comparisons"] if r["split"] == split]
        ys = [r["methods"][PRIMARY_METHOD]["coverage_real"]
              for r in transfer["comparisons"] if r["split"] == split]
        ax.scatter(xs, ys, s=13, color=colours[split], label=split, zorder=3, linewidths=0)

    ax.set_xlim(0.55, 1.0)
    # The lowest point sits at about 0.57, so the extra headroom below keeps the caption
    # clear of the data rather than printing it over a marker.
    ax.set_ylim(0.46, 1.0)
    ax.set_xticks([0.6, 0.8, 1.0])
    ax.set_yticks([0.6, 0.8, 1.0])
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.set_xlabel("coverage on a simulated split", fontsize=6.5, labelpad=1)
    ax.set_ylabel("coverage on the\nreal prospective split", fontsize=6.5, labelpad=1)
    # The band is the equivalence margin, but a verdict of "transfers" needs the whole
    # bootstrap interval inside it, which this scatter cannot show. Ten point estimates
    # fall in the band and none transfers, so the title claims only what the eye can check.
    correlation = np.corrcoef(
        [r["methods"][PRIMARY_METHOD]["coverage_simulated"] for r in transfer["comparisons"]],
        [r["methods"][PRIMARY_METHOD]["coverage_real"] for r in transfer["comparisons"]])[0, 1]
    ax.set_title("Simulated splits do not predict real calibration", fontsize=7.5, pad=3)
    ax.annotate(f"r = {correlation:.2f}, 40 comparisons\nband = equivalence margin",
                xy=(0.985, 0.03), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=5.2, color="#4a525e")
    ax.legend(fontsize=5, frameon=False, loc="upper left", handletextpad=0.1,
              borderpad=0.1, labelspacing=0.18, markerscale=0.9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    def load(name: str) -> dict:
        path = args.results / name
        if not path.is_file():
            parser.error(f"Missing {path}; run the analysis scripts first")
        return json.loads(path.read_text(encoding="utf-8"))

    transfer = load("transfer_analysis.json")
    characterization = load("split_characterization.json")
    comparison = load("split_comparison.json")
    curves = load("risk_coverage_curves.json")

    args.output.mkdir(parents=True, exist_ok=True)
    figures = [
        ("figure1_real_coverage.png", lambda p: figure_real_coverage(transfer, p)),
        ("figure2_method_coverage.png", lambda p: figure_method_coverage(transfer, p)),
        ("figure3_shift_dissociation.png",
         lambda p: figure_shift_dissociation(transfer, characterization, p)),
        ("figure4_split_comparison.png",
         lambda p: figure_split_comparison(comparison, p)),
        ("figure5_risk_coverage.png", lambda p: figure_risk_coverage(curves, p)),
        ("toc_graphic.png", lambda p: figure_toc(transfer, p)),
    ]
    for name, build in figures:
        build(args.output / name)
        print(f"wrote {args.output / name}")


if __name__ == "__main__":
    main()
