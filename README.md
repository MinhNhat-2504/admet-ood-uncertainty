# Uncertainty under real prospective shift in ADMET prediction

Research code testing whether uncertainty estimates that look calibrated on benchmark
splits stay calibrated against a **real** prospective shift, and whether simulated splits
reveal it when they do not.

## Research question

Uncertainty estimates decide which predictions a project acts on, and they are validated
on splits defined by chemical structure: scaffold, cluster, size, frontier. Whether a
method validated that way survives the shift a model meets in use had not been tested,
because public ADMET datasets with genuine temporal provenance did not exist.

The OpenADMET–ExpansionRx release provides one. This repository uses it to ask: **does
calibration transfer from a simulated split to a real campaign boundary?**

The analysis plan was registered with a checksum lock before any model was fitted
([`protocol/`](protocol/)), and the real test set was opened once, after the
simulated-split results were frozen.

## Data

| Dataset | Molecules | Endpoints | Split | Licence |
| --- | ---: | ---: | --- | --- |
| [OpenADMET-ExpansionRx](https://doi.org/10.5281/zenodo.21504733) | 7,608 | 9 (+1 raw) | real early-to-late campaign boundary | CC-BY-4.0 |

Raw files are not redistributed. The fetcher verifies the published MD5 digests and
refuses a mismatch:

```powershell
python -m admetood.fetch_data
```

## Findings

All measured; reproduced by the commands below.

**1. The reference standard is clean.** Across five levels of molecular identity
(canonical SMILES, InChIKey, salt/charge-stripped parent, stereo-insensitive,
tautomer-insensitive), **0 of 2,282** test molecules recur in training. Mean
nearest-neighbour Tanimoto is 0.613 — a near-neighbour shift, not far-OOD.

**2. The estimator is correct before shift.** Over 20 independent random splits,
split conformal attains nominal 90% coverage on all eight eligible endpoints
(0.899–0.915).

**3. Calibration breaks under real shift, differently per endpoint.** Intervals
calibrated to 90% deliver:

| Endpoint | Coverage |
| --- | ---: |
| Caco-2 Efflux Ratio | 0.587 |
| Caco-2 Papp A>B | 0.713 |
| LogD | 0.760 |
| KSOL | 0.779 |
| MLM CLint | 0.915 |
| HLM CLint | 0.947 |

**4. A five-member ensemble's ±1.645σ interval covers 0.177 — for boosted trees.**
Ensemble spread measures agreement among models trained on the same data; under shift they
agree and are wrong together. Refitting with a neural network, the same construction covers
0.813, so this failure belongs to low-variance learners rather than to ensembles as such.
The conformal undercoverage does reproduce across both families (mean 0.779 and 0.810,
six of eight endpoints under nominal), and so does the distance-conditioned fix (0.891 and
0.907).

**5. Distance-conditioned calibration recovers most of it.** Mean coverage 0.891, with
comparisons below 0.80 cut from 29 of 40 to 5 of 40. Named in the protocol before any run,
and it produces **no unbounded intervals**: the figure is a claim about every test
molecule. The Mondrian variant looks comparable at 0.879 but declines to bound 21.7% of
molecules, and covers 0.848 on those it does bound.
It does not, however, improve the *ordering* of predictions by reliability: on
selective prediction it is the weakest of the three ranking signals (−0.005 MAE against
ensemble spread, better on 5 of 8 endpoints), while quantile width is the strongest
(+0.030 [+0.009, +0.051], better on 7 of 8). Interval validity and error ranking are
different capabilities and no method here wins both.

**6. No simulated split reveals the failure.** Across 40 split × endpoint comparisons,
**0 transfers** within the prespecified ±5 pp margin. The five constructors are also not
distinguishable from one another (Friedman p = 0.52), random assignment included: mean
absolute gaps run 8.4–13.6 pp and no pairwise difference survives a paired test across the
eight endpoints. The structure-based machinery bought no demonstrable advantage over a
random partition, though eight endpoints cannot exclude a modest one.

**7. Why.** On these molecules structural distance from training is one fixed quantity
while label shift differs by endpoint in both magnitude and sign: 0.003 training standard
deviations for MLM CLint, +1.176 for Caco-2 efflux ratio and −0.765 for Caco-2 Papp A>B,
which move in opposite directions. A partition defined on structure cannot express that.

Results come from one organization's campaigns in one therapeutic area: they establish
that these failures *can* occur, not how often.

## Reproduce

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:PYTHONPATH = ".;src"

python -m admetood.fetch_data
python protocol\lock.py verify
python -m admetood.audit_leakage --train data\raw\expansionrx\expansion_data_train.csv --test data\raw\expansionrx\expansion_data_test.csv --output results\leakage_audit.json
python -m admetood.characterize_splits
python -m admetood.validate_conformal
python -m admetood.run_simulated      # simulated splits only
python -m admetood.run_real           # opens the real test set
python -m admetood.compare_splits
python -m admetood.risk_curves
python -m admetood.interval_audit
python -m admetood.robustness     # exploratory second model family
python -m admetood.make_figures   # writes paper/figures/
python -m pytest
```

Scripts refuse to overwrite an existing report, and every scoring script verifies the
lock before writing. `data/`, `results/` and `paper/` are excluded from Git; artifacts are
identified by the SHA-256 values recorded in each report.
