# ADMET prediction under distribution shift

Research code for evaluating whether **structure-based out-of-distribution (OOD) splits**
predict model behaviour under **real prospective shift**, and how uncertainty
quantification behaves in each case.

## Research question

Public ADMET benchmarks test generalization with structure-based splits: scaffold,
cluster, Lo-Hi, DataSAIL, structural frontier. These splits are defined on molecular
structure alone, so for a set of molecules measured on several endpoints they assign
**the same** difficulty to every endpoint.

A real prospective split does not behave that way. On the ExpansionRx early-to-late
campaign split, structural distance from training is uniform across endpoints, yet
label shift ranges from negligible to very large (Section "Initial findings"). If real
difficulty is endpoint-specific while structure-based difficulty is not, structure-based
splits may rank methods and calibrate uncertainty in ways that do not transfer.

This repository tests that claim.

## Data

| Dataset | Molecules | Endpoints | Split | Licence |
| --- | ---: | ---: | --- | --- |
| [OpenADMET-ExpansionRx](https://doi.org/10.5281/zenodo.21504733) | 7,608 | 9 (+1 raw) | real early-to-late campaign time split | CC-BY-4.0 |

The released archive contains labels for the previously blinded test split, so the real
prospective split can be scored directly. Raw files are not committed; download them with
their published checksums:

```powershell
python -m admetood.fetch_data
```

## Initial findings

Measured, not assumed. Both are reproduced by the commands below.

**1. The official time split has no molecular-identity leakage.** Across six identity
levels (canonical SMILES, InChIKey, salt/charge-stripped parent, stereo-insensitive,
tautomer-insensitive), **0 of 2,282** test molecules reappear in training. Only Murcko
scaffolds overlap (625 test rows, 27.4%), which is expected in lead optimization.
Mean nearest-neighbour Tanimoto is 0.613 — a near-neighbour shift, not a far-OOD one.

**2. Label shift is strongly endpoint-dependent on the same molecules.**

| Endpoint | Shift (training SD) | KS |
| --- | ---: | ---: |
| Caco-2 Efflux Ratio | +1.18 | 0.366 |
| Caco-2 Papp A>B | −0.77 | 0.281 |
| MPPB | +0.31 | 0.167 |
| HLM CLint | +0.20 | 0.155 |
| KSOL | +0.18 | 0.177 |
| MGMB | +0.09 | 0.201 |
| MBPB | +0.06 | 0.098 |
| MLM CLint | +0.00 | 0.149 |
| LogD | +0.01 | 0.042 |

Same molecules, same structural distance, difficulty differing by two orders of
magnitude. Structure alone cannot express this.

## Reproduce

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
$env:PYTHONPATH = "src"
python -m admetood.fetch_data
python -m admetood.audit_leakage --train data/raw/expansionrx/expansion_data_train.csv --test data/raw/expansionrx/expansion_data_test.csv --output results/leakage_audit_expansionrx_official.json
python -m pytest
```

`data/`, `results/` and `paper/` are excluded from Git; artifacts are identified by the
SHA-256 values recorded in each report.

## Status

Protocol and analysis plan are being written before any model is fitted. No predictive
result is claimed yet.
