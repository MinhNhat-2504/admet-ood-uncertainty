# Protocol — do structure-based OOD splits predict real prospective shift?

Version 1.0, written 17 October 2026, **before any model was fitted**. Amendments must be
dated, appended, and must never replace earlier text. The companion
[analysis plan](ANALYSIS_PLAN.md) fixes endpoints, metrics and decision rules.

## 1. Research question

Public ADMET benchmarks evaluate generalization with splits defined on molecular
structure: Bemis–Murcko scaffold, Butina/sphere-exclusion clustering, Lo-Hi, DataSAIL,
structural frontier. For a set of molecules assayed on several endpoints, such a split
necessarily assigns **the same partition, and therefore the same nominal difficulty, to
every endpoint**.

A real prospective split does not behave that way. Measured on the ExpansionRx
early-to-late campaign split (Section 3), the structural distance from training is a
single fixed quantity, while the shift in the label distribution ranges from
+0.01 to +1.18 training standard deviations depending on the endpoint.

**Hypothesis.** Because structure-based splits cannot express endpoint-specific
difficulty, the error levels, uncertainty calibration and method rankings they produce
do not transfer to a real prospective split, and the size of the mismatch varies by
endpoint.

**This is a falsifiable claim.** If simulated splits reproduce the real split's
calibration and rankings within the prespecified margins, the hypothesis is rejected and
the paper reports that structure-based splits are adequate surrogates.

## 2. What is frozen before any model run

- The dataset snapshot and its checksums (Section 3).
- The eligible endpoint list and sample-size rule (analysis plan Section 1).
- The simulated split constructors (Section 4).
- The model and uncertainty methods (Section 5).
- The primary endpoint, secondary endpoints and decision rules (analysis plan).
- The seeds: model seeds `0-4`, split seed `20261017`, bootstrap seed `20261018`.

`protocol/lock.py` records SHA-256 digests of this file, the analysis plan, the data
files and the source modules. The lock must be created before the first experiment and
verified by every scoring script.

## 3. Data

[OpenADMET–ExpansionRx](https://doi.org/10.5281/zenodo.21504733), CC-BY-4.0, released in
full after the blind challenge closed on 19 January 2026. Real ADMET measurements from
lead-optimization campaigns at Expansion Therapeutics.

| Split | Molecules | Provenance |
| --- | ---: | --- |
| `expansion_data_train.csv` | 5,326 | earlier-stage campaign compounds |
| `expansion_data_test.csv` | 2,282 | later-stage campaign compounds, labels released post-challenge |

Nine endpoints; a tenth (RLM CLint) appears only in the raw file and is excluded. MD5
digests are checked on download by `admetood.fetch_data`, which refuses to write a file
that does not match the published record.

**Verified properties of this snapshot** (`results/leakage_audit_expansionrx_official.json`):

- **No molecular-identity leakage.** 0 of 2,282 test molecules recur in training at any
  of five identity levels: canonical SMILES, InChIKey, salt/charge-stripped parent
  InChIKey, stereo-insensitive skeleton, tautomer-canonicalized skeleton.
- Murcko scaffold overlap 625/2,282 (27.4%), expected for analogue series.
- Nearest-neighbour Tanimoto to training: mean 0.613, median 0.625, 35.5% above 0.7,
  1.0% above 0.9. This is a **near-neighbour** shift, not a far-OOD one.

**Measured label shift** (training standard deviations; log10 for skewed endpoints):
Caco-2 Efflux +1.18, Caco-2 Papp −0.77, MPPB +0.31, HLM CLint +0.20, KSOL +0.18,
MGMB +0.09, MBPB +0.06, LogD +0.01, MLM CLint +0.00.

These three facts are the motivation and are reported as descriptive findings, not as
results of the hypothesis test.

## 4. Design

### 4.1 Reference standard

The real early-to-late split is the reference. It is the only split in this study that
was not constructed by us.

### 4.2 Simulated splits are built from the training set only

Every simulated split partitions the **5,326 training molecules** into `train'` and
`test'` at a 70/30 ratio, matching the real split's ratio (0.30). **The real test set is
never used to construct, tune or select anything.** It is opened once per prespecified
comparison, under the analysis plan.

Constructors, all prespecified:

| Name | Rule |
| --- | --- |
| `random` | uniform random assignment; the no-shift control |
| `scaffold` | Bemis–Murcko scaffold sets assigned whole, largest first (the TDC/MoleculeNet convention) |
| `butina` | Butina clustering at Tanimoto 0.65 on Morgan r=2 2048-bit; clusters assigned whole |
| `frontier` | hold out the molecules least similar to the bulk: rank by mean Tanimoto to all others, take the lowest 30% |
| `size` | hold out the 30% with the largest heavy-atom count |

`random` anchors the low-shift end and `frontier` the high-shift end; neither is expected
to match the real split, and that expectation is part of what is being tested.

### 4.3 Controlling for training-set size

A simulated split trains on ~3,728 molecules while the real comparison could train on all
5,326, so a naive comparison confounds shift with sample size. For every simulated split
of size `N'`, the real-split model is **also** fitted on a random subset of the training
set of size `N'`, averaged over the five model seeds. All comparisons between a simulated
split and the real split use size-matched models.

### 4.4 Endpoint handling

Each endpoint is modelled independently on the rows where it is measured; missing values
are never imputed. Skewed endpoints are modelled as log10 (KSOL, HLM/MLM CLint, both
Caco-2 endpoints, MPPB, MBPB, MGMB); LogD is modelled untransformed. Transformations are
fixed here and not revisited.

## 5. Models and uncertainty methods

Deliberately modest and reproducible on one consumer GPU. The paper's claim is about
**evaluation protocols**, not about a new architecture, so no new model is proposed.

**Representation.** Morgan count fingerprints (r=2, 2048 bits) concatenated with RDKit
physicochemical descriptors. One representation for all methods, so differences are
attributable to the uncertainty method rather than to features.

**Point predictors.**

| Model | Role |
| --- | --- |
| LightGBM | main regressor; strongest family on small ADMET data in published comparisons |
| Random forest | second family, variance-based uncertainty for free |

**Uncertainty methods.**

| Method | Type |
| --- | --- |
| Ensemble spread | 5 seeds, predictive standard deviation |
| Quantile regression | LightGBM at 5th/95th quantile |
| Split conformal | distribution-free, nominal coverage by construction on exchangeable data |
| Mondrian conformal | conformal within bins of nearest-neighbour similarity to training |
| Distance to training | mean Tanimoto to the k=5 nearest training molecules; applicability domain |

Split conformal is the critical case: it guarantees nominal coverage **only under
exchangeability**, which a prospective split violates by construction. Mondrian conformal
and the distance method are the prespecified candidate mitigations.

Gaussian-process regression with a Tanimoto kernel is prespecified as **exploratory**;
it is reported but cannot support a primary conclusion.

## 6. What this study does not claim

- No claim that any method is clinically or operationally safe.
- No claim about endpoints outside this dataset, or about far-OOD shift; the measured
  shift here is near-neighbour and from a single organization's campaigns.
- No new model or architecture is proposed.
- A single dataset cannot establish that structure-based splits fail in general. A
  negative transfer result here demonstrates that they *can* fail, and identifies the
  mechanism; generalization requires further prospective sets as they are released.

## 7. Known risks

- **Single source.** One company, one therapeutic area. Chemical diversity is narrower
  than a cross-pharma benchmark. Reported as the primary limitation.
- **Endpoint sample sizes differ by 20×.** Handled by the eligibility rule and by
  reporting per-endpoint confidence intervals rather than pooled averages.
- **Multiplicity.** Eight endpoints × five splits × five uncertainty methods. The
  analysis plan fixes one primary endpoint and applies a correction; everything else is
  explicitly secondary or exploratory.
- **The real split is used repeatedly.** Each prespecified comparison opens it, so it is
  not an untouched test set after the first analysis. No method may be modified after the
  first real-split result; doing so requires a dated amendment and reporting both.
