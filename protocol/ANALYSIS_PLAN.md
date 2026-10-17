# Analysis plan

Version 1.0, written 17 October 2026, **before any model was fitted**. Companion to
[PROTOCOL.md](PROTOCOL.md). Amendments must be dated and appended, never substituted.

## 1. Eligible endpoints

Prespecified rule: an endpoint enters the primary analysis if it has **≥500 labelled
training rows and ≥250 labelled real-test rows**. Applied to the frozen snapshot this
admits eight endpoints and excludes one.

| Endpoint | Train | Real test | Model scale | Primary |
| --- | ---: | ---: | --- | :---: |
| LogD | 5,039 | 2,270 | linear | yes |
| KSOL | 5,128 | 2,170 | log10 | yes |
| MLM CLint | 4,522 | 1,170 | log10 | yes |
| HLM CLint | 3,759 | 782 | log10 | yes |
| Caco-2 Efflux | 2,161 | 1,616 | log10 | yes |
| Caco-2 Papp A>B | 2,157 | 1,616 | log10 | yes |
| MPPB | 1,302 | 454 | log10 | yes |
| MBPB | 975 | 451 | log10 | yes |
| MGMB | 222 | 209 | log10 | no — exploratory |

## 2. Primary endpoint

**Coverage transfer gap of split conformal prediction.**

Split conformal is calibrated to **90% nominal coverage**. For each simulated split `S`
and endpoint `E`:

```
Δ(S, E) = coverage on the simulated held-out set − coverage on the real prospective test set
```

Both sides use size-matched training sets (Protocol 4.3) and the same model, features,
seeds and calibration procedure. Δ is reported in percentage points.

If a structure-based split were an adequate surrogate for prospective shift, Δ would be
near zero. The primary analysis is **40 comparisons**: 5 splits × 8 endpoints.

### Equivalence margin

Prespecified at **±5 percentage points**. A 90% interval that empirically covers between
85% and 95% is treated as practically equivalent; outside that band the surrogate is
treated as misleading for decision-making. The margin is fixed here and may not be
revised after seeing results.

### Decision rule, per comparison

Using the 95% bootstrap confidence interval of Δ:

| Verdict | Condition |
| --- | --- |
| **transfers** | CI lies entirely inside ±5 pp |
| **fails to transfer** | CI lies entirely outside ±5 pp (on either side) |
| **inconclusive** | CI straddles a margin boundary |

### Headline result

For each simulated split, the count of eligible endpoints in each verdict category. The
hypothesis in Protocol Section 1 is **supported** if, for every structure-based split
(`scaffold`, `butina`, `frontier`, `size`), at least four of eight endpoints are
"fails to transfer". It is **rejected** if any structure-based split achieves "transfers"
on at least six of eight endpoints. Any other pattern is reported as **mixed**, with no
general claim.

## 3. Secondary endpoints

Reported with confidence intervals; none may rescue a null primary result.

1. **Absolute coverage under real shift.** Empirical coverage of the 90% interval on the
   real test set, per endpoint and uncertainty method. Tests directly whether conformal
   guarantees survive a prospective split.
2. **Decision value — risk–coverage.** Fraction of real-test compounds retained when
   abstaining on the most uncertain predictions, at a target absolute error of
   **0.5 log10 units** (about three-fold; for LogD, 0.5 log units). Reported as retained
   fraction at the target, and as the full risk–coverage curve. The literature scan found
   no ADMET study reporting this quantity prospectively.
3. **Error-ranking quality.** Spearman correlation between predicted uncertainty and
   absolute error, per endpoint, on the real test set and on each simulated held-out set.
4. **Method-ranking agreement.** Spearman correlation between the ranking of the five
   uncertainty methods by coverage error under a simulated split and under the real split,
   per endpoint. Reported with the caveat that five methods give a coarse ranking.
5. **Point accuracy transfer.** RMSE and Spearman of predictions, simulated versus real,
   to show how much of the transfer gap is accuracy rather than calibration.

## 4. Uncertainty and multiplicity

- **Bootstrap.** 10,000 resamples, seed `20261018`, resampling **molecules within an
  endpoint**, paired across the compared conditions so Δ is resampled directly. Percentile
  intervals.
- **Multiplicity.** The primary family is the 40 comparisons of Section 2. Verdicts are
  reported both at nominal 95% and after Benjamini–Hochberg control of the false discovery
  rate at 5% across the family. The headline count uses the **adjusted** intervals.
- Secondary endpoints are not corrected and are labelled exploratory-strength evidence.

## 5. Reporting commitments

- Every eligible endpoint is reported, including those that contradict the hypothesis.
- Per-molecule predictions, uncertainties and seeds are saved for every run.
- Failed, degenerate or numerically unstable runs are reported, never silently dropped.
- The `random` split is reported alongside the structure-based splits; if `random`
  transfers as well as the structure-based splits, that is evidence the structure-based
  machinery adds nothing, and must be stated plainly.
- Exact counts are reported even when a confidence interval cannot be estimated.

## 6. Fail-closed conditions

The analysis stops and the condition is published rather than worked around if:

- the data checksums do not match the frozen lock;
- a simulated split yields a held-out set smaller than 250 labelled rows for an endpoint
  that was declared eligible;
- split conformal does not attain 88–92% coverage on a **random** split, which would
  indicate an implementation error rather than a shift effect;
- any test-set row is found in training at any identity level during the final audit.

## 7. Sequence

1. Create the lock (`protocol/lock.py`), recording digests of the data, this plan, the
   protocol and the source modules.
2. Build and freeze the simulated splits from the training set only; publish their
   descriptive statistics, including the shift measures of Protocol Section 3.
3. Fit models and uncertainty methods on the simulated splits. The real test set stays
   closed.
4. Open the real test set once, for the prespecified comparisons.
5. Report. Any change after step 4 requires a dated amendment and the publication of
   results under both the original and the amended analysis.
