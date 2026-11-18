# The event-study library

Technical notes on `src/keyperson/`. The paper uses the engine with CRSP returns (Sections 4–7) and with Yahoo prices (Nvidia, Section 8).

## Engine

The model is the one in [mega7-decomposition](https://github.com/xm0207/mega7-decomposition) (`mega7.event_study`, commit `4e554eb`), installed as a dependency: the market model with an optional sector factor orthogonalised to the market,

```
r = alpha + beta_m m + beta_s s* + e,      s* = s - (a + b m)   (a, b from s on m in the estimation window)
```

so that alpha and beta_m equal the market-only estimates and the market-only abnormal return splits exactly into a sector part and the company-specific part e. `fit_model` re-implements the estimator so that it can also return the residuals and (X'X)^-1; a test checks that its coefficients equal mega7's.

- **Estimation window.** mega7's rule: 250 trading days ending 21 trading days before the first day of the event window. The event window is the union of all requested windows, so with the default pre-event window [−60, −2] the estimation window ends on day −81 and no pre-event day is used for estimation. `EstimationRule(anchor=...)` sets that first day explicitly; `min_observations` allows shorter histories.
- **Windows** (trading days relative to day 0, default): pre-event [−60, −2] and [−20, −2]; announcement [0, 0], [0, +1] and [−1, +1]; post-event [+2, +20] and [+2, +60]. A window cut by the end of the data is flagged (`complete = False`).
- **Day 0** (`dates.day_zero`): the announcement date if the news came before the open or during the session, otherwise the next trading day.
- **Per window:** CAR and its split into market, sector and alpha parts; z = CAR / (σ√L) as in mega7; a t-statistic using the standard error of a sum of L out-of-sample OLS prediction errors; the SQ test; and the same CAR, t and SQ p-value for the market-only model.
- **Input:** any DataFrame of daily simple returns (one column per security). `keyperson.data` loads them from this project's Yahoo database, and `keyperson.crsp` builds the same panel from CRSP.

## Tests of significance

- **SQ test** (`sq_test.py`). Gelbach, Helland and Klick (2013) test a single event-day abnormal return against the sample quantiles of the estimation-window residuals rather than a normal distribution. p-values use the rank convention (1 + count)/(N + 1), so with 250 residuals the smallest one-sided p-value is 1/251 and the smallest two-sided one 2/251 = 0.008. For a window of L days the CAR is compared with the sums of L consecutive estimation-window residuals; this multi-day version is this project's extension, not the paper's, and is approximate because the overlapping sums are dependent. A simulation in `tests/test_sq_test.py` checks the size: with 250 t(3) residuals and 6,000 draws, the SQ test rejects 0.8% of the time at the 1% level and the normal test 2.6%.
- **Cross-section** (`cross_section.py`): the cross-sectional t-test; the BMP standardised test (Boehmer, Masumeci and Poulsen 1991, as the publisher spells the second author; usually cited as Boehmer, Musumeci and Poulsen), using each event's own prediction-error standard error; the Kolari and Pynnönen (2010) adjustment for cross-correlation, with the average residual correlation over events whose estimation windows overlap (the formula follows the eventstudytools.com documentation and has not yet been checked against the paper); and a sign test.

## Priced-in share and post-event drift

`measures.py`, per event: priced-in share = CAR_pre / (CAR_pre + CAR_ann), and post-event ratio = CAR_post / CAR_ann (positive: continuation; negative: reversal). Both are undefined when the denominator is below a threshold. Because per-event ratios are noisy, the cross-section is summarised by sign-aligned ratios of sums, which weight events by the size of their move.

In the paper's main specification the intercept is set to zero (`car + alpha_part`), because an intercept estimated more than a year before the event would otherwise be projected over windows of up to 60 days. The full market model and market-adjusted returns are reported as robustness checks. The per-event priced-in ratio is noisy when the pre-event move is small; the scandal study therefore reports the share of the mean move (`measures.share_of_mean_move`) instead.
