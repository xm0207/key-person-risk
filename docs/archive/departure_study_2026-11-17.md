> **Superseded working note.** Written on 17 November 2026 during the project; later corrections changed several numbers and definitions. The current results are in the paper (`paper/personal_scandals.tex`) and `results/`.

# CEO departures, 2000–2024: first event-study results (17 Nov 2026)

The figures come from `scripts/run_departure_study.py`, and the tables are in `results/departure_*.csv`.

## Set-up
- **Sample:** 4,470 departures with a full estimation window: 1,290 performance dismissals, 192 misconduct dismissals, 63 deaths, 93 illnesses, 2,704 retirements and 128 departures for a new job.
- **Model:** the market model on the Fama–French market return, estimated over trading days [−390, −141].
- **Delisting returns** are included.
- **CRSP coverage:** CRSP daily data in the legacy format end on 31 Dec 2024.
- **Day 0:** the first trading day the news could move. This is the first RavenPack story naming the CEO; failing that, Gentry et al.'s announcement date; failing that, the day the CEO left office.

## Mean CARs

| Reason | n | [−120, −2] | [−20, −2] | [−1, +1] | [+2, +60] |
|---|---|---|---|---|---|
| Dismissed, performance | 1,290 | −10.8% (t −8.4) | −2.9% (t −5.7) | −1.3% (t −3.4) | +2.6% (t 2.6) |
| Dismissed, misconduct | 192 | −8.6% (t −2.8) | −3.0% (t −2.7) | −3.4% (t −4.6) | −0.1% |
| Left for a new job | 128 | −2.7% | −0.5% | −3.0% (t −4.9) | +2.4% |
| Retired | 2,704 | −0.6% | −0.4% | −0.3% (t −2.7) | −1.1% (t −2.7) |
| Illness | 93 | −2.4% | −0.7% | −0.9% | −1.1% |
| Death | 63 | −0.4% | +1.4% | −0.2% | −1.2% |

## Share of the move that came before the news

This is mean CAR before ÷ (mean CAR before + mean announcement CAR), with bootstrap 95% CIs from 2,000 draws.

| Reason | [−20, −2] | [−60, −2] | [−120, −2] |
|---|---|---|---|
| Dismissed, performance | 0.75 (0.59–0.95) | 0.88 (0.79–0.98) | 0.92 (0.86–0.98) |
| Dismissed, misconduct | 0.50 (0.21–0.70) | 0.65 (0.29–0.82) | 0.76 (0.48–0.88) |
| Left for a new job | 0.13 (−0.55 to 0.46) | — | — |

The share is not defined for retirements, deaths and illnesses: their total moves are too close to zero, so the intervals are unbounded.

## Correction after the sanity checks (`scripts/check_departure_study.py`, `results/departure_checks.csv`)
The shares above mix events with different dating. When an event is dated only by the day the CEO left, the
real announcement can fall inside the pre-event window. In those cases the share priced in is pushed towards 1.
- **Performance dismissals.** When dated only by the exit day (459 events), they show an announcement CAR of 0.0% (t 0.0). Their share priced in is therefore 1.0 by construction.
- **News-dated events only** (a RavenPack story or Gentry's announcement date):

| Reason | n | Mean CAR [−20, −2] | Mean CAR [−1, +1] | Share before, [−20, −2] | Share before, [−60, −2] |
|---|---|---|---|---|---|
| Dismissed, performance | 825 | −2.4% | −2.0% | 0.55 (0.39–0.68) | 0.77 (0.68–0.84) |
| Dismissed, misconduct | 119 | −3.0% | −2.6% | 0.53 (0.18–0.78) | 0.63 (0.14–0.84) |
| Left for a new job | 93 | +0.2% | −4.0% | −0.06 (−1.0 to 0.28) | 0.17 |

With dating held fixed, performance and misconduct exits are both about half priced in over the final month. The large gap in the first table was mostly a dating artefact.

**From now on, the primary sample is news-dated events.** Misconduct exits dated by the exit day are kept as a separate group. Gentry et al. describe these exits as almost always immediate, and indeed they show the largest reaction, −4.6% (t −3.9, n = 73).

## Checks against economic logic

| Prediction | Evidence | Verdict |
|---|---|---|
| Returns are bounded below by −100%, and extreme days are rare | Minimum −99.9%; 0.01% of days move more than 50% | Pass |
| Single-stock betas are near 1 and R² is around 0.1–0.4 | Mean betas 1.04–1.20; median R² 0.18–0.25 | Pass |
| Boards fire CEOs after long underperformance | Estimation-window alpha for performance dismissals is −14.7% a year, a year before the exit | Pass. The +2.6% post-event "recovery" is therefore an alpha artefact (−0.5% with alpha set to zero) |
| CEOs retire after good runs | Retirement alpha is +3.7% a year | Pass |
| With correct dates, the reaction spikes on days 0 and +1 | Day 0: performance −1.4%, new job −2.4%, retirement −0.3%. Misconduct is spread over days −1, 0 and +1 (−0.7%, −1.2%, −1.6%) | Pass. Misconduct dates are less precise |
| An exit date is not news, except for immediate exits | Exit-dated: performance 0.0%, new job −0.3%, but misconduct −4.6% | Pass |
| Results do not hinge on outliers, the crisis or small caps | Medians are smaller (the distribution is skewed), but winsorised means, results without 2008–09 and results without the smallest quarter keep the same sign and size | Pass |
| A worse run-up leaves less to learn at the announcement | Performance dismissals: the worst third by run-up has the *most* negative announcement CAR (−2.2%, against −0.7% and −1.3%); the slope is about 0 | **Fails.** Likely confound: firings announced together with bad earnings or restructurings. Testing it needs earnings announcement dates (Compustat quarterly `rdq`) |
| A CEO leaving for another job is a talent loss, not an insider fleeing a sinking ship | No run-up (+0.2%), −4.0% at the news, no further decline afterwards | Consistent with talent loss |

## What this says so far
1. **Performance dismissals follow long declines.** The stock falls about 11% in the six months before (18% with alpha set to zero). On news-dated events the announcement adds a further −2.0%, and over the final month about half of the move comes first (see the correction above).
   - The +2.6% over the next 60 days disappears when alpha is set to zero (−0.5%). It is therefore probably the estimation window's negative alpha, not a recovery.
2. **Misconduct dismissals are about half priced in over the final month, like performance dismissals once dating is held fixed.** Immediate exits react most (−4.6%).
   - In the cross-section, with year effects, a misconduct exit adds −2.9 points to the announcement CAR relative to a retirement (CI −4.3 to −1.6).
3. **Losing a CEO to another job costs about 3% at the announcement, almost all of it unanticipated.**
   - This is the cleanest key-person signal in the data. Nothing is wrong at the firm; the market simply loses the person.
4. **Deaths show no average effect** (n = 63). That fits the view that a sudden loss can be good or bad news depending on the CEO. Testing it needs the power interactions, with more statistical power than 63 events give.
5. **CEO and firm power do not explain the announcement CAR in this first pass.**
   - None of duality, ownership, size or sales share is significant.
   - The interactions of misconduct with duality, size and tenure are not significant either.
   - Tenure is weakly positive (p = 0.06) and pay slice weakly negative (p = 0.09).

## A methodological finding
The sign-aligned aggregate priced-in share (`measures.aggregate_priced_in_share`) is unreliable here.
- It weights each event by |CAR_pre + CAR_ann|. With a long pre-event window, that weight is mostly noise, so the share drifts towards 1.
- For example, retirements, whose mean CARs are near zero, score 0.97 on [−120, −2].
- The ratio of means (`measures.share_of_mean_move`) lets the noise cancel. A test reproduces the problem on simulated data.

## Next
- **Channels:** go through the stage-3 RavenPack news around the 192 misconduct cases (insider buys, buybacks, rights plans, board changes, settlements), and compare cases where trouble was public first with sudden exits.
- **Scandals the CEO survived:** the current sample contains departures only. A full test of power needs those cases too, found from legal and regulatory news naming the CEO.
- **Robustness:**
  - Fama–French three-factor and momentum models.
  - An estimation window after the event.
  - Excluding departures that coincide with earnings announcements.
- **Hazard model:** CEO power and the probability of dismissal, using all CEO-years in ExecuComp.
