> **Superseded working note.** Written on 17 November 2026 during the project; later corrections changed several numbers and definitions. The current results are in the paper (`paper/personal_scandals.tex`) and `results/`.

# CEO personal scandals, 2000–2024: when does the market find out, and what does it cost?

The figures come from `scripts/run_personal_scandals.py`. The tables are `results/personal_scandal_*.csv`, and the hand coding is in `events/`.

## Sample
The sample has 90 personal-conduct scandals at US listed firms that have CRSP returns:
- 81 exits that Gentry et al. code as misconduct and that I read as personal conduct, not corporate fraud (`events/gentry_misconduct_coding.csv`);
- RavenPack episodes coded P1, which add CEOs who stayed and cases Gentry does not cover;
- six hand checks against RavenPack headlines (`events/personal_scandal_overrides.csv`): four earlier first-news dates and two duplicates.

Personal conduct overtook corporate fraud as the misconduct reason for CEO exits:
- **2000–09:** 75 corporate cases against 35 personal;
- **2015–22:** 17 corporate against 34 personal.

## Three ways a scandal plays out

| Pattern | Cases | Mean CAR [−20, −2] | Mean CAR at first news [−1, +1] | Mean CAR first news → exit | Mean CAR at exit | Mean total |
|---|---|---|---|---|---|---|
| One-stage: the scandal becomes public with the exit | 72 | −2.4% | −3.0% (t −2.8; median −1.3%) | — | — | −5.3% (t −2.3) |
| Two-stage: an allegation or board probe is public first, the exit follows (median 25 days later) | 6 | −2.2% | −8.2% (median −4.2%) | −2.9% | +0.5% | −12.3% (median −1.5%) |
| The CEO stays at least a year | 7 | −2.5% | +0.7% | — | — | −1.8% |

Four further cases are kept apart: the CEO left within a year, but for a stated reason other than the scandal (Gentry codes performance or retirement). One case has an unknown outcome.

## What it suggests
1. **In two-stage cases the price moves on the allegation, not on the exit.** The exit window averages +0.5%, so essentially all of the loss comes before the CEO goes. This is the Nvidia pattern of November 2026: about 80% of the company-specific fall came before the resignation.
2. **Who gets a two-stage scandal, or survives one, is not random.**

   | | One-stage | Two-stage | Survived |
   |---|---|---|---|
   | Share of founders | 3% | 33% | 57% |
   | Mean CEO tenure, years | 8 | 12 | 14 |
   | Firm size, mean log market value | 8.0 | 9.4 | 10.2 |

   Powerful CEOs at large firms see their scandals play out in public, or they survive them. Less powerful CEOs at smaller firms are investigated quietly and leave the day it is announced.
3. **Survivors barely move the price at the first news (+0.7%).** Either the allegations were less serious, or the market expected these CEOs to stay. With seven cases the two cannot be separated.

## Caveats
- **The two-stage and survivor groups are small**, with six and seven cases. They are case evidence, not statistics.
- **Totals are dominated by single firms.** For example, AdaptHealth's −53% coincides with a collapse in its share price.
- **First-news dates come from Dow Jones headlines** that name the CEO. Earlier reports elsewhere would turn some one-stage cases into two-stage ones. Chesapeake's loan story, for instance, appeared in other outlets two days before the first Dow Jones story found here.
- **The founder flag is incomplete.** It relies on ExecuComp titles that mention "founder", so it misses founders whose titles do not.
- **CRSP returns end in 2024.** Nvidia (2026) is therefore a case study measured on other data, not part of the sample.
