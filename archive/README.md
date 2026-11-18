# Archive

Earlier phases of the project and checks that the paper does not use. Everything here still runs from the project folder, for example `python archive/scripts/run_megacap_events.py`. The scripts write to `archive/results/`.

## How the project got here

1. **Seven mega-caps (16 November 2026).** The project began as a study of CEO events at Apple, Microsoft, Alphabet, Amazon, Nvidia, Meta and Tesla. Its purpose was to build and check the event-study engine. Nineteen events were too few for a paper, and the question was widened to all US listed firms.
2. **All CEO departures (16–17 November).** All CEO departures in the Gentry et al. database were linked to CRSP, ExecuComp and RavenPack and studied by reason for leaving. That study is still part of the pipeline (`scripts/run_departure_study.py`), because the paper uses its comparison of departures over personal conduct and over corporate wrongdoing. Its checks and exploratory tables are archived here.
3. **Firm-level misconduct episodes (17 November).** Regulatory and legal episodes at the firm turned out to involve the chief executive personally only rarely (the CEO left within a year in 13% of 1,327 episodes). They were set aside.
4. **CEO personal scandals (17–18 November).** The question narrowed to scandals like Nvidia's, which is the paper.

## Contents

| Script | What it did | Output |
|---|---|---|
| `run_megacap_events.py` | the engine on 19 dated CEO events at the seven mega-caps (`events/megacap_events.csv`), with sector-fund, ex-firm basket and market-only benchmarks, and the check against mega7's published Nvidia result | `megacap_*.csv`, `engine_check_mega7.csv` |
| `check_departure_study.py` | checks on the departure study: pre-window drift, daily profiles by reason | `departure_checks.csv`, `departure_daily_profile.csv` |
| `build_episodes.py` | firm-level misconduct episodes from RavenPack | `episodes_summary.csv` |
| `explore_sample.py` | first look at the departure sample: characteristics by reason, departures by year, two linear-probability models | `explore_*.csv` |
| `explore_angles.py` | printed comparisons that led to the paper's splits by kind of misconduct, earnings timing and consumer-facing industries. The paper's numbers now come from `scripts/paper_numbers.py` | printed only |
| `wrds_explore.py`, `wrds_ravenpack_taxonomy.py` | lists of WRDS libraries, tables and RavenPack categories | `data/wrds/explore/` |

`docs/archive/` holds the working notes of 16–17 November. Their numbers were superseded by later corrections. The paper and `results/` are current.

## Mega-cap events (`events/megacap_events.csv`)

19 dated events and one "none found" row for Meta. For each: announcement date, time relative to market hours, day 0, a primary source (an SEC filing or company release where one exists, otherwise news), a source for the timing, and any confounding news within ±1 trading day. EDGAR acceptance times are converted from UTC to New York time. `archive/scripts/run_megacap_events.py` checks that every day 0 follows from the announcement date and timing on SPY's trading calendar.

| Status | Events |
|---|---|
| included (14) | Apple 14 Jan 2009 (medical leave), 24 Aug 2011 (resignation), 5 Oct 2011 (death of the chairman and former CEO); Microsoft 23 Aug 2013 (retirement), 4 Feb 2014 (new CEO); Alphabet 3 Dec 2019 (founders step back); Amazon 9 Jan 2019 (divorce announcement), 4 Apr 2019 (settlement terms, 8-K); Tesla 7 Aug 2018 ("funding secured"), 24 Aug 2018 (going-private abandoned), 25 Apr 2022 (Twitter agreement), 5 Jun 2025 (public dispute with the President); Nvidia 8 Nov 2026 (board investigation), 12 Nov 2026 (CEO resignation) |
| included, confounded within ±1 day (4) | Apple 17 Jan 2011 (leave; earnings after the close on day 0); Tesla 27 Sep 2018 (SEC complaint; settlement on day +1), 29 Sep 2018 (settlement; preliminary Q3 information on day 0), 1 May 2020 ("stock price too high"; earnings reaction on day −1) |
| excluded (1) | Amazon 2 Feb 2021 (CEO transition announced in the earnings release) |
| none found | Meta: no CEO departure, health or personal-conduct event identified |

Nvidia's board announced the CEO's resignation on 12 November 2026, before the open, after the 10 November cut-off of mega7's data; this project's prices now run to 17 November 2026, so post-event windows for both Nvidia events are incomplete. The board's investigation was ongoing at that date, with no findings published. The CSV names no private individual and gives no reason for the resignation, because the company's release gave none.

## Engine check on real data

- **Agreement with mega7** (`archive/results/engine_check_mega7.csv`). With mega7's anchor (4 November 2026), the engine's abnormal return for Nvidia on 9 November 2026 differs from mega7's published value by 1.3e-8, and its z-score by 8e-6. The differences come from a fresh Yahoo download: daily returns up to 10 November agree with mega7's database to within 2e-6.
- **Sector ETFs hold the firm** (`megacap_event_windows.csv`). On 5 June 2025, with XLY as the sector factor (estimated sector beta 3.97, `megacap_estimation.csv`), Tesla's day-0 company-specific return is −5.5%. With an equal-weight basket of the other six firms it is −13.0%, and with the market alone −13.2%. A sector factor that contains the firm absorbs part of the firm's own shock. A broad sample needs sector or industry returns that exclude the event firm.
