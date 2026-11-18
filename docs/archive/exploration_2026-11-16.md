> **Superseded working note.** Written on 16 November 2026 during the project; later corrections changed several numbers and definitions. The current results are in the paper (`paper/personal_scandals.tex`) and `results/`.

# First look at the CEO-departure sample (16 Nov 2026, before any returns)

The figures come from `scripts/explore_sample.py` and the tables are in `results/explore_*.csv`.

- **Sample:** departures from Gentry et al. (2021) since 2000 that link to CRSP: 4,578 departures at 2,426 firms.
- **Method:** all regressions are linear probability models with year fixed effects and errors clustered by firm.
- **Status:** the results are descriptive, not causal.

## What the data say so far
1. **Misconduct dismissals cluster in time.**
   - 2002–2006 had 12–18 a year (the years after Enron and the option-backdating cases).
   - 2018 had 16, against 2–11 in the neighbouring years.
   - Performance dismissals run at 50–80 a year.
2. **Which departures are dismissals.** This model compares dismissals (for performance or misconduct) with voluntary exits (retirement or a new job). n = 3,216, of which 32% are dismissals.
   - CEOs who also chair the board are 15 points less likely to have been dismissed (CI −19 to −12).
   - Each extra year of age lowers the probability by 1.6 points.
   - Each log unit of market value lowers it by 2.4 points.
   - Each earlier employer in ExecuComp raises it by 4 points. Outsiders are fired more often.
   - Ownership and pay slice show no effect at conventional levels.
3. **Careers afterwards.** n = 2,933 departures up to fiscal 2019. Of these, 5.7% reappear as an executive at another ExecuComp firm within five years.
   - A misconduct dismissal lowers the chance by 4.8 points (CI −7.9 to −1.7), almost wiping out the base rate.
   - With age held fixed, a performance dismissal does no worse than a retirement.
   - Network and stature help. Each earlier ExecuComp employer adds 4.6 points, and each log unit of the old firm's market value adds 1.2 points.
4. **News before the departure.**
   - In the year before a misconduct dismissal, the firm had 4.2 CEO-related RavenPack stories on average, against 3.7 before a performance dismissal and 2.3 before a retirement.
   - RavenPack's executive-scandal tag almost never names the CEO before the departure, so on its own it cannot show that trouble was already public. That will need the legal and regulatory news from the stage-3 pull.
5. **Leaks.** This uses departures where both dates exist.
   - For 12% of performance dismissals, a RavenPack story naming the CEO came more than 3 days before Gentry's announcement date. For misconduct dismissals the figure is only 5%.
   - Misconduct exits are sudden; performance exits are more often rumoured first. The priced-in share should therefore be measured separately by reason.

## Caveats
- **Career rates are lower bounds.** ExecuComp sees only the top five executives of S&P 1500 firms, so "reappears" misses private firms, smaller listed firms and board seats.
- **Duality may be overstated for retirements.** It is read from the CEO's title in their last full year. Retiring CEOs often hold the chair during a planned handover.
- **The sample is conditioned on departure.** Model 2 compares kinds of exit; it does not estimate the hazard of being fired. A hazard model needs all CEO-years, which ExecuComp provides. This is a next step.

## Next steps, once CRSP daily returns are in (scripts/wrds_pull_stage3.py)
- **Returns by reason:** abnormal returns around the first news and around the announcement, and the share priced in before the announcement.
- **Power:** whether CEO power (duality, tenure, ownership) and firm power (size, sales share) change the announcement return and the drift before it.
- **Channels:** from the stage-3 RavenPack news, event by event, look for insider buys, buybacks, rights plans, board changes and settlements around misconduct cases.
