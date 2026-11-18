"""Run the event-study engine on the verified mega-cap CEO events (events/megacap_events.csv).

This is a validation run of the engine on real data, not the study itself (whose sample is still being
designed). It writes summary statistics only (Yahoo data are personal use; mega7's ResultsWriter refuses
anything that looks like a daily series):

- archive/results/megacap_estimation.csv   one row per event and specification: estimation window and coefficients
- archive/results/megacap_event_windows.csv one row per event, specification and window: CAR, z, t, SQ p-values
- archive/results/megacap_event_profile.csv priced-in share and post-event ratio per event (primary specification)
- archive/results/engine_check_mega7.csv    the engine against mega7's published Nvidia result for 9 November 2026

Specifications: `primary` (market + the firm's sector proxy, orthogonalised), `ex_firm_basket` (market +
an equal-weight basket of the other six firms) and, for Nvidia, `alt_sector` (XLK instead of SMH). Every
row also carries the market-only CAR.
"""

from __future__ import annotations

import argparse

import pandas as pd

from mega7 import ResultsWriter

from keyperson.config import ALTERNATIVE_SECTOR, FIRMS, MARKET, MIN_BASKET_MEMBERS, SECTOR_PROXY
from keyperson.data import ex_firm_basket, load_returns
from keyperson.dates import day_zero
from keyperson.engine import EstimationRule, EventStudyEngine, Window
from keyperson.measures import event_profile

MEGA7_NVDA = {"date": "2026-11-09", "abnormal": -0.0844863, "z": -5.03154}  # mega7 results/event_study_days.csv
WINDOW_COLUMNS = [
    "event_id", "ticker", "spec", "sector", "window", "rel_start", "rel_end", "date_start", "date_end", "days",
    "complete", "sum_of_returns", "market_part", "sector_part", "alpha_part", "car", "z", "se", "t", "p_t",
    "sq_p_lower", "sq_p_upper", "sq_p_two_sided", "sq_reference_size", "car_market_only", "z_market_only",
    "t_market_only", "p_t_market_only", "sq_p_two_sided_market_only",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="data/keyperson.duckdb")
    parser.add_argument("--events", default="events/megacap_events.csv")
    parser.add_argument("--results", default="archive/results")
    args = parser.parse_args()

    returns = load_returns(args.db)
    for firm in FIRMS:
        returns[f"basket_ex_{firm}"] = ex_firm_basket(returns, FIRMS, firm, MIN_BASKET_MEMBERS)
    events = pd.read_csv(args.events, parse_dates=["day0"])
    events = events[events["day0"].notna()].copy()

    # Day 0 in the CSV must follow from the announcement date and its timing on SPY's trading calendar
    calendar = returns[MARKET].dropna().index
    for e in events.itertuples(index=False):
        derived = day_zero(calendar, e.announcement_date, e.timing)
        if derived != e.day0:
            raise ValueError(f"{e.event_id}: day 0 in the CSV is {e.day0.date()}, the timing rule gives {derived.date()}")
    print(f"day 0 of all {len(events)} events agrees with the announcement date and timing")

    specs = []
    for e in events.itertuples(index=False):
        specs.append({"event_id": e.event_id, "ticker": e.ticker, "day0": e.day0, "spec": "primary",
                      "sector": SECTOR_PROXY[e.ticker]})
        specs.append({"event_id": e.event_id, "ticker": e.ticker, "day0": e.day0, "spec": "ex_firm_basket",
                      "sector": f"basket_ex_{e.ticker}"})
        if e.ticker in ALTERNATIVE_SECTOR:
            specs.append({"event_id": e.event_id, "ticker": e.ticker, "day0": e.day0, "spec": "alt_sector",
                          "sector": ALTERNATIVE_SECTOR[e.ticker]})
    specs = pd.DataFrame(specs)

    engine = EventStudyEngine(returns, MARKET)
    tables, estimations = [], []
    for spec, rows in specs.groupby("spec", sort=False):
        windows, estimation, _ = engine.run(rows)
        windows.insert(2, "spec", spec)
        estimation.insert(2, "spec", spec)
        tables.append(windows)
        estimations.append(estimation)
    windows = pd.concat(tables, ignore_index=True)[WINDOW_COLUMNS]
    estimation = pd.concat(estimations, ignore_index=True)

    writer = ResultsWriter(args.results)
    writer.csv("megacap_estimation", estimation)
    writer.csv("megacap_event_windows", windows)
    primary = windows[windows["spec"] == "primary"]
    profile = event_profile(primary, "pre_20_2", "ann_m1_1", "post_2_20")
    writer.csv("megacap_event_profile", profile)

    # Check against mega7: the same model and estimation rule (anchored at 4 November 2026, day -3)
    check_engine = EventStudyEngine(returns, MARKET, EstimationRule(anchor=-3), (Window("day_0", 0, 0),))
    row = check_engine.run_event("NVDA-2026-11-08", "NVDA", MEGA7_NVDA["date"], "SMH").windows.iloc[0]
    check = pd.DataFrame([
        {"quantity": "abnormal return, 9 Nov 2026 (SMH)", "mega7": MEGA7_NVDA["abnormal"], "engine": row["car"]},
        {"quantity": "z, 9 Nov 2026 (SMH)", "mega7": MEGA7_NVDA["z"], "engine": row["z"]},
    ])
    check["difference"] = check["engine"] - check["mega7"]
    writer.csv("engine_check_mega7", check)

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(estimation[["event_id", "spec", "status", "day0", "estimation_start", "estimation_end", "beta_market",
                          "beta_sector", "sigma_daily"]].round(4).to_string(index=False))
        show = windows[windows["window"].isin(["day_0", "ann_m1_1"]) & (windows["spec"] == "primary")]
        print(show[["event_id", "window", "car", "z", "t", "sq_p_two_sided", "car_market_only"]].round(4).to_string(index=False))
        print(profile.round(3).to_string(index=False))
        print(check.to_string(index=False))


if __name__ == "__main__":
    main()
