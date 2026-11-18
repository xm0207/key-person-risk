"""CEO and firm characteristics for every departure in the event sample (no WRDS connection needed).

Adds CEO power, firm power, network, career and pre-departure news measures (keyperson.power) to
data/derived/event_sample.parquet and writes data/derived/event_features.parquet (gitignored), which
run_departure_study.py uses. The exploratory tables this script used to write (as explore_sample.py) are now
made by archive/scripts/explore_sample.py.

    ~/.venvs/key-person-risk/bin/python scripts/build_event_features.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from keyperson import power

ROOT = Path(__file__).resolve().parents[1]
WRDS = ROOT / "data/wrds"
LAST_FOLLOW_UP_FYEAR = 2019  # five later fiscal years must be observable in ExecuComp
MEASURES = ["tenure_years", "age", "own_pct", "duality", "pay_slice", "log_mv", "sales_share", "prior_firms",
            "pre_ceo_stories", "pre_scandal_named", "news_before_gentry", "later_exec_5y", "later_ceo_5y"]


def build() -> pd.DataFrame:
    ev = pd.read_parquet(ROOT / "data/derived/event_sample.parquet")
    ev = ev[ev["permno"].notna()].copy()
    anncomp = pd.read_csv(WRDS / "execucomp_anncomp.csv", dtype={"gvkey": str, "execid": str})
    funda = pd.read_csv(WRDS / "comp_funda_annual.csv", dtype={"gvkey": str},
                        usecols=["gvkey", "fyear", "at", "sale", "csho", "prcc_f", "sich"])
    cols = ["rp_entity_id", "timestamp_utc", "type", "position_name", "headline", "event_text", "rp_story_id"]
    rp = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted((WRDS / "ravenpack").glob("dj_exec_*.parquet"))])
    rp = rp[rp["rp_entity_id"].isin(set(ev["rp_entity_id"].dropna()))]
    ev = power.ceo_power(ev, anncomp)
    ev = power.firm_power(ev, funda)
    ev = power.network_and_career(ev, anncomp)
    ev = power.news_before(ev, rp)
    # News first: a RavenPack story naming the CEO more than 3 days before Gentry's announcement date
    ev["news_before_gentry"] = np.where(ev["gentry_gap_days"].notna(), (ev["gentry_gap_days"] < -3).astype(float), np.nan)
    late = ev["fyear"] > LAST_FOLLOW_UP_FYEAR
    ev.loc[late, ["later_exec_5y", "later_ceo_5y"]] = np.nan
    ev["year"] = ev["event_date"].dt.year
    ev["duality"] = ev["duality"].astype(float)
    return ev


def main() -> None:
    ev = build()
    ev.to_parquet(ROOT / "data/derived/event_features.parquet", index=False)
    print(f"{len(ev):,} departures; coverage:", ev[MEASURES].notna().mean().round(2).to_dict())


if __name__ == "__main__":
    main()
