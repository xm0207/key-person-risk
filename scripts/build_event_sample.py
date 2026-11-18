"""Build the CEO-departure event sample from local files (no WRDS connection needed).

Inputs (all gitignored):
    data/external/gentry/CEO Dismissal Database Posted to Web 10Oct24.xlsx   Gentry et al. (CC BY 4.0)
    data/wrds/ccm_link_table.csv, data/wrds/comp_funda_annual.csv
    data/wrds/ravenpack/company_mappings_hist.parquet, dj_exec_*.parquet

Outputs:
    data/derived/event_sample.parquet   one row per departure, with PERMNO, RavenPack entity and event date
    results/event_sample_summary.csv    counts and link rates by departure reason (summary only)

    ~/.venvs/key-person-risk/bin/python scripts/build_event_sample.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from keyperson import sample

ROOT = Path(__file__).resolve().parents[1]
GENTRY = ROOT / "data/external/gentry/CEO Dismissal Database Posted to Web 10Oct24.xlsx"
WRDS = ROOT / "data/wrds"


def main() -> None:
    events = sample.load_gentry(pd.read_excel(GENTRY))
    events = sample.link_permno(events, pd.read_csv(WRDS / "ccm_link_table.csv", dtype={"gvkey": str}))
    funda = pd.read_csv(WRDS / "comp_funda_annual.csv", usecols=["gvkey", "fyear", "cusip"],
                        dtype={"gvkey": str, "cusip": str})
    events = sample.link_rp_entity(events, funda, pd.read_parquet(WRDS / "ravenpack/company_mappings_hist.parquet"))

    ids = set(events["rp_entity_id"].dropna())
    cols = ["rp_entity_id", "timestamp_utc", "type", "position_name", "headline", "event_text", "rp_story_id"]
    rp = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted((WRDS / "ravenpack").glob("dj_exec_*.parquet"))])
    rp = rp[rp["rp_entity_id"].isin(ids)]
    events = sample.choose_event_date(sample.first_departure_news(events, rp))

    out = ROOT / "data/derived"
    out.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out / "event_sample.parquet", index=False)

    g = events.groupby(["departure_code", "reason"])
    summary = pd.DataFrame({
        "departures": g.size(),
        "permno_linked": g["permno"].apply(lambda s: s.notna().sum()),
        "ravenpack_linked": g["rp_entity_id"].apply(lambda s: s.notna().sum()),
        "dated_by_ravenpack": g["date_source"].apply(lambda s: (s == "ravenpack").sum()),
        "gentry_date_available": g["date_departure_announced"].apply(lambda s: s.notna().sum()),
        "median_abs_gap_days_vs_gentry": g["gentry_gap_days"].apply(lambda s: s.abs().median()),
        "share_within_3_days_of_gentry": g["gentry_gap_days"].apply(
            lambda s: round((s.abs() <= 3).sum() / s.notna().sum(), 3) if s.notna().any() else None),
        "first_year": g["event_date"].min().dt.year,
        "last_year": g["event_date"].max().dt.year,
    }).reset_index()
    summary.to_csv(ROOT / "results/event_sample_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"{len(events)} departures; {events['permno'].notna().sum()} with a PERMNO; "
          f"{events['permno'].nunique()} firms")


if __name__ == "__main__":
    main()
