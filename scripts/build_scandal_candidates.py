"""Candidate CEO personal-scandal episodes for hand coding (local files only).

    ~/.venvs/key-person-risk/bin/python scripts/build_scandal_candidates.py

Writes data/derived/scandal_candidates.csv and, for the second pass with the whole-word list,
data/derived/scandal_candidates_strict.csv (both gitignored: it holds Dow Jones headlines, which may not be
redistributed). The committed coding file, events/ceo_scandal_coding.csv, holds identifiers and codes only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from keyperson import sample, scandals

ROOT = Path(__file__).resolve().parents[1]
WRDS = ROOT / "data/wrds"


def entity_map() -> pd.DataFrame:
    """RavenPack entity for every ExecuComp firm, through the Compustat CUSIP (keyperson.sample)."""
    anncomp = pd.read_csv(WRDS / "execucomp_anncomp.csv", usecols=["gvkey", "year"], dtype={"gvkey": str})
    firms = anncomp.groupby("gvkey")["year"].max().reset_index()
    firms["gvkey"] = firms["gvkey"].str.zfill(6)
    firms["ref_date"] = pd.to_datetime(firms["year"].astype(str) + "-06-30")
    funda = pd.read_csv(WRDS / "comp_funda_annual.csv", usecols=["gvkey", "fyear", "cusip"], dtype={"gvkey": str, "cusip": str})
    linked = sample.link_rp_entity(firms, funda, pd.read_parquet(WRDS / "ravenpack/company_mappings_hist.parquet"))
    return linked[["rp_entity_id", "gvkey"]].dropna()


def main() -> None:
    cols = ["timestamp_utc", "rp_story_id", "rp_entity_id", "entity_name", "relevance", "event_sentiment_score",
            "type", "news_type", "headline"]
    heads = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted((WRDS / "stage4/ravenpack").glob("dj_ceo_scandal_*.parquet"))])
    em = entity_map()
    anncomp = pd.read_csv(WRDS / "execucomp_anncomp.csv", dtype={"gvkey": str, "execid": str})
    stories = scandals.candidate_stories(heads, em, scandals.sitting_ceos(anncomp))
    ep = None
    for name, s in [("scandal_candidates.csv", stories), ("scandal_candidates_strict.csv", scandals.strict_stories(stories))]:
        out = ROOT / "data/derived" / name
        prev = pd.read_csv(out, dtype={"gvkey": str, "execid": str}) if out.exists() else None
        e = scandals.stable_ids(scandals.build_episodes(s), prev)
        if prev is not None:
            new = e[~e["candidate_id"].isin(prev["candidate_id"])]
            gone = prev[~prev["candidate_id"].isin(e["candidate_id"])]
            print(f"{name}: {len(new)} new episodes {list(new['candidate_id'])}, {len(gone)} no longer found {list(gone['candidate_id'])}")
        e.sort_values("first_ts").to_csv(out, index=False)
        ep = e if ep is None else ep
    print(f"{len(heads):,} headline rows; {em['gvkey'].nunique():,} ExecuComp firms linked to RavenPack; "
          f"{len(stories):,} stories naming the sitting CEO; {len(ep):,} episodes; {ep['execid'].nunique():,} CEOs")
    print(ep["keyword_class"].value_counts().to_string())
    print(ep.groupby(ep["first_ts"].dt.year).size().to_string())


if __name__ == "__main__":
    main()
