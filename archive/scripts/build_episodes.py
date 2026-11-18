"""Build the misconduct-episode sample from local files (no WRDS connection needed).

    ~/.venvs/key-person-risk/bin/python scripts/build_episodes.py

Outputs: data/derived/misconduct_episodes.parquet (gitignored); archive/results/episodes_summary.csv (counts only).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from keyperson import episodes, sample

ROOT = Path(__file__).resolve().parents[2]
WRDS = ROOT / "data/wrds"


def main() -> None:
    cols = ["timestamp_utc", "rp_entity_id", "entity_name", "type", "event_sentiment_score", "news_type",
            "relevance", "headline", "rp_story_id"]
    rp = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted((WRDS / "stage3/ravenpack").glob("dj_channels_*.parquet"))])
    ep = episodes.build_episodes(episodes.qualifying_stories(rp))
    deps = pd.read_parquet(ROOT / "data/derived/event_sample.parquet")
    anncomp = pd.read_csv(WRDS / "execucomp_anncomp.csv", dtype={"gvkey": str, "execid": str})
    ep = episodes.attach_ceo_and_outcome(ep, deps[["rp_entity_id", "gvkey"]].dropna(), anncomp, deps)
    ep = sample.link_permno(ep.assign(ref_date=ep["event_date"]),
                            pd.read_csv(WRDS / "ccm_link_table.csv", dtype={"gvkey": str})).drop(columns="ref_date")
    ep.to_parquet(ROOT / "data/derived/misconduct_episodes.parquet", index=False)

    ok = ep[ep["co_per_rol"].notna()]
    summary = pd.DataFrame([
        {"statistic": "episodes", "value": len(ep)},
        {"statistic": "firms", "value": ep["rp_entity_id"].nunique()},
        {"statistic": "episodes with a PERMNO", "value": int(ep["permno"].notna().sum())},
        {"statistic": "episodes with the sitting CEO identified", "value": len(ok)},
        {"statistic": "CEO left within a year (any reason), share", "value": round(ok["ceo_left_within_year"].mean(), 4)},
        {"statistic": "CEO dismissed within a year (codes 3-4), share", "value": round(ok["departure_code"].isin([3, 4]).mean(), 4)},
        {"statistic": "episodes with 3+ stories in the first 30 days", "value": int((ep["stories_30d"] >= 3).sum())},
    ])
    summary.to_csv(ROOT / "archive/results/episodes_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(ok.groupby(ok["stories_30d"].clip(upper=5))["ceo_left_within_year"].agg(["mean", "count"]).round(3))
    print(ok["exit_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
