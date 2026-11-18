"""Misconduct episodes at the firm: the first news of a regulatory investigation, fraud, corruption or discrimination
allegation, and whether the sitting CEO survives it.

An episode starts with a RavenPack story that
  * has one of the misconduct types below, negative event sentiment, and relevance >= 90 to the firm;
  * is not a press release (law-firm "investor alerts" are press releases);
  * comes at least `GAP_DAYS` after the firm's previous qualifying story.
The CEO in office is the ExecuComp CEO of that fiscal year. The CEO "leaves within a year" when Gentry et al.
record that CEO's departure (any reason) within 365 days of the first story.

This is a firm-level design: the allegation concerns the firm, under the CEO's watch, not necessarily the CEO
personally. RavenPack rarely ties legal news to a named, sitting CEO (see docs/departure_study_2026-11-17.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MISCONDUCT_TYPES = {"fraud", "regulatory-investigation", "corruption", "embezzlement", "tax-evasion",
                    "blackmail", "discrimination"}
GAP_DAYS = 365
MIN_RELEVANCE = 90
FOLLOW_DAYS = 365


def qualifying_stories(rp: pd.DataFrame) -> pd.DataFrame:
    s = rp[rp["type"].isin(MISCONDUCT_TYPES) & rp["news_type"].ne("PRESS-RELEASE")
           & (rp["relevance"] >= MIN_RELEVANCE) & (rp["event_sentiment_score"] < 0)].copy()
    s["ts"] = pd.to_datetime(s["timestamp_utc"])
    return s.sort_values("ts").drop_duplicates(["rp_entity_id", "rp_story_id"])


def build_episodes(stories: pd.DataFrame, start: str = "2001-01-01") -> pd.DataFrame:
    """One row per episode, with the number of qualifying stories in its first 30 days."""
    s = stories.sort_values("ts").copy()
    s["gap"] = s.groupby("rp_entity_id")["ts"].diff().dt.days
    ep = s[(s["gap"].isna() | (s["gap"] > GAP_DAYS)) & (s["ts"] >= pd.Timestamp(start))].copy()
    by_entity = {k: v["ts"].to_numpy() for k, v in s.groupby("rp_entity_id")}
    ep["stories_30d"] = [int(((t := by_entity[e]) >= ts.to_datetime64()).sum()
                             - (t >= (ts + pd.Timedelta(days=30)).to_datetime64()).sum())
                         for e, ts in zip(ep["rp_entity_id"], ep["ts"])]
    ny = ep["ts"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    after_close = ny.dt.hour * 60 + ny.dt.minute >= 16 * 60
    ep["event_date"] = ny.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")
    ep = ep.reset_index(drop=True)
    ep.insert(0, "episode_id", ["E" + str(i + 1).zfill(5) for i in range(len(ep))])
    return ep[["episode_id", "rp_entity_id", "entity_name", "ts", "event_date", "type", "headline", "stories_30d"]]


def attach_ceo_and_outcome(ep: pd.DataFrame, entity_gvkey: pd.DataFrame, anncomp: pd.DataFrame,
                           departures: pd.DataFrame) -> pd.DataFrame:
    """Add gvkey, the sitting CEO (co_per_rol, execid) and whether that CEO leaves within FOLLOW_DAYS."""
    out = ep.merge(entity_gvkey.drop_duplicates("rp_entity_id"), on="rp_entity_id", how="left")
    out["fyear"] = out["event_date"].dt.year
    ceos = anncomp[anncomp["ceoann"].eq("CEO")].assign(gvkey=lambda d: d["gvkey"].astype(str).str.zfill(6))
    ceos = ceos.drop_duplicates(["gvkey", "year"])[["gvkey", "year", "co_per_rol", "execid"]]
    cur = out.merge(ceos.rename(columns={"year": "fyear"}), on=["gvkey", "fyear"], how="left")
    prev = out.assign(fyear=out["fyear"] - 1).merge(ceos.rename(columns={"year": "fyear"}), on=["gvkey", "fyear"], how="left")
    for c in ["co_per_rol", "execid"]:
        out[c] = cur[c].fillna(prev[c]).to_numpy()
    dep = departures[["co_per_rol", "ref_date", "departure_code", "reason"]].dropna(subset=["co_per_rol"])
    m = out[["episode_id", "co_per_rol", "event_date"]].merge(dep, on="co_per_rol", how="inner")
    days = (m["ref_date"] - m["event_date"]).dt.days
    m = m[(days > -30) & (days <= FOLLOW_DAYS)].sort_values("ref_date").drop_duplicates("episode_id")
    out = out.merge(m[["episode_id", "ref_date", "departure_code", "reason"]].rename(
        columns={"ref_date": "ceo_exit_date", "reason": "exit_reason"}), on="episode_id", how="left")
    out["ceo_left_within_year"] = np.where(out["co_per_rol"].notna(), out["ceo_exit_date"].notna().astype(float), np.nan)
    return out
