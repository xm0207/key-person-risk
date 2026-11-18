"""Build the CEO-departure event sample: Gentry et al. departure reasons, linked to CRSP and RavenPack.

The event list is the open CEO dismissal database of Gentry, Harrison, Quigley and Boivie (2021, SMJ
42(5):968-991; version 10 Oct 2024 on Zenodo, record 17725939). It codes the reason for every CEO
turnover in ExecuComp: 1 death, 2 illness, 3 dismissed for performance, 4 dismissed for legal or
behavioural problems, 5 retired, 6 new opportunity, 7 other, 8 missing, 9 ExecuComp error.

Each departure is linked to
  * a CRSP PERMNO through the CCM link table (link types LC/LU, primary links P/C, valid on the date);
  * a RavenPack entity through the company's CUSIP;
  * the first RavenPack story about that CEO's departure, which dates the news to the minute.

All functions are pure (frames in, frames out) so they can be tested offline.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

REASONS = {
    1: "death", 2: "illness", 3: "dismissed_performance", 4: "dismissed_misconduct",
    5: "retired", 6: "new_opportunity",
}
INVOLUNTARY = {1, 2, 3, 4}

# RavenPack event types that report a CEO leaving. Appointments are left out: for a short-tenure CEO
# the story announcing their own appointment would otherwise be taken as the departure news.
DEPARTURE_TYPES = {
    "executive-resignation", "executive-firing", "executive-retirement", "executive-health",
    "executive-death", "executive-search", "executive-scandal",
}
SEARCH_BEFORE_DAYS = 400  # retirements are often announced many months before the CEO leaves
SEARCH_AFTER_DAYS = 7

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "cpa", "esq"}


def surname(full_name: str) -> str:
    """Last name from an ExecuComp full name, without suffixes: 'John A. Smith Jr.' -> 'Smith'."""
    tokens = [t for t in re.split(r"[\s,]+", str(full_name)) if t]
    while tokens and tokens[-1].lower().replace(".", "") in _SUFFIXES:
        tokens.pop()
    return tokens[-1].strip(".") if tokens else ""


def load_gentry(frame: pd.DataFrame, start: str = "2000-01-01") -> pd.DataFrame:
    """Clean the Gentry et al. file: codes 1-6 only, dated from `start`, one row per departure."""
    g = frame.copy()
    g["gvkey"] = g["gvkey"].astype("Int64").astype(str).str.zfill(6)
    g["left_ceo_office"] = pd.to_datetime(g["left_ceo_office"], errors="coerce")
    g["date_departure_announced"] = pd.to_datetime(g["date_departure_announced"], errors="coerce")
    g = g[g["departure_code"].isin(list(REASONS))].copy()
    g["departure_code"] = g["departure_code"].astype(int)
    g["reason"] = g["departure_code"].map(REASONS)
    g["involuntary"] = g["departure_code"].isin(INVOLUNTARY)
    # Reference date: the first announcement when Gentry et al. record it, else the date the CEO left.
    g["ref_date"] = g["date_departure_announced"].fillna(g["left_ceo_office"])
    g = g[g["ref_date"] >= pd.Timestamp(start)].dropna(subset=["ref_date"])
    g["surname"] = g["exec_fullname"].map(surname)
    keep = ["dismissal_dataset_id", "gvkey", "coname", "fyear", "co_per_rol", "exec_fullname", "surname",
            "departure_code", "reason", "involuntary", "interim_coceo", "left_ceo_office",
            "date_departure_announced", "ref_date"]
    return g[keep].reset_index(drop=True)


def link_permno(events: pd.DataFrame, ccm: pd.DataFrame) -> pd.DataFrame:
    """Add `permno` from the CCM link valid on `ref_date` (LC/LU links; primary P before C)."""
    links = ccm.rename(columns=str.lower)
    links = links[links["linktype"].isin(["LC", "LU"]) & links["linkprim"].isin(["P", "C"])].copy()
    links["gvkey"] = links["gvkey"].astype(str).str.zfill(6)
    links["linkdt"] = pd.to_datetime(links["linkdt"], errors="coerce")
    links["linkenddt"] = pd.to_datetime(links["linkenddt"].replace("E", None), errors="coerce").fillna(
        pd.Timestamp("2099-12-31"))
    links["prim_rank"] = links["linkprim"].map({"P": 0, "C": 1})
    m = events.reset_index().merge(links[["gvkey", "lpermno", "linkdt", "linkenddt", "prim_rank"]],
                                   on="gvkey", how="left")
    valid = (m["ref_date"] >= m["linkdt"]) & (m["ref_date"] <= m["linkenddt"])
    m = m[valid].sort_values(["index", "prim_rank"]).drop_duplicates("index")
    out = events.copy()
    out["permno"] = pd.Series(m.set_index("index")["lpermno"]).reindex(out.index).astype("Int64")
    return out


def link_rp_entity(events: pd.DataFrame, funda: pd.DataFrame, rp_hist: pd.DataFrame) -> pd.DataFrame:
    """Add `rp_entity_id` by matching the Compustat CUSIP to RavenPack's CUSIP history.

    The 8-character CUSIP (issuer plus issue) is tried first, then the 6-character issuer code. When a
    CUSIP maps to more than one entity, the one whose mapping range covers `ref_date` wins.
    """
    cus = (funda.dropna(subset=["cusip"]).assign(gvkey=lambda d: d["gvkey"].astype(str).str.zfill(6))
           .sort_values("fyear").drop_duplicates("gvkey", keep="last")[["gvkey", "cusip"]])
    rp = rp_hist.dropna(subset=["cusip"]).copy()
    rp["range_start"] = pd.to_datetime(rp["range_start"], errors="coerce")
    rp["range_end"] = pd.to_datetime(rp["range_end"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    base = events.reset_index().merge(cus, on="gvkey", how="left")
    found = {}
    for width, label in [(8, "cusip8"), (6, "cusip6")]:
        left = base[~base["index"].isin(found)].assign(key=lambda d: d["cusip"].str[:width])
        right = rp.assign(key=rp["cusip"].str[:width])[["key", "rp_entity_id", "range_start", "range_end"]]
        m = left.merge(right, on="key", how="inner")
        if m.empty:
            continue
        m["covers"] = (m["ref_date"] >= m["range_start"]) & (m["ref_date"] <= m["range_end"])
        m = m.sort_values(["index", "covers"], ascending=[True, False]).drop_duplicates("index")
        for idx, ent in zip(m["index"], m["rp_entity_id"]):
            found[idx] = (ent, label)
    out = events.copy()
    out["rp_entity_id"] = [found.get(i, (None, None))[0] for i in out.index]
    out["rp_link"] = [found.get(i, (None, None))[1] for i in out.index]
    return out


def to_new_york(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.tz_convert("America/New_York")


def first_departure_news(events: pd.DataFrame, rp: pd.DataFrame) -> pd.DataFrame:
    """Date each departure by the first RavenPack story about it.

    Candidate stories: same RavenPack entity, a departure-type event, a CEO position, published between
    SEARCH_BEFORE_DAYS before and SEARCH_AFTER_DAYS after `left_ceo_office` (or `ref_date`). A story counts
    as naming the CEO when the surname appears in the headline or event text. The first naming story is
    preferred; the first unnamed CEO story is kept separately as a weaker date.
    """
    stories = rp[rp["type"].isin(DEPARTURE_TYPES)
                 & rp["position_name"].fillna("").str.contains("Chief Executive", case=False)].copy()
    stories["ts_ny"] = to_new_york(stories["timestamp_utc"])
    text = (stories["headline"].fillna("") + " " + stories["event_text"].fillna("")).str.lower()
    stories["text"] = text
    by_entity = {k: v for k, v in stories.groupby("rp_entity_id")}

    rows = []
    for idx, ev in events.iterrows():
        rec = {"index": idx, "rp_named_ts": pd.NaT, "rp_named_type": None, "rp_named_headline": None,
               "rp_any_ts": pd.NaT, "rp_n_stories": 0}
        s = by_entity.get(ev.get("rp_entity_id"))
        anchor = ev["left_ceo_office"] if pd.notna(ev["left_ceo_office"]) else ev["ref_date"]
        if s is not None and pd.notna(anchor):
            lo = pd.Timestamp(anchor - pd.Timedelta(days=SEARCH_BEFORE_DAYS)).tz_localize("America/New_York")
            hi = pd.Timestamp(anchor + pd.Timedelta(days=SEARCH_AFTER_DAYS + 1)).tz_localize("America/New_York")
            w = s[(s["ts_ny"] >= lo) & (s["ts_ny"] < hi)].sort_values("ts_ny")
            rec["rp_n_stories"] = len(w)
            if len(w):
                rec["rp_any_ts"] = w["ts_ny"].iloc[0]
                name = str(ev["surname"]).lower()
                named = w[w["text"].str.contains(re.escape(name), regex=True)] if len(name) >= 3 else w.iloc[0:0]
                if len(named):
                    first = named.iloc[0]
                    rec.update(rp_named_ts=first["ts_ny"], rp_named_type=first["type"],
                               rp_named_headline=first["headline"])
        rows.append(rec)
    found = pd.DataFrame(rows).set_index("index")
    return events.join(found)


def effective_date(ts_ny: pd.Series) -> pd.Series:
    """Calendar date on which the market can first react: news after 16:00 New York time counts for the
    next day (rolled forward to a trading day later, against the CRSP calendar)."""
    ts = pd.to_datetime(ts_ny)
    day = ts.dt.tz_localize(None).dt.normalize() if ts.dt.tz is not None else ts.dt.normalize()
    after_close = (ts.dt.hour * 60 + ts.dt.minute) >= 16 * 60
    return day + pd.to_timedelta(after_close.astype(int), unit="D")


def choose_event_date(events: pd.DataFrame) -> pd.DataFrame:
    """Event date: the RavenPack story naming the CEO when there is one, else Gentry's announcement date,
    else the date the CEO left office. `date_source` records which, and `gentry_gap_days` compares the
    RavenPack date with Gentry's announcement date where both exist."""
    out = events.copy()
    rp_day = effective_date(out["rp_named_ts"]) if out["rp_named_ts"].notna().any() else pd.Series(pd.NaT, index=out.index)
    out["event_date"] = rp_day
    out["date_source"] = np.where(rp_day.notna(), "ravenpack", None)
    fill = out["event_date"].isna() & out["date_departure_announced"].notna()
    out.loc[fill, "event_date"] = out.loc[fill, "date_departure_announced"]
    out.loc[fill, "date_source"] = "gentry_announced"
    fill = out["event_date"].isna()
    out.loc[fill, "event_date"] = out.loc[fill, "left_ceo_office"]
    out.loc[fill, "date_source"] = "left_office"
    out["gentry_gap_days"] = (rp_day - out["date_departure_announced"]).dt.days
    return out
