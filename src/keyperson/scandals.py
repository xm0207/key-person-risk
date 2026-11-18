"""Candidate CEO personal-scandal episodes from RavenPack headlines, before hand coding.

A candidate story is a Dow Jones story about a US company (stage-4 search: the headline names a CEO, chief
executive, chairman or founder and contains a scandal word) whose headline also contains the surname of the
firm's sitting CEO (ExecuComp CEO of that fiscal year or the year before). Stories about a former CEO are
dropped, as are surnames shorter than four letters and surnames that are part of the company's name.

An episode is the first candidate story for a CEO at a firm after at least `GAP_DAYS` without one. Each
episode carries its first three headlines (first 30 days) for coding, and a keyword class:
  * personal: words that point at the CEO's own conduct (relationship, harassment, arrest, apology, ...)
  * legal: words that fit firm-level legal trouble as well (probe, lawsuit, fraud, subpoena, ...)
The class only orders the coding; the code itself is assigned by reading the headlines (see codebook).
"""
from __future__ import annotations

import re

import pandas as pd

from keyperson.sample import surname

GAP_DAYS = 365
FORMER = re.compile(r"\b(?:ex-|former|onetime|one-time|late)\b", re.IGNORECASE)
PERSONAL = re.compile(
    r"affair|relationship|sexual|harass|misconduct|inappropriate|conduct|arrest|dui\b|indict|charged|guilty|plea"
    r"|insider trading|lied|misled|expense|apolog|leave of absence|tweet|ethic|personal|allegation|alleged|accus"
    r"|controvers|outrage|backlash|scandal", re.IGNORECASE)


def sitting_ceos(anncomp: pd.DataFrame) -> pd.DataFrame:
    """gvkey, year, execid, co_per_rol, exec_fullname, surname: the CEO of each fiscal year, also listed under
    the next year so that a story early in a calendar year can match the CEO of the fiscal year before."""
    a = anncomp[anncomp["ceoann"].eq("CEO")].copy()
    a["gvkey"] = a["gvkey"].astype(str).str.zfill(6)
    a["surname"] = a["exec_fullname"].map(surname)
    a = a[["gvkey", "year", "execid", "co_per_rol", "exec_fullname", "surname"]]
    return pd.concat([a, a.assign(year=a["year"] + 1)]).drop_duplicates(["gvkey", "year", "execid"])


def candidate_stories(headlines: pd.DataFrame, entity_gvkey: pd.DataFrame, ceos: pd.DataFrame) -> pd.DataFrame:
    h = headlines.drop_duplicates(["rp_entity_id", "rp_story_id"]).copy()
    h["ts"] = pd.to_datetime(h["timestamp_utc"])
    h["year"] = h["ts"].dt.year
    h = h[~h["headline"].fillna("").str.contains(FORMER)]
    # An entity can map to more than one gvkey (Activision Inc. and Activision Blizzard); keep each, the CEO-year
    # match below picks the right one.
    h = h.merge(entity_gvkey.drop_duplicates(["rp_entity_id", "gvkey"]), on="rp_entity_id", how="inner")
    m = h.merge(ceos, on=["gvkey", "year"], how="inner")
    m = m[m["surname"].str.len() >= 4]
    m = m[pd.Series([s.lower() not in str(e).lower() for s, e in zip(m["surname"], m["entity_name"])],
                    index=m.index, dtype=bool)]
    named = pd.Series([bool(re.search(r"\b" + re.escape(s) + r"\b", str(t), re.IGNORECASE))
                       for s, t in zip(m["surname"], m["headline"])], index=m.index, dtype=bool)
    return m[named].sort_values("ts").drop_duplicates(["rp_story_id", "execid"])


def build_episodes(stories: pd.DataFrame) -> pd.DataFrame:
    s = stories.sort_values("ts").copy()
    s["gap"] = s.groupby(["gvkey", "execid"])["ts"].diff().dt.days
    s["episode_no"] = (s["gap"].isna() | (s["gap"] > GAP_DAYS)).groupby([s["gvkey"], s["execid"]]).cumsum()
    rows = []
    for (gvkey, execid, k), g in s.groupby(["gvkey", "execid", "episode_no"], sort=False):
        first = g.iloc[0]
        window = g[g["ts"] < first["ts"] + pd.Timedelta(days=30)]
        heads = list(dict.fromkeys(window["headline"].astype(str)))[:3]
        rows.append({
            "gvkey": gvkey, "execid": execid, "co_per_rol": first["co_per_rol"], "exec_fullname": first["exec_fullname"],
            "rp_entity_id": first["rp_entity_id"], "entity_name": first["entity_name"], "first_ts": first["ts"],
            "first_story_id": first["rp_story_id"], "stories_30d": len(window), "stories_episode": len(g),
            "last_ts": g["ts"].iloc[-1], "headlines": " || ".join(heads),
            "keyword_class": "personal" if any(PERSONAL.search(x) for x in heads) else "legal",
        })
    ep = pd.DataFrame(rows).sort_values("first_ts").reset_index(drop=True)
    ep.insert(0, "candidate_id", ["C" + str(i + 1).zfill(4) for i in range(len(ep))])
    return ep


# Whole-word scandal terms (no "Semiconductor" -> "conduct", no "Pleased" -> "plea"), used to date allegations.
STRICT = re.compile(
    r"\b(relationship with|personal relationship|romantic|affair|sexual|harass\w*|misconduct|inappropriate|"
    r"personal conduct|improper conduct|conduct|arrest\w*|dui|indict\w*|charged|pleads?|guilty|insider trading|"
    r"lied|misled|apologi\w*|allegation\w*|alleged\w*|accus\w*|scandal|blackmail|probe|investigation)\b",
    re.IGNORECASE)


def stable_ids(ep: pd.DataFrame, previous: pd.DataFrame | None, prefix: str = "C") -> pd.DataFrame:
    """Give episodes already in `previous` (same firm, CEO and first story) their old IDs and number new episodes
    after the last old one, so that hand codes keyed on the ID stay valid when the search is re-run."""
    if previous is None or previous.empty:
        return ep
    key = ["gvkey", "execid", "first_story_id"]
    old = previous.assign(gvkey=previous["gvkey"].astype(str).str.zfill(6), execid=previous["execid"].astype(str))
    old_id = dict(zip(map(tuple, old[key].astype(str).to_numpy()), old["candidate_id"]))
    # Fallback for two stories with the same time stamp, where either can come first: the CEO and the time.
    when = lambda d: pd.to_datetime(d["first_ts"]).astype("int64").astype(str)
    old_time = dict(zip(zip(old["execid"], when(old)), old["candidate_id"]))
    ids = [old_id.get(k, old_time.get((e, t))) for k, e, t in
           zip(map(tuple, ep[key].astype(str).to_numpy()), ep["execid"].astype(str), when(ep))]
    nxt = max(int(str(i)[1:]) for i in old["candidate_id"]) + 1
    for j, i in enumerate(ids):
        if i is None:
            ids[j], nxt = prefix + str(nxt).zfill(4), nxt + 1
    return ep.assign(candidate_id=ids)


def strict_stories(stories: pd.DataFrame) -> pd.DataFrame:
    text = stories["headline"].fillna("").str.replace("semiconductor", "", case=False, regex=False)
    return stories[text.map(lambda x: bool(STRICT.search(x)))]


def classify_disclosure(first_news: pd.Timestamp, exit_date, min_gap_days: int = 2) -> str:
    """'two-stage' when the allegation or probe was public at least `min_gap_days` before the exit,
    'one-stage' when it became public with the exit, 'survived' when there was no exit within a year."""
    if pd.isna(exit_date):
        return "survived"
    return "two-stage" if (pd.Timestamp(exit_date) - pd.Timestamp(first_news)).days >= min_gap_days else "one-stage"


# Narrower list for dating the first public allegation before a personal-conduct exit: words that describe the
# CEO's own conduct, not business relationships, medical leave or firm-level probes.
DATING = re.compile(
    r"\b(personal relationship|inappropriate relationship|romantic|affair|sexual\w*|harass\w*|misconduct|inappropriate|"
    r"personal conduct|improper conduct|conduct|arrest\w*|dui|indict\w*|charged|pleads?|guilty|insider trading|lied|"
    r"misled|allegation\w*|alleged\w*|accus\w*|scandal|blackmail)\b", re.IGNORECASE)


def dating_stories(stories: pd.DataFrame) -> pd.DataFrame:
    text = stories["headline"].fillna("").str.replace("semiconductor", "", case=False, regex=False)
    return stories[text.map(lambda x: bool(DATING.search(x)))]
