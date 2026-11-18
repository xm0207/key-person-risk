"""CEO power, firm power, executive network and career outcomes for each departure.

CEO power (measured in the CEO's last full fiscal year in office, the ExecuComp year before the departure
year when available):
  * tenure_years  years since becoming CEO (ExecuComp `becameceo`; else the count of ExecuComp years as CEO,
                  which is left-censored at 1992)
  * age, own_pct  shares owned excluding options, % of shares outstanding
  * duality       the CEO's title includes chair (chairman, chmn, chair)
  * pay_slice     CEO total pay (tdc1) over the total pay of the firm's five best-paid executives that year
                  (Bebchuk, Cremers and Peyer 2011)
Firm power (Compustat, same fiscal year): log market value, log assets, and the firm's share of sales in its
2-digit SIC industry that year.
Network and career (ExecuComp mobility; BoardEx is not available):
  * prior_firms      other ExecuComp firms at which the executive appears before the departure year
  * later_exec_5y    appears as an executive at another ExecuComp firm within five fiscal years afterwards
  * later_ceo_5y     ... as that firm's CEO
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

CHAIR = re.compile(r"chair|chmn", re.IGNORECASE)
FOLLOW_UP_YEARS = 5


def _prep_anncomp(anncomp: pd.DataFrame) -> pd.DataFrame:
    a = anncomp.copy()
    a["gvkey"] = a["gvkey"].astype(str).str.zfill(6)
    a["execid"] = a["execid"].astype(str)
    a["is_ceo"] = a["ceoann"].eq("CEO")
    a["becameceo"] = pd.to_datetime(a["becameceo"], errors="coerce")
    top5 = a.sort_values("tdc1", ascending=False).groupby(["gvkey", "year"]).head(5)
    a = a.merge(top5.groupby(["gvkey", "year"])["tdc1"].sum().rename("top5_tdc1").reset_index(),
                on=["gvkey", "year"], how="left")
    return a


def ceo_power(events: pd.DataFrame, anncomp: pd.DataFrame) -> pd.DataFrame:
    a = _prep_anncomp(anncomp)
    own = a.drop_duplicates(["co_per_rol", "year"]).set_index(["co_per_rol", "year"])
    ceo_years = a[a["is_ceo"]].groupby("co_per_rol")["year"].agg(list)
    rows = []
    for idx, ev in events.iterrows():
        rec = {"index": idx, "execid": None, "power_year": np.nan}
        for yr in (ev["fyear"] - 1, ev["fyear"]):
            if (ev["co_per_rol"], yr) in own.index:
                r = own.loc[(ev["co_per_rol"], yr)]
                rec.update(execid=r["execid"], power_year=yr, age=r["age"], own_pct=r["shrown_excl_opts_pct"],
                           duality=bool(CHAIR.search(str(r["title"]))) if pd.notna(r["title"]) else np.nan,
                           pay_slice=r["tdc1"] / r["top5_tdc1"] if r["top5_tdc1"] and r["top5_tdc1"] > 0 else np.nan)
                if pd.notna(r["becameceo"]):
                    rec["tenure_years"] = (ev["ref_date"] - r["becameceo"]).days / 365.25
                else:
                    yrs = [y for y in ceo_years.get(ev["co_per_rol"], []) if y <= ev["fyear"]]
                    rec["tenure_years"] = float(len(yrs)) if yrs else np.nan
                break
        rows.append(rec)
    return events.join(pd.DataFrame(rows).set_index("index"))


def firm_power(events: pd.DataFrame, funda: pd.DataFrame) -> pd.DataFrame:
    f = funda.copy()
    f["gvkey"] = f["gvkey"].astype(str).str.zfill(6)
    f = f.sort_values(["gvkey", "fyear"]).drop_duplicates(["gvkey", "fyear"], keep="last")
    f["mv"] = f["csho"] * f["prcc_f"]
    f["sic2"] = (f["sich"] // 100).astype("Int64")
    ind = f.dropna(subset=["sic2", "sale"]).groupby(["fyear", "sic2"])["sale"].transform("sum")
    f.loc[ind.index, "sales_share"] = f.loc[ind.index, "sale"] / ind.where(ind > 0)
    f["log_mv"] = np.log(f["mv"].where(f["mv"] > 0))
    f["log_at"] = np.log(f["at"].where(f["at"] > 0))
    cols = ["log_mv", "log_at", "sales_share", "sic2"]
    out = events.copy()
    out["firm_year"] = out["fyear"] - 1
    m = out.reset_index().merge(f[["gvkey", "fyear"] + cols].rename(columns={"fyear": "firm_year"}),
                                on=["gvkey", "firm_year"], how="left").set_index("index")
    for c in cols:
        out[c] = m[c]
    return out


def network_and_career(events: pd.DataFrame, anncomp: pd.DataFrame) -> pd.DataFrame:
    a = _prep_anncomp(anncomp)[["execid", "gvkey", "year", "is_ceo"]]
    by_exec = {k: v for k, v in a.groupby("execid")}
    rows = []
    for idx, ev in events.iterrows():
        rec = {"index": idx, "prior_firms": np.nan, "later_exec_5y": np.nan, "later_ceo_5y": np.nan}
        h = by_exec.get(ev.get("execid"))
        if h is not None:
            other = h[h["gvkey"] != ev["gvkey"]]
            rec["prior_firms"] = other.loc[other["year"] < ev["fyear"], "gvkey"].nunique()
            later = other[(other["year"] > ev["fyear"]) & (other["year"] <= ev["fyear"] + FOLLOW_UP_YEARS)]
            rec["later_exec_5y"] = float(len(later) > 0)
            rec["later_ceo_5y"] = float(later["is_ceo"].any())
        rows.append(rec)
    return events.join(pd.DataFrame(rows).set_index("index"))


def news_before(events: pd.DataFrame, rp: pd.DataFrame, days: int = 365) -> pd.DataFrame:
    """RavenPack CEO stories about the firm before the event date: how many, and whether any scandal story
    named the CEO (a sign that trouble was public before the departure)."""
    s = rp[rp["position_name"].fillna("").str.contains("Chief Executive", case=False)].copy()
    s["day"] = pd.to_datetime(s["timestamp_utc"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    s["text"] = (s["headline"].fillna("") + " " + s["event_text"].fillna("")).str.lower()
    by_entity = {k: v for k, v in s.groupby("rp_entity_id")}
    rows = []
    for idx, ev in events.iterrows():
        rec = {"index": idx, "pre_ceo_stories": np.nan, "pre_scandal_named": np.nan}
        h = by_entity.get(ev.get("rp_entity_id"))
        if pd.notna(ev.get("rp_entity_id")):
            rec.update(pre_ceo_stories=0, pre_scandal_named=0.0)
        if h is not None:
            w = h[(h["day"] < ev["event_date"]) & (h["day"] >= ev["event_date"] - pd.Timedelta(days=days))]
            rec["pre_ceo_stories"] = w["rp_story_id"].nunique()
            name = str(ev["surname"]).lower()
            sc = w[(w["type"] == "executive-scandal") & w["text"].str.contains(re.escape(name))] if len(name) >= 3 else w.iloc[0:0]
            rec["pre_scandal_named"] = float(len(sc) > 0)
        rows.append(rec)
    return events.join(pd.DataFrame(rows).set_index("index"))
