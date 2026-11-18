"""Exploratory angles on the departure and scandal data (summary output only; nothing here is in the paper yet).

    ~/.venvs/key-person-risk/bin/python scripts/explore_angles.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "data/wrds"
pd.set_option("display.width", 220)


def t(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan


def show(title, df):
    print(f"\n=== {title}\n{df.round(4).to_string()}")


ev = pd.read_parquet(ROOT / "data/derived/event_sample.parquet")
ev["event_id"] = ev["dismissal_dataset_id"].astype(str)
feat = pd.read_parquet(ROOT / "data/derived/event_features.parquet").assign(event_id=lambda d: d["dismissal_dataset_id"].astype(str))
w = pd.read_parquet(ROOT / "data/derived/departure_windows.parquet")
w = w[w["complete"].astype(bool)]
car = w.pivot_table(index="event_id", columns="window", values="car_noalpha", aggfunc="first")
g = pd.read_csv(ROOT / "events/gentry_misconduct_coding.csv")
d = ev.set_index("event_id").join(car).join(feat.set_index("event_id")[["founder" if "founder" in feat else "duality", "sic2", "log_mv", "duality", "tenure_years"]].loc[:, lambda x: ~x.columns.duplicated()])
d = d.join(g.assign(event_id=g["dismissal_dataset_id"].astype(str)).set_index("event_id")["gentry_class"])
d["group"] = np.where(d["departure_code"] == 4, "misconduct: " + d["gentry_class"].fillna("?"), d["reason"])
news_dated = d["date_source"].isin(["ravenpack", "gentry_announced"]) | (d["departure_code"] == 4)
dd = d[news_dated & d["ann_m1_1"].notna()]

# 1. Personal versus corporate misconduct exits (alpha-free market model, news-dated except code 4)
rows = []
for k, x in dd.groupby("group"):
    rows.append({"group": k, "n": len(x), "pre_120_2": x["pre_120_2"].mean(), "pre_20_2": x["pre_20_2"].mean(),
                 "ann_m1_1": x["ann_m1_1"].mean(), "t_ann": t(x["ann_m1_1"]), "median_ann": x["ann_m1_1"].median(),
                 "post_2_60": x["post_2_60"].mean()})
show("1. CAR by departure group (alpha = 0)", pd.DataFrame(rows).set_index("group"))

# 2. Earnings on the same days
rdq = pd.read_parquet(W / "stage4/fundq_rdq.parquet").dropna(subset=["rdq"])
rdq["gvkey"] = rdq["gvkey"].astype(str).str.zfill(6)
rdq["rdq"] = pd.to_datetime(rdq["rdq"])
by = {k: v["rdq"].to_numpy() for k, v in rdq.groupby("gvkey")}
def near(gv, day, k=1):
    a = by.get(gv)
    return bool(a is not None and (np.abs((a - np.datetime64(day)) / np.timedelta64(1, "D")) <= k + 2).any())
dd = dd.assign(earnings=[near(a, b) for a, b in zip(dd["gvkey"], dd["event_date"])])
show("2. Share of exits within a few days of an earnings date, and CAR with/without",
     dd.groupby("group").apply(lambda x: pd.Series({"n": len(x), "share_with_earnings": x["earnings"].mean(),
                                                     "ann_with": x.loc[x["earnings"], "ann_m1_1"].mean(),
                                                     "ann_without": x.loc[~x["earnings"], "ann_m1_1"].mean(),
                                                     "t_without": t(x.loc[~x["earnings"], "ann_m1_1"])})))

# 3. Consumer-facing industries (SIC 2-digit: consumer goods, apparel, retail, restaurants, hotels, media, entertainment)
CONSUMER = {20, 21, 22, 23, 31, 52, 53, 54, 55, 56, 57, 58, 59, 70, 72, 78, 79, 48, 27}
pers = dd[dd["group"].isin(["misconduct: PERS", "misconduct: CORP"])].copy()
pers["consumer"] = pers["sic2"].isin(CONSUMER)
show("3. Misconduct exits: consumer-facing vs other industries",
     pers.groupby(["group", "consumer"]).agg(n=("ann_m1_1", "size"), ann=("ann_m1_1", "mean"), median=("ann_m1_1", "median"),
                                              post=("post_2_60", "mean")))

# 4. Successors: insider or outsider, and does the new CEO also chair?
a = pd.read_csv(W / "execucomp_anncomp.csv", dtype={"gvkey": str, "execid": str}, usecols=["gvkey", "year", "execid", "co_per_rol", "ceoann", "title"])
a["gvkey"] = a["gvkey"].str.zfill(6)
ceo = a[a["ceoann"].eq("CEO")]
rows = []
for r in d[d["group"].isin(["misconduct: PERS", "misconduct: CORP", "dismissed_performance", "retired"])].itertuples():
    nxt = ceo[(ceo["gvkey"] == r.gvkey) & (ceo["year"].isin([r.fyear, r.fyear + 1])) & (ceo["co_per_rol"] != r.co_per_rol)]
    if nxt.empty:
        continue
    s = nxt.sort_values("year").iloc[-1]
    hist = a[(a["gvkey"] == r.gvkey) & (a["execid"] == s["execid"]) & (a["year"] < r.fyear)]
    rows.append({"group": r.group, "insider": len(hist) > 0,
                 "chairs": bool(pd.Series([s["title"]]).fillna("").str.contains("chair|chmn", case=False).iloc[0]),
                 "old_ceo_chaired": r.duality})
succ = pd.DataFrame(rows)
show("4. Successors", succ.groupby("group").agg(n=("insider", "size"), insider=("insider", "mean"), new_ceo_chairs=("chairs", "mean"),
                                                old_ceo_chaired=("old_ceo_chaired", "mean")))

# 5. Channels after firm-level misconduct episodes: what follows in the next 180 days, CEO stays vs leaves
ep = pd.read_parquet(ROOT / "data/derived/misconduct_episodes.parquet")
ep = ep[ep["co_per_rol"].notna()].copy()
cols = ["rp_entity_id", "timestamp_utc", "type", "position_name", "rp_story_id"]
ch = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted((W / "stage3/ravenpack").glob("dj_channels_*.parquet"))])
ch["ts"] = pd.to_datetime(ch["timestamp_utc"])
ch = ch.drop_duplicates(["rp_entity_id", "rp_story_id", "type"])
ceo_pos = ch["position_name"].fillna("").str.contains("Chief Executive", case=False)
kinds = {"ceo_insider_buy": (ch["type"] == "insider-buy") & ceo_pos, "ceo_insider_sell": (ch["type"] == "insider-sell") & ceo_pos,
         "buyback": ch["type"].eq("buybacks"), "board_appointment": ch["type"].eq("board-member-appointment"),
         "rights_plan": ch["type"].eq("shareholder-rights-plan"), "settlement": ch["type"].eq("settlement")}
idx = {k: {e: v["ts"].to_numpy() for e, v in ch[m].groupby("rp_entity_id")} for k, m in kinds.items()}
for k in kinds:
    ep[k] = [int(((x := idx[k].get(e, np.array([], dtype="datetime64[ns]"))) >= np.datetime64(s)).sum()
                 - (x >= np.datetime64(s + pd.Timedelta(days=180))).sum()) > 0
             for e, s in zip(ep["rp_entity_id"], pd.to_datetime(ep["ts"]))]
    ep[k + "_before"] = [int(((x := idx[k].get(e, np.array([], dtype="datetime64[ns]"))) >= np.datetime64(s - pd.Timedelta(days=180))).sum()
                 - (x >= np.datetime64(s)).sum()) > 0
             for e, s in zip(ep["rp_entity_id"], pd.to_datetime(ep["ts"]))]
ep["left"] = ep["ceo_left_within_year"].map({1.0: "CEO left within a year", 0.0: "CEO stayed"})
show("5. Share of misconduct episodes with each channel in the 180 days after (and before) the first story",
     ep.groupby("left")[[c for k in kinds for c in (k, k + "_before")]].mean().T)
print("episodes:", ep["left"].value_counts().to_dict())
