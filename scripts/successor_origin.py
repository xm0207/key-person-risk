"""Which successors do investors welcome? Abnormal returns at the announcement of a chief executive's departure,
by who succeeded, for all departures in the Gentry et al. database, 2000-2024.

Successor: the first executive at the same company whose ExecuComp record shows ceoann == "CEO" and who became
chief executive between 60 days before and two years after the departure announcement.
    - named at the departure: became chief executive within 30 days of the announcement;
    - named later: became chief executive more than 30 days after it (an interim chief executive in between, or
      a search under way).
    - outsider: joined the company at most one year before becoming chief executive (the definition of Huson,
      Malatesta and Parrino, 2004); insider: joined earlier, or appears in ExecuComp at the company in an
      earlier year. Successors with neither date nor earlier record are left unclassified.
    - former chief executive elsewhere: ExecuComp lists the successor as chief executive of another company
      in an earlier year.
    - chief financial officer: the successor's ExecuComp title at the company in the year before becoming chief
      executive mentions the CFO or chief financial officer.
Abnormal returns are those of run_departure_study.py (market model, constant set to zero), days [-1, +1] around
the departure announcement and [+2, +60] after it. Forced: Gentry codes 3 and 4 (dismissed for performance, or
for legal violations or concerns). Writes summary statistics only, to results/successor_origin.csv.

    ~/.venvs/key-person-risk/bin/python scripts/successor_origin.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def successors(events: pd.DataFrame, ex: pd.DataFrame) -> pd.DataFrame:
    ceo = ex[ex["ceoann"] == "CEO"].dropna(subset=["becameceo"]).drop_duplicates(["gvkey", "execid", "becameceo"])
    first_year = ex.groupby(["gvkey", "execid"])["year"].min()
    ceo_elsewhere = ex[ex["ceoann"] == "CEO"].groupby("execid")[["gvkey", "year"]].apply(lambda d: list(zip(d.gvkey, d.year)))
    titles = ex.set_index(["gvkey", "execid", "year"])["title"].str.lower()
    rows = []
    for r in events.itertuples():
        c = ceo[(ceo["gvkey"] == r.gvkey) & (ceo["execid"] != r.execid)
                & (ceo["becameceo"] >= r.event_date - pd.Timedelta(days=60))
                & (ceo["becameceo"] <= r.event_date + pd.Timedelta(days=730))].sort_values("becameceo")
        if c.empty:
            continue
        s = c.iloc[0]
        start_year = s.becameceo.year
        earlier = first_year.get((r.gvkey, s.execid), 9999) < start_year
        if pd.notna(s.joined_co):
            origin = "outsider" if s.joined_co >= s.becameceo - pd.Timedelta(days=365) else "insider"
        else:
            origin = "insider" if earlier else "unclassified"
        if origin == "outsider" and earlier:
            origin = "insider"
        prev = titles.get((r.gvkey, s.execid, start_year - 1), "")
        rows.append({"event_id": str(r.dismissal_dataset_id), "successor_gap_days": (s.becameceo - r.event_date).days,
                     "origin": origin,
                     "ceo_elsewhere": any(g != r.gvkey and y < start_year for g, y in ceo_elsewhere.get(s.execid, [])),
                     "was_cfo": isinstance(prev, str) and ("cfo" in prev or "chief financial" in prev),
                     "interim_title": "interim" in str(s.title).lower() or "acting" in str(s.title).lower()})
    return pd.DataFrame(rows)


def main() -> None:
    ev = pd.read_parquet(ROOT / "data/derived/event_features.parquet")
    ev = ev.dropna(subset=["execid"]).assign(event_date=lambda d: pd.to_datetime(d["event_date"]))
    ex = pd.read_csv(ROOT / "data/wrds/execucomp_anncomp.csv", dtype={"gvkey": str, "execid": str},
                     parse_dates=["becameceo", "joined_co"], usecols=["gvkey", "execid", "year", "becameceo",
                                                                      "joined_co", "title", "ceoann"])
    s = successors(ev, ex)
    w = pd.read_parquet(ROOT / "data/derived/departure_windows.parquet")
    w["car0"] = w["car"] + w["alpha_part"]
    cars = w.pivot_table(index="event_id", columns="window", values="car0")[["ann_m1_1", "post_2_60"]]
    d = (ev.assign(event_id=ev["dismissal_dataset_id"].astype(str))[["event_id", "departure_code", "event_date"]]
         .merge(s, on="event_id").merge(cars, left_on="event_id", right_index=True))
    d = d[~d["interim_title"]]
    d["forced"] = d["departure_code"].isin([3, 4])
    d["timing"] = np.where(d["successor_gap_days"] <= 30, "named at the departure", "named later")
    d.to_parquet(ROOT / "data/derived/successor_origin.parquet", index=False)

    def summ(g: pd.DataFrame, label: str) -> dict:
        x, y = g["ann_m1_1"].dropna(), g["post_2_60"].dropna()
        return {"group": label, "cases": len(x), "mean_m1_1": x.mean(), "se_m1_1": x.std() / np.sqrt(len(x)),
                "median_m1_1": x.median(), "share_positive": (x > 0).mean(), "mean_p2_60": y.mean(),
                "se_p2_60": y.std() / np.sqrt(len(y))}

    out = []
    for forced, fl in [(True, "forced"), (False, "other departures")]:
        f = d[d["forced"] == forced]
        out.append(summ(f, f"{fl}: all with an identified successor"))
        for t in ["named at the departure", "named later"]:
            out.append(summ(f[f["timing"] == t], f"{fl}: {t}, any origin"))
            for o in ["outsider", "insider"]:
                out.append(summ(f[(f["timing"] == t) & (f["origin"] == o)], f"{fl}: {t}, {o}"))
        n = f[(f["timing"] == "named at the departure")]
        o = n[n["origin"] == "outsider"]
        out.append(summ(o[o["ceo_elsewhere"]], f"{fl}: named at the departure, outsider, former CEO elsewhere"))
        out.append(summ(o[~o["ceo_elsewhere"]], f"{fl}: named at the departure, outsider, no CEO record elsewhere"))
        out.append(summ(n[n["was_cfo"]], f"{fl}: named at the departure, chief financial officer"))
    out = pd.DataFrame(out).assign(rank_sum_p=np.nan)
    f = d[d["forced"] & (d["timing"] == "named at the departure")]
    a, b = f.loc[f["origin"] == "outsider", "ann_m1_1"].dropna(), f.loc[f["origin"] == "insider", "ann_m1_1"].dropna()
    # Difference in means with its standard error, difference in medians, and the Mann-Whitney rank-sum p-value
    out.loc[len(out)] = {"group": "forced, named at the departure: outsider minus insider", "cases": len(a) + len(b),
                         "mean_m1_1": a.mean() - b.mean(), "se_m1_1": np.sqrt(a.var() / len(a) + b.var() / len(b)),
                         "median_m1_1": a.median() - b.median(),
                         "rank_sum_p": stats.mannwhitneyu(a, b, alternative="two-sided").pvalue}
    out.to_csv(ROOT / "results/successor_origin.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string(index=False))
    print(d.groupby(["forced", "timing", "origin"]).size())


if __name__ == "__main__":
    main()
