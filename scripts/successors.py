"""After a chief executive leaves over personal conduct and an interim chief executive takes over: how does
the share price respond when the permanent successor is named?

Cases: the departures in data/derived/personal_scandals.parquet after which the company named an interim or
acting chief executive (found from Dow Jones headlines within three days of the departure that mention an
interim or acting chief executive). For each, events/successor_announcements.csv gives the permanent
successor, the announcement date (the first trading day on which the news could affect the close) and
whether the successor was hired from outside, promoted from inside, or the interim chief executive made
permanent, with the Dow Jones headline as evidence. Comtech (2024) is left out because its successor was
named in 2025, after the CRSP data end.

Abnormal returns use the market model of run_personal_scandals.py (constant set to zero) around the
announcement. Writes results/successor_summary.csv and data/derived/successors.parquet.

    ~/.venvs/key-person-risk/bin/python scripts/successors.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from keyperson import crsp
from keyperson.engine import EventStudyEngine, Window

import run_personal_scandals as rps  # same folder

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "data/wrds"
WINDOWS = (Window("m1_1", -1, 1), Window("d0", 0, 0), Window("p2_20", 2, 20))


def main() -> None:
    s = pd.read_csv(ROOT / "events/successor_announcements.csv", parse_dates=["announced"])
    cases = pd.read_parquet(ROOT / "data/derived/personal_scandals.parquet")[["case", "permno", "exit_date", "disclosure"]]
    s = s.merge(cases, on="case", how="left")
    s["days_from_departure"] = (s["announced"] - s["exit_date"]).dt.days
    dsf = crsp.add_delisting(crsp.load_dsf(W / "stage4/dsf"), pd.read_parquet(W / "stage4/dsedelist.parquet"))
    panel = crsp.returns_panel(dsf, pd.read_parquet(W / "stage3/ff_factors_daily.parquet"))
    eng = EventStudyEngine(panel, crsp.MARKET, rps.RULE, WINDOWS)
    rows = []
    for r in s.itertuples():
        res = eng.run_event(r.case, str(int(r.permno)), r.announced)
        if res.status == "ok":
            w = res.windows.set_index("window")
            rows.append({"case": r.case, **{f"car_{k}": w.loc[k, "car"] + w.loc[k, "alpha_part"] for k in w.index}})
    s = s.merge(pd.DataFrame(rows), on="case", how="left")
    s.to_parquet(ROOT / "data/derived/successors.parquet", index=False)

    def summ(g: pd.DataFrame, label: str) -> dict:
        x = g["car_m1_1"].dropna()
        return {"group": label, "cases": len(x), "mean_m1_1": x.mean(), "median_m1_1": x.median(),
                "se_m1_1": x.std() / np.sqrt(len(x)) if len(x) > 2 else np.nan, "positive": int((x > 0).sum()),
                "mean_p2_20": g["car_p2_20"].mean(), "median_days_from_departure": g["days_from_departure"].median()}
    out = pd.DataFrame([summ(s, "all")] + [summ(s[s["origin"] == o], o)
                                           for o in ["outside hire", "insider", "interim made permanent"]])
    out.to_csv(ROOT / "results/successor_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    print(s[["company", "successor", "origin", "announced", "days_from_departure", "car_d0", "car_m1_1", "car_p2_20"]]
          .round(4).sort_values("announced").to_string(index=False))


if __name__ == "__main__":
    main()
