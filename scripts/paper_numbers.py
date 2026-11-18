"""Numbers quoted in the text of paper/personal_scandals.tex that no table shows.

Every such number is computed here from the pipeline's outputs and written to results/paper_text_numbers.csv,
with the string as the paper quotes it and whether that string appears in the .tex file. A number that does
not appear is printed, so a change in the data cannot leave a stale number in the text unnoticed.

Inputs: data/derived/personal_scandals.parquet (run_personal_scandals.py), data/derived/departure_windows.parquet
(run_departure_study.py), data/derived/event_sample.parquet, events/gentry_misconduct_coding.csv,
data/wrds/stage4/fundq_rdq.parquet, results/nvidia_timeline.csv.

    ~/.venvs/key-person-risk/bin/python scripts/paper_numbers.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "data/wrds"
TEX = ROOT / "paper/personal_scandals.tex"
# Consumer-facing industries, two-digit SIC: food, tobacco, textiles, apparel, leather, retail, restaurants,
# hotels, personal services, film, amusement, communications, publishing
CONSUMER = {20, 21, 22, 23, 31, 52, 53, 54, 55, 56, 57, 58, 59, 70, 72, 78, 79, 48, 27}
EARNINGS_DAYS = 3  # a departure within three calendar days of a quarterly earnings date
# Nvidia's shares outstanding on 23 March 2026 (2026 proxy statement); used only to express returns in dollars
SHARES_OUTSTANDING = 24_312_141_810
EXAMPLES = {"G14087-200503": "Boeing 2005", "G47026-201708": "Intel 2018", "G43399-201807": "Texas Instruments 2018",
            "G50557-201911": "McDonald's 2019", "G46235-201604": "Booking 2016", "G30132-201008": "HP 2010"}


def t(x: pd.Series) -> float:
    x = x.dropna()
    return x.mean() / (x.std() / np.sqrt(len(x)))


def pc(x: float, sign: bool = False, math: bool = False) -> str:
    """A share as the paper writes it: 3.2\\%, $-1.5$\\% or $+2.2$\\%."""
    v = 100 * x
    if math:
        return f"${v:+.1f}$\\%" if sign else f"${v:.1f}$\\%"
    return f"{abs(v):.1f}\\%"


def tt(x: float) -> str:
    return f"$t = {x:.1f}$"


def se(x: pd.Series) -> str:
    """Standard error of a mean, in percentage points, as the paper writes it."""
    x = x.dropna()
    return f"standard error {100 * x.std() / np.sqrt(len(x)):.1f}"


def gentry_departures() -> pd.DataFrame:
    """Three-day announcement returns of Gentry code-4 departures by personal or corporate misconduct, from the
    departure study (market model with the intercept set to zero)."""
    ev = pd.read_parquet(ROOT / "data/derived/event_sample.parquet")
    ev["event_id"] = ev["dismissal_dataset_id"].astype(str)
    w = pd.read_parquet(ROOT / "data/derived/departure_windows.parquet")
    w = w[w["complete"].astype(bool)]
    car = w.pivot_table(index="event_id", columns="window", values="car_noalpha", aggfunc="first")
    g = pd.read_csv(ROOT / "events/gentry_misconduct_coding.csv")
    d = ev.set_index("event_id").join(car)
    d = d.join(g.assign(event_id=g["dismissal_dataset_id"].astype(str)).set_index("event_id")["gentry_class"])
    d = d[(d["departure_code"] == 4) & d["ann_m1_1"].notna()]
    return d


def nvidia_closes() -> pd.Series:
    import duckdb
    con = duckdb.connect(str(ROOT / "data/keyperson.duckdb"), read_only=True)
    try:
        d = con.execute("select date, value from observations where dataset = 'price' and series = 'NVDA' "
                        "and date between '2026-10-01' and '2026-12-31'").df()
    finally:
        con.close()
    return d.assign(date=d["date"].astype(str)).set_index("date")["value"]


def near_earnings(cases: pd.DataFrame) -> pd.Series:
    rdq = pd.read_parquet(W / "stage4/fundq_rdq.parquet").dropna(subset=["rdq"])
    rdq["gvkey"] = rdq["gvkey"].astype(str).str.zfill(6)
    by = {k: pd.to_datetime(v["rdq"]).to_numpy() for k, v in rdq.groupby("gvkey")}
    out = []
    for gv, day in zip(cases["gvkey"].astype(str).str.zfill(6), cases["first_news"]):
        a = by.get(gv)
        out.append(bool(a is not None and (np.abs((a - np.datetime64(day)) / np.timedelta64(1, "D")) <= EARNINGS_DAYS).any()))
    return pd.Series(out, index=cases.index)


def main() -> None:
    d = pd.read_parquet(ROOT / "data/derived/personal_scandals.parquet").set_index("case")
    rows = []

    def add(section, quantity, value, quoted):
        rows.append({"section": section, "quantity": quantity, "value": value, "as_quoted": quoted})

    one = d[d["disclosure"] == "one-stage"]
    add("Cases: quiet exits", "one-stage: mean three-day CAR at the announcement", one["main_first"].mean(), pc(one["main_first"].mean()))
    add("Cases: quiet exits", "one-stage: standard error of the mean", t(one["main_first"]), se(one["main_first"]))
    add("Cases: quiet exits", "one-stage: median", one["main_first"].median(), pc(one["main_first"].median()))
    add("Cases: quiet exits", "one-stage: mean CAR [-20,-2]", one["main_pre"].mean(), pc(one["main_pre"].mean()))
    add("Cases: quiet exits", "one-stage: standard error of the pre-window mean", t(one["main_pre"]), se(one["main_pre"]))
    usd = np.exp(one["log_mv"]) * one["main_first"]
    add("Cases: quiet exits", "one-stage: median dollar loss ($m)", usd.median(), f"\\${-usd.median():.0f} million")

    earn = near_earnings(one)
    add("Cases: quiet exits", "one-stage: departures within three days of earnings", int(earn.sum()), "Nine of the 66")
    add("Cases: quiet exits", "one-stage with earnings: mean CAR", one.loc[earn, "main_first"].mean(), pc(one.loc[earn, "main_first"].mean()))
    x = one.loc[~earn, "main_first"]
    add("Cases: quiet exits", "one-stage without earnings: mean CAR and standard error", x.mean(), f"{pc(x.mean())} ({se(x)})")

    cons = one["sic2"].isin(CONSUMER)
    x, y = one.loc[cons, "main_first"], one.loc[~cons, "main_first"]
    add("Cases: quiet exits", "one-stage consumer-facing: cases", int(cons.sum()), f"the {int(cons.sum())} one-stage firms")
    add("Cases: quiet exits", "one-stage consumer-facing: mean, median, t", x.mean(),
        f"{pc(x.mean())} (median {pc(x.median())}, {se(x)})")
    add("Cases: quiet exits", "one-stage other industries: mean and median", y.mean(), f"{pc(y.mean())} (median {pc(y.median())})")

    g = gentry_departures()
    for cls, label in [("PERS", "personal conduct"), ("CORP", "corporate wrongdoing")]:
        z = g.loc[g["gentry_class"] == cls, "ann_m1_1"]
        add("Cases: quiet exits", f"Gentry departures over {label}: cases", len(z), f"{len(z)} departures over {label}")
        add("Cases: quiet exits", f"Gentry departures over {label}: mean and t", z.mean(), f"{pc(z.mean())}")
        add("Cases: quiet exits", f"Gentry departures over {label}: standard error", t(z), se(z))

    for case, label in EXAMPLES.items():
        v = d.loc[case, "main_first"]
        add("Cases: quiet exits", f"{label}: three-day CAR", v, pc(v) if label.startswith(("HP", "Boeing", "Intel")) else
            pc(v, sign=True, math=True) if v > 0 else pc(v, math=True))

    two = d[d["disclosure"] == "two-stage"]
    path = two[["main_first", "main_interim", "main_exit"]].fillna(0).sum(axis=1)
    add("Cases: allegation first", "two-stage: first news to exit, mean", path.mean(), pc(path.mean(), math=True))
    add("Cases: allegation first", "two-stage: first news to exit, median", path.median(), pc(path.median(), math=True))
    add("Cases: allegation first", "Wynn Resorts: first news to exit", path["G37104-201802"], f"about {abs(100 * path['G37104-201802']):.0f}\\%")

    nv = pd.read_csv(ROOT / "results/nvidia_timeline.csv")
    nv = nv[nv["start"].notna()].set_index("start")
    episode = nv.loc[nv.index >= "2026-10-29", "car_basket"].sum()
    add("Nvidia", "company-specific return from 29 October to the last day in the data", episode, f"{pc(episode)}")
    px = nvidia_closes()
    before = SHARES_OUTSTANDING * px["2026-10-28"] / 1e12
    add("Nvidia", "market value before the first report ($tn)", before, f"\\${before:.1f} trillion")
    bn = -nv.loc[nv.index == "2026-11-09", "car_basket"].sum() * SHARES_OUTSTANDING * px["2026-11-06"] / 1e9
    add("Nvidia", "9-10 November loss in dollars ($bn, value on 6 November)", bn, f"about \\${round(bn, -2):.0f} billion")
    after = nv.loc[["2026-11-13", "2026-11-16", "2026-11-17"], "car_basket"].sum()
    add("Nvidia", "Nvidia: 13-17 November, company-specific", after, pc(after, math=True))
    add("Nvidia", "Nvidia: 16 November", nv.loc["2026-11-16", "car_basket"], pc(nv.loc["2026-11-16", "car_basket"], math=True))
    add("Nvidia", "Nvidia: 17 November", nv.loc["2026-11-17", "car_basket"], pc(nv.loc["2026-11-17", "car_basket"], sign=True, math=True))

    out = pd.DataFrame(rows)
    tex = TEX.read_text()
    out["in_paper"] = [q in tex for q in out["as_quoted"]]
    out.to_csv(ROOT / "results/paper_text_numbers.csv", index=False)
    missing = out[~out["in_paper"]]
    print(f"{len(out)} numbers; {len(missing)} not found in the paper as quoted")
    if len(missing):
        print(missing[["section", "quantity", "as_quoted"]].to_string(index=False))


if __name__ == "__main__":
    main()
