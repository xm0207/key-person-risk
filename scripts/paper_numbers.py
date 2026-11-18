"""Numbers quoted in the text of paper/personal_scandals.tex that no table shows.

Every such number is computed here from the pipeline's outputs and written to results/paper_text_numbers.csv,
with the string as the paper quotes it and whether that string appears in the .tex file. A number that does
not appear is printed, so a change in the data cannot leave a stale number in the text unnoticed.

Inputs: data/derived/personal_scandals.parquet (run_personal_scandals.py), data/derived/departure_windows.parquet
(run_departure_study.py), data/derived/event_sample.parquet, events/gentry_misconduct_coding.csv,
data/wrds/stage4/fundq_rdq.parquet, results/nvidia_timeline.csv, results/nvidia_attribution.csv, the Yahoo chip-maker
prices, results/prediction_market_*.csv, results/successor_origin.csv and results/successor_summary.csv.

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

    nvidia_extra(add, px)
    successor_numbers(add)

    out = pd.DataFrame(rows)
    tex = TEX.read_text()
    out["in_paper"] = [q in tex for q in out["as_quoted"]]
    out.to_csv(ROOT / "results/paper_text_numbers.csv", index=False)
    missing = out[~out["in_paper"]]
    print(f"{len(out)} numbers; {len(missing)} not found in the paper as quoted")
    if len(missing):
        print(missing[["section", "quantity", "as_quoted"]].to_string(index=False))


def nvidia_extra(add, px: pd.Series) -> None:
    """Nvidia: sign test of the earlier cases, chip makers, attribution, Kalshi steps and reinstatement odds."""
    from scipy.stats import binomtest

    d = pd.read_parquet(ROOT / "data/derived/personal_scandals.parquet")
    two = d[d["disclosure"] == "two-stage"]["main_exit"].dropna()
    p = binomtest(int((two > 0).sum()), len(two), 0.5, alternative="greater").pvalue
    add("Cases: allegation first", "sign test, rises at the exit (one-sided p)", p, f"probability of {p:.3f}")

    att = pd.read_csv(ROOT / "results/nvidia_attribution.csv").set_index("model")
    a, b = att.loc["A: SPY + mega-caps"], att.loc["B: SPY + mega-caps + chip makers"]
    ep, d4 = a["29 Oct - 17 Nov: episode"], a["4 Nov: day after the election"]
    add("Nvidia", "episode with the chip basket", b["29 Oct - 17 Nov: episode"], pc(b["29 Oct - 17 Nov: episode"], math=True))
    add("Nvidia", "9-10 Nov with the chip basket", b["9-10 Nov: board investigation; CEO's email"],
        pc(b["9-10 Nov: board investigation; CEO's email"], math=True))
    add("Nvidia", "4 Nov abnormal, model A", d4, pc(d4, math=True))
    add("Nvidia", "4 Nov abnormal, with chips", b["4 Nov: day after the election"], pc(b["4 Nov: day after the election"], math=True))
    add("Nvidia", "episode without 4 Nov", ep - d4, pc(ep - d4, math=True))
    add("Nvidia", "5-6 Nov abnormal", a["5-6 Nov: House request; memo report"], pc(a["5-6 Nov: House request; memo report"], sign=True, math=True))
    add("Nvidia", "episode in dollars ($tn, value on 28 Oct)", -ep * SHARES_OUTSTANDING * px["2026-10-28"] / 1e12,
        f"about \\${-ep * SHARES_OUTSTANDING * px['2026-10-28'] / 1e12:.1f} trillion")

    import keyperson.data as kd
    from keyperson.config import FIRMS, MIN_BASKET_MEMBERS
    r = kd.load_returns(ROOT / "data/keyperson.duckdb")
    r["mega"] = kd.ex_firm_basket(r, FIRMS, "NVDA", MIN_BASKET_MEMBERS)
    chips = pd.read_parquet(ROOT / "data/yahoo_sessions/semis_daily_adjclose.parquet").pct_change()
    r["chips"] = chips.mean(axis=1).reindex(r.index)
    two_day = lambda x: (1 + x.loc["2026-11-09":"2026-11-10"]).prod() - 1
    add("Nvidia", "chip basket 9-10 Nov", two_day(r["chips"]), pc(two_day(r["chips"])))
    add("Nvidia", "AMD 9-10 Nov", two_day(chips["AMD"]), f"AMD rose by {pc(two_day(chips['AMD']))}")
    add("Nvidia", "Broadcom 9-10 Nov", two_day(chips["AVGO"]), f"Broadcom by {pc(two_day(chips['AVGO']))}")
    for day, col, txt in [("2026-10-30", "chips", "by {} on 30 October"), ("2026-11-05", "chips", "by {} on 5 November"),
                          ("2026-11-05", "SPY", "SPY fell by {}"), ("2026-11-05", "NVDA", "and Nvidia by {}"),
                          ("2026-11-16", "chips", "and by {} on 16 November"), ("2026-11-04", "SPY", "SPY rose by {}"),
                          ("2026-11-04", "mega", "the other mega-caps by {}"), ("2026-11-04", "chips", "the chip makers fell by {}"),
                          ("2026-11-04", "NVDA", "and Nvidia fell by {}")]:
        v = r.loc[day, col]
        add("Nvidia", f"{col} on {day}", v, txt.format(pc(v)))

    k = pd.read_csv(ROOT / "results/prediction_market_probabilities.csv").set_index("time_et")
    add("Nvidia", "2029 contract, 6 Nov close", k.loc["2026-11-06 16:00", "exit_by_2029"], f"{100 * k.loc['2026-11-06 16:00', 'exit_by_2029']:.1f}\\%")
    add("Nvidia", "2029 contract, 10 Nov close", k.loc["2026-11-10 16:00", "exit_by_2029"], f"{100 * k.loc['2026-11-10 16:00', 'exit_by_2029']:.1f}\\%")
    add("Nvidia", "2027 contract, 8 Nov evening", k.loc["2026-11-08 18:00", "exit_by_2027"], f"{100 * k.loc['2026-11-08 18:00', 'exit_by_2027']:.1f}\\%")
    add("Nvidia", "2027 contract, 10 Nov close", k.loc["2026-11-10 16:00", "exit_by_2027"], f"{100 * k.loc['2026-11-10 16:00', 'exit_by_2027']:.1f}\\%")
    board = nv_board = pd.read_csv(ROOT / "results/nvidia_timeline.csv").set_index("start").loc["2026-11-09", "car_basket"]
    p10, p6, first_trade = k.loc["2026-11-10 16:00", "exit_by_2029"], k.loc["2026-11-06 16:00", "exit_by_2029"], 0.11
    add("Nvidia", "board step: rise in p (pp)", p10 - first_trade, f"between {100 * (p10 - first_trade):.1f} and {100 * (p10 - p6):.0f} percentage points")
    lo, hi = -board / (p10 - p6), -board / (p10 - first_trade)
    add("Nvidia", "board step: K - D bounds", lo, f"between {100 * lo:.0f}\\% and {100 * hi:.0f}\\%")
    v6 = SHARES_OUTSTANDING * px["2026-11-06"] / 1e12
    add("Nvidia", "board step: K - D bounds ($tn)", lo * v6, f"\\${lo * v6:.1f} to \\${hi * v6:.1f} trillion")

    st = pd.read_csv(ROOT / "results/prediction_market_departure_steps.csv").set_index("step")
    ev, an, cc = (st.loc["11 Nov, 16:00-20:00: evening reports"], st.loc["12 Nov, 08:45-08:55: resignation announced"],
                  st.loc["11 Nov 16:00 - 12 Nov 16:00: close to close"])
    add("Nvidia", "evening: 2027 from", ev["exit_by_2027_start"], f"from {100 * ev['exit_by_2027_start']:.1f}\\% to {100 * ev['exit_by_2027_end']:.1f}\\%")
    add("Nvidia", "evening: abnormal", ev["nvidia_abnormal"], pc(ev["nvidia_abnormal"], math=True))
    add("Nvidia", "announcement: 2027 from", an["exit_by_2027_start"], f"from {100 * an['exit_by_2027_start']:.1f}\\% to {100 * an['exit_by_2027_end']:.1f}\\%")
    add("Nvidia", "announcement: abnormal", an["nvidia_abnormal"], pc(an["nvidia_abnormal"], math=True))
    rises = [an[f"exit_by_{c}_end"] - an[f"exit_by_{c}_start"] for c in ("2029", "2027", "dec1")]
    add("Nvidia", "announcement: rises in p", rises[0], f"by {100 * rises[0]:.0f} percentage points on the contract with a deadline of 2029, by {100 * rises[1]:.0f} points")
    add("Nvidia", "announcement: rise in p, Dec", rises[2], f"by {100 * rises[2]:.0f} points on the contract with a deadline of 1 December")
    k29, k27, kdec = (-an["nvidia_abnormal"] / x for x in rises)
    add("Nvidia", "announcement: K - D with the 2029 contract", k29, f"$K - D$ was {100 * k29:.1f}\\% of Nvidia's value with the 2029 contract")
    add("Nvidia", "announcement: K - D with the 2027 and Dec contracts", k27, f"and {100 * k27:.1f}\\% or {100 * kdec:.1f}\\% with the contracts")
    ep = pd.read_csv(ROOT / "results/nvidia_attribution.csv").set_index("model").loc["A: SPY + mega-caps"]
    ex4 = ep["29 Oct - 17 Nov: episode"] - ep["4 Nov: day after the election"]
    v28 = SHARES_OUTSTANDING * px["2026-10-28"] / 1e12
    add("Nvidia", "episode without 4 Nov ($tn)", -ex4 * v28, f"about \\${-ex4 * v28:.1f} trillion")
    add("Nvidia", "abstract range in dollars", -ex4 * v28, f"(\\${-ex4 * v28:.1f} to \\${-ep['29 Oct - 17 Nov: episode'] * v28:.1f} trillion)")
    ev29 = (ev["exit_by_2029_start"], ev["exit_by_2029_end"])
    add("Nvidia", "evening: 2029 contract", ev29[0], f"$p$ rose from {100 * ev29[0]:.1f}\\% to {100 * ev29[1]:.1f}\\%")
    add("Nvidia", "rest of pre-market", st.loc["12 Nov, 08:55-09:30: rest of the pre-market session", "nvidia_abnormal"],
        pc(st.loc["12 Nov, 08:55-09:30: rest of the pre-market session", "nvidia_abnormal"], math=True))
    add("Nvidia", "close to close (5-minute)", cc["nvidia_abnormal"], pc(cc["nvidia_abnormal"], math=True))
    r29, r27 = cc["exit_by_2029_end"] - cc["exit_by_2029_start"], cc["exit_by_2027_end"] - cc["exit_by_2027_start"]
    add("Nvidia", "close to close: rises in p", r29, f"by {100 * r29:.0f} points on the 2029 contract and by {100 * r27:.0f} points")
    lo, hi = -cc["nvidia_abnormal"] / r27, -cc["nvidia_abnormal"] / r29
    add("Nvidia", "close to close: K - D bound", lo, f"{100 * lo:.1f}\\% to {100 * hi:.1f}\\%")

    co = pd.read_csv(ROOT / "results/prediction_market_comoves.csv")
    for ev_name, fmt in [("16 Nov: report that the board is not considering a return", "between {} and {}"),
                         ("17 Nov: offer to return", "between {} and {}"), ("17 Nov: chair rules out a return on the call", "from {} to {}")]:
        x = co[co["event"] == ev_name]
        dq = (x["odds_end"] - x["odds_start"]).abs()
        lo_r, hi_r = sorted([x["nvidia_abnormal"].min(), x["nvidia_abnormal"].max()], key=abs) if ev_name.startswith("16") else (x["nvidia_abnormal"].min(), x["nvidia_abnormal"].max())
        sgn = not ev_name.startswith("16")
        add("Nvidia", f"{ev_name}: abnormal range", lo_r, fmt.format(pc(lo_r, sign=sgn, math=True), pc(hi_r, sign=sgn, math=True)))
        if not ev_name.startswith("17 Nov: chair"):  # the text gives no odds change for the call
            add("Nvidia", f"{ev_name}: odds change range (pp)", dq.min(), f"by {100 * dq.min():.0f} to {100 * dq.max():.0f}")
    cond = pd.read_csv(ROOT / "results/prediction_market_conditional.csv").iloc[0]
    add("Nvidia", "Polymarket: P(>4T | named)", cond["p_above_4T_if_named"], f"{100 * cond['p_above_4T_if_named']:.0f}\\% probability")
    add("Nvidia", "Polymarket: P(>4T | not named)", cond["p_above_4T_if_not_named"], f"and {100 * cond['p_above_4T_if_not_named']:.0f}\\% if he were not")


def successor_numbers(add) -> None:
    k = pd.read_csv(ROOT / "results/successor_origin.csv").set_index("group")
    a, o, i = (k.loc["forced: all with an identified successor"], k.loc["forced: named at the departure, outsider"],
               k.loc["forced: named at the departure, insider"])
    now, later = k.loc["forced: named at the departure, any origin"], k.loc["forced: named later, any origin"]
    cfo, diff = k.loc["forced: named at the departure, chief financial officer"], k.loc["forced, named at the departure: outsider minus insider"]
    add("Successors", "forced departures with a successor", a["cases"], f"{int(a['cases']):,} forced departures")
    add("Successors", "in office within 30 days", now["cases"], f"{int(now['cases'])} had a successor in office within 30 days")
    add("Successors", "outsider", o["median_m1_1"], f"was {pc(o['median_m1_1'], sign=True, math=True)} (mean {pc(o['mean_m1_1'], sign=True, math=True)}, standard error {100 * o['se_m1_1']:.1f}, {int(o['cases'])} cases)")
    add("Successors", "insider", i["median_m1_1"], f"it was {pc(i['median_m1_1'], math=True)} (mean {pc(i['mean_m1_1'], math=True)}, standard error {100 * i['se_m1_1']:.1f}, {int(i['cases'])} cases)")
    add("Successors", "difference in medians and rank-sum p", diff["median_m1_1"], f"{100 * diff['median_m1_1']:.1f} percentage points, with a p-value of {diff['rank_sum_p']:.2f}")
    add("Successors", "named later", later["mean_m1_1"], f"was {pc(later['mean_m1_1'], math=True)} (standard error {100 * later['se_m1_1']:.1f}), against {pc(now['mean_m1_1'], math=True)} (standard error {100 * now['se_m1_1']:.1f})")
    add("Successors", "CFO successors", cfo["median_m1_1"], f"In the {['zero','one','two','three','four','five','six','seven','eight','nine','ten'][int(cfo['cases'])]} forced departures in which the chief financial officer became chief executive at the departure, the median abnormal return was {pc(cfo['median_m1_1'], math=True)}")
    h = pd.read_csv(ROOT / "results/successor_summary.csv").set_index("group")
    add("Successors", "hand-checked: all", h.loc["all", "mean_m1_1"], f"the mean abnormal return was {pc(h.loc['all', 'mean_m1_1'], math=True)}, and it was positive in {['zero','one','two','three','four','five','six','seven','eight','nine','ten'][int(h.loc['all', 'positive'])]} of the {int(h.loc['all', 'cases'])} cases")
    add("Successors", "hand-checked: outside", h.loc["outside hire", "mean_m1_1"], f"{pc(h.loc['outside hire', 'mean_m1_1'], math=True)} for the {int(h.loc['outside hire', 'cases'])} successors hired from outside")
    add("Successors", "hand-checked: interim made permanent", h.loc["interim made permanent", "mean_m1_1"], f"{pc(h.loc['interim made permanent', 'mean_m1_1'], sign=True, math=True)} for the five interim")
    add("Successors", "hand-checked: median days", h.loc["all", "median_days_from_departure"], f"median of {h.loc['all', 'median_days_from_departure']:.0f} days")


if __name__ == "__main__":
    main()
