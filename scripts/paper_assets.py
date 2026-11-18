"""Tables and the figure for paper/personal_scandals.tex, from the personal-scandal results.

    cd scripts && ~/.venvs/key-person-risk/bin/python paper_assets.py

Writes paper/tables/*.tex and paper/fig_paths.pdf. Every number comes from data/derived/personal_scandals.parquet,
events/gentry_misconduct_coding.csv, results/personal_scandal_*.csv or results/nvidia_timeline.csv; nothing is
typed in by hand. Numbers quoted in the text but not shown in a table come from scripts/paper_numbers.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keyperson import crsp
from keyperson.engine import EventStudyEngine, Window

import run_personal_scandals as rps  # same folder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/tables"
ORDER = ["one-stage", "two-stage", "survived"]
LABEL = {"one-stage": "One-stage", "two-stage": "Two-stage", "survived": "CEO stays"}


def pct(x, d=1):
    return "--" if pd.isna(x) else f"{100 * x:.{d}f}"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body)


def table_period() -> None:
    g = pd.read_csv(ROOT / "events/gentry_misconduct_coding.csv", parse_dates=["event_date"])
    g["period"] = pd.cut(g["event_date"].dt.year, [1999, 2004, 2009, 2014, 2019, 2024],
                         labels=["2000--04", "2005--09", "2010--14", "2015--19", "2020--22"])
    t = g.pivot_table(index="period", columns="gentry_class", values="dismissal_dataset_id", aggfunc="count",
                      fill_value=0, observed=False)
    rows = [f"{p} & {r.get('PERS', 0)} & {r.get('CORP', 0)} & {r.get('OTHER', 0) + r.get('UNC', 0)} \\\\"
            for p, r in t.iterrows()]
    tot = t.sum()
    rows.append(r"\midrule All & " + f"{tot.get('PERS', 0)} & {tot.get('CORP', 0)} & {tot.get('OTHER', 0) + tot.get('UNC', 0)} \\\\")
    write("t_period.tex", "\n".join(rows) + "\n")


def table_cars(d: pd.DataFrame) -> None:
    """Two rows per group: mean [median]; then the standard error of the mean (one-stage) or negatives out of cases."""
    cols = ["pre", "first_day0", "first", "interim", "exit", "total", "post"]
    rows = []
    for k in ORDER:
        g = d[d["disclosure"] == k]
        top, bottom = [], []
        for c in cols:
            x = g[f"main_{c}"].dropna()
            if not len(x):
                top.append("--"); bottom.append("")
                continue
            top.append(f"{pct(x.mean())} [{pct(x.median())}]")
            if k == "one-stage":
                bottom.append(f"({100 * x.std() / np.sqrt(len(x)):.1f})")
            else:
                bottom.append(f"{int((x < 0).sum())}/{len(x)}")
        rows.append(f"{LABEL[k]} & {len(g)} & " + " & ".join(top) + r" \\")
        rows.append(r" & & " + " & ".join(r"{\scriptsize " + b + "}" if b else "" for b in bottom) + r" \\[2pt]")
    write("t_cars.tex", "\n".join(rows) + "\n")


def table_robust() -> None:
    s = pd.read_csv(ROOT / "results/personal_scandal_summary.csv")
    r = pd.read_csv(ROOT / "results/personal_scandal_robustness.csv")
    rows = []
    for k in ORDER:
        x = s[s["disclosure"] == k].set_index("variant")
        loo = r[(r["disclosure"] == k) & (r["sample"] == "all") & (r["window"] == "first")]
        loo_txt = f"{pct(loo['loo_min'].iloc[0])} to {pct(loo['loo_max'].iloc[0])}" if len(loo) else "--"
        rows.append(f"{LABEL[k]} & {pct(x.loc['main', 'mean_first'])} & {pct(x.loc['full', 'mean_first'])} & "
                    f"{pct(x.loc['mktadj', 'mean_first'])} & {loo_txt} & {pct(x.loc['main', 'mean_exit'])} & "
                    f"{pct(x.loc['full', 'mean_exit'])} & {pct(x.loc['mktadj', 'mean_exit'])} \\\\")
    core = r[(r["sample"].str.startswith("core")) & (r["window"] == "first")]
    rows.append(f"CEO stays, without Amazon (a divorce) & {pct(core['mean'].iloc[0])} & & & & & & \\\\")
    write("t_robust.tex", "\n".join(rows) + "\n")
    write("v_core_n.tex", str(int(core["cases"].iloc[0])))


def table_coverage() -> None:
    c = pd.read_csv(ROOT / "results/personal_scandal_coverage.csv")
    rows = []
    for t in ["small", "middle", "large"]:
        x = c[c["size_tercile"] == t].set_index("public")
        cell = lambda p, col, f: (f(x.loc[p, col]) if p in x.index else "--")
        rows.append(f"{t.capitalize()} & {cell(False, 'cases', int)} & {cell(True, 'cases', int)} & "
                    f"{cell(False, 'coverage_median', lambda v: f'{v:.0f}')} & {cell(True, 'coverage_median', lambda v: f'{v:.0f}')} & "
                    f"{cell(False, 'founder_share', lambda v: pct(v, 0))} & {cell(True, 'founder_share', lambda v: pct(v, 0))} \\\\")
    write("t_coverage.tex", "\n".join(rows) + "\n")


def table_power(d: pd.DataFrame) -> None:
    d = d.assign(mv_bn=np.exp(d["log_mv"]) / 1000)
    rows = []
    for k in ORDER:
        g = d[d["disclosure"] == k]
        rows.append(f"{LABEL[k]} & {len(g)} & {int(g['founder'].sum())} & {pct(g['duality'].mean(), 0)} & "
                    f"{g['tenure_years'].median():.1f} & {g['mv_bn'].median():.1f} & {g['coverage'].median():.0f} "
                    f"({int(g['coverage'].notna().sum())}) \\\\")
    write("t_power.tex", "\n".join(rows) + "\n")


CASES = {  # case id -> (firm, CEO, what became public first), in the words of the sources
    "G31032-201205": ("Chesapeake Energy", "Aubrey McClendon", "news agency report on personal loans backed by his stakes in the firm's wells"),
    "G59706-202103": ("Tronox", "Jeffry Quinn", "leave of absence; referred to in an insider-trading complaint about another firm's shares"),
    "G37104-201802": ("Wynn Resorts", "Steve Wynn", "newspaper report of sexual-misconduct allegations"),
    "G29214-201809": ("CBS", "Leslie Moonves", "board to investigate allegations of misconduct; magazine article"),
    "G68457-202106": ("AdaptHealth", "Luke McGee", "charged with tax fraud over past private activity; unpaid leave"),
    "G51870-202207": ("WWE", "Vince McMahon", "board probe of payments by the CEO"),
    "S0110": ("Norfolk Southern", "Alan Shaw", "board investigation of the CEO's conduct"),
    "M0001": ("Yahoo", "Scott Thompson", "activist letter: a degree in his biography that he did not hold"),
    "C0101": ("Whole Foods Market", "John Mackey", "pseudonymous online posts about the firm and a rival"),
    "C0253": ("Amazon", "Jeff Bezos", "CEO announces his divorce; reports on his private life follow"),
    "C0263": ("Best Buy", "Corie Barry", "letter alleging an undisclosed relationship"),
    "S0115": ("Activision Blizzard", "Bobby Kotick", "report that he knew of misconduct claims; settled claims about his own conduct"),
    "S0101": ("Tesla", "Elon Musk", "report of an alleged affair, denied"),
}


def table_cases(d: pd.DataFrame) -> None:
    rows = []
    for k in ["two-stage", "survived"]:
        g = d[d["disclosure"] == k].sort_values("first_news")
        rows.append(r"\multicolumn{7}{l}{\textit{" + LABEL[k] + r"}} \\")
        for r in g.itertuples():
            firm, ceo, what = CASES[r.case]
            mark = r"$^\dagger$" if (r.flag_alpha or r.flag_jump) else ""
            ex = "--" if pd.isna(r.exit_date) else f"{r.exit_date:%d %b %Y}"
            rows.append(f"{firm} ({ceo}){mark} & {what} & {r.first_news:%d %b %Y} & {ex} & {pct(r.main_first_day0)} & "
                        f"{pct(r.main_first)} & {pct(r.main_exit) if k == 'two-stage' else '--'} \\\\")
    write("t_cases.tex", "\n".join(rows) + "\n")


def table_nvidia() -> None:
    t = pd.read_csv(ROOT / "results/nvidia_timeline.csv")
    steps = t[t["start"].notna()]
    rows = []
    for r in steps.itertuples():
        label = re.sub(r"(\d+)-(\d+)", r"\1--\2", r.window).replace("%", r"\%")
        se = abs(100 * r.car_basket / r.t_basket)
        rows.append(f"{label} & {pct(r.car_basket)} & ({se:.1f}) & {pct(r.car_smh)} & {pct(r.car_market_only)} \\\\")
    write("t_nvidia.tex", "\n".join(rows) + "\n")
    share = t.loc[t["window"].str.contains(r"\(basket\)"), "car_basket"].iloc[0]
    share_smh = t.loc[t["window"].str.contains(r"\(smh\)"), "car_smh"].iloc[0]
    write("v_nvidia_share.tex", f"{100 * share:.0f}")
    write("v_nvidia_share_smh.tex", f"{100 * share_smh:.0f}")


def table_sessions() -> None:
    """Price discovery by trading session (scripts/nvidia_sessions.py)."""
    s = pd.read_csv(ROOT / "results/nvidia_sessions.csv")
    rows = []
    for w in ["two months before", "one month before", "episode"]:
        x = s[s["window"] == w].set_index("series")
        cs, bk, nv = x.loc["Nvidia, company-specific"], x.loc["peer basket"], x.loc["Nvidia"]
        span = f"{pd.Timestamp(cs['first_day']):%d %b}--{pd.Timestamp(cs['last_day']):%d %b}"
        rows.append(f"{w.capitalize()} ({span}) & {int(cs['days'])} & {100 * cs['wpc_pre']:.0f} & {100 * cs['wpc_regular']:.0f} & "
                    f"{100 * cs['wpc_after']:.0f} & {100 * (cs['wpc_pre'] + cs['wpc_after']):.0f} & "
                    f"{100 * (bk['wpc_pre'] + bk['wpc_after']):.0f} & {cs['mean_abs_bp_day']:.0f} & "
                    f"{nv['volume_vs_two_months_before']:.1f} \\\\")
    write("t_sessions.tex", "\n".join(rows) + "\n")
    k = pd.read_csv(ROOT / "results/nvidia_session_days.csv")
    rows = []
    for r in k.itertuples():
        rows.append(f"{pd.Timestamp(r.day):%d %b} & {r.news} & {pct(r.pre)} & {pct(r.regular)} & {pct(r.after)} & "
                    f"{pct(r.total)} & {r.volume_vs_two_months_before:.1f} \\\\")
    write("t_session_days.tex", "\n".join(rows) + "\n")


def figure(d: pd.DataFrame) -> None:
    dsf = crsp.add_delisting(crsp.load_dsf(rps.WRDS / "stage4/dsf"), pd.read_parquet(rps.WRDS / "stage4/dsedelist.parquet"))
    panel = crsp.returns_panel(dsf, pd.read_parquet(rps.WRDS / "stage3/ff_factors_daily.parquet"))
    days = tuple(Window(f"d{k:+d}", k, k) for k in range(-20, 41))
    eng = EventStudyEngine(panel, crsp.MARKET, rps.RULE, days + (Window("anchor", -120, -120),))
    d = d[d["main_first"].notna()]
    ev = pd.DataFrame({"event_id": d["case"], "ticker": d["permno"].astype("int64").astype(str), "day0": d["first_news"]})
    w, _, _ = eng.run(ev)
    w = w[w["window"] != "anchor"].merge(d[["case", "disclosure"]], left_on="event_id", right_on="case")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    styles = {"one-stage": ("#1f77b4", "-"), "two-stage": ("#d62728", "-"), "survived": ("#2ca02c", "--")}
    for k in ORDER:
        g = w[w["disclosure"] == k]
        path = (g["car"] + g["alpha_part"]).groupby(g["rel_start"]).mean().sort_index().cumsum()
        n = g["event_id"].nunique()
        ax.plot(path.index, 100 * path.values, color=styles[k][0], ls=styles[k][1], label=f"{LABEL[k]} (n = {n})")
    med = (d.loc[d["disclosure"] == "two-stage", "exit_date"] - d.loc[d["disclosure"] == "two-stage", "first_news"]).dt.days.median()
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Trading days from the first public news")
    ax.set_ylabel("Mean cumulative abnormal return, %")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.5, ax.get_ylim()[0] * 0.95, "first news", fontsize=7, color="grey")
    fig.tight_layout()
    fig.savefig(ROOT / "paper/fig_paths.pdf")
    write("v_median_gap.tex", f"{med:.0f}")
    vals = {"gap": f"{med:.0f}", "nvshare": (OUT / "v_nvidia_share.tex").read_text().strip(),
            "nvsharesmh": (OUT / "v_nvidia_share_smh.tex").read_text().strip(),
            "coren": (OUT / "v_core_n.tex").read_text().strip()}
    write("values.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in vals.items()))


def main() -> None:
    d = pd.read_parquet(ROOT / "data/derived/personal_scandals.parquet")
    d = d.rename(columns={"index": "case"}) if "index" in d else d
    d = d[d["disclosure"].isin(ORDER)]
    table_period(); table_cars(d); table_robust(); table_coverage(); table_power(d); table_cases(d); table_nvidia(); table_sessions(); figure(d)
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
