"""CEO personal scandals, 2000-2024: one-stage versus two-stage disclosure, and what the market loses.

Sample (all hand-coded, see docs/scandal_codebook.md):
  * exits: Gentry et al. code-4 departures coded as personal conduct (events/gentry_misconduct_coding.csv);
  * RavenPack episodes coded P1 (events/ceo_scandal_coding.csv), which add the CEOs who stayed and the cases
    Gentry does not cover. A P1 episode of a CEO who left within a year is merged with that exit;
  * cases the headline search cannot find, added by hand (events/personal_scandal_additions.csv).
Personal conduct is conduct outside the CEO's job at the firm; the CEO's conduct in running the firm, including
trading in or statements about its securities, is coded W and left out.
First news: the earliest whole-word scandal story naming the sitting CEO in the year before the exit (stage-4
headlines) or the P1 episode's first story, whichever is earlier; else the exit date itself.
Disclosure: two-stage when the first news came at least two days before the exit (the Nvidia pattern),
one-stage when the scandal became public with the exit, survived when there was no exit within a year.

Returns: CRSP daily 1999-2024 (stage 4) with delisting returns; market model on the Fama-French market,
estimated over [-390, -141] trading days before the first news. Windows, in trading days, relative to the
first news (day 0) and, for two-stage cases, to the exit (day k):
    pre [-20, -2], first [-1, +1], interim [+2, k-2], exit [k-1, k+1], post [k+2, k+60] (k = 0 otherwise).

Outputs (summary statistics only): results/personal_scandal_summary.csv, results/personal_scandal_by_power.csv.
Row-level results: data/derived/personal_scandals.parquet.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from keyperson import crsp, power, sample, scandals
from keyperson.engine import EstimationRule, EventStudyEngine, Window

import build_scandal_candidates as bsc  # same folder

ROOT = Path(__file__).resolve().parents[1]
WRDS = ROOT / "data/wrds"
RULE = EstimationRule(length=250, gap=21, anchor=-120)
CRSP_END = pd.Timestamp("2024-12-31")


def effective(ts: pd.Series) -> pd.Series:
    ny = pd.to_datetime(ts).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    late = ny.dt.hour * 60 + ny.dt.minute >= 16 * 60
    return ny.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(late.astype(int), unit="D")


def assemble() -> pd.DataFrame:
    ev = pd.read_parquet(ROOT / "data/derived/event_sample.parquet")
    gcode = pd.read_csv(ROOT / "events/gentry_misconduct_coding.csv", dtype={"gvkey": str})
    exits = ev.merge(gcode[gcode["gentry_class"].eq("PERS")][["dismissal_dataset_id"]], on="dismissal_dataset_id")
    exits = exits[["gvkey", "co_per_rol", "fyear", "event_date", "surname", "rp_entity_id"]].rename(columns={"event_date": "exit_date"})
    exits["source"] = "gentry"
    exits["case"] = [f"G{c}-{d:%Y%m}" for c, d in zip(exits["co_per_rol"], exits["exit_date"])]
    # Hand checks (events/personal_scandal_overrides.csv): exit dates, first-news dates and duplicates.
    ov = pd.read_csv(ROOT / "events/personal_scandal_overrides.csv", parse_dates=["first_news"])
    fix_exit = ov[ov["action"].eq("exit_date")].set_index("case")["first_news"]
    hit = exits["case"].isin(fix_exit.index)
    exits.loc[hit, "exit_date"] = exits.loc[hit, "case"].map(fix_exit)

    cod = pd.read_csv(ROOT / "events/ceo_scandal_coding.csv", dtype={"gvkey": str, "execid": str}, parse_dates=["first_ts"])
    p1 = cod[cod["code"].eq("P1")].copy()
    cand = pd.concat([pd.read_csv(ROOT / f"data/derived/{f}", usecols=["candidate_id", "rp_entity_id"])
                      .assign(candidate_id=lambda x, f=f: x["candidate_id"].str.replace("C", "S") if "strict" in f else x["candidate_id"])
                      for f in ["scandal_candidates.csv", "scandal_candidates_strict.csv"]])
    p1 = p1.merge(cand.drop_duplicates("candidate_id"), on="candidate_id", how="left")
    p1["first_news_rp"] = effective(p1["first_ts"])

    # Stage-4 whole-word scandal stories naming the sitting CEO, to date allegations that preceded exits.
    cols = ["timestamp_utc", "rp_story_id", "rp_entity_id", "entity_name", "headline"]
    heads = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted((WRDS / "stage4/ravenpack").glob("dj_ceo_scandal_*.parquet"))])
    anncomp = pd.read_csv(WRDS / "execucomp_anncomp.csv", dtype={"gvkey": str, "execid": str})
    stories = scandals.dating_stories(scandals.candidate_stories(heads, bsc.entity_map(), scandals.sitting_ceos(anncomp)))
    stories = stories.assign(day=effective(stories["ts"]))[["co_per_rol", "day"]]

    rows = []
    used_p1 = set()
    for r in exits.itertuples():
        lo, hi = r.exit_date - pd.Timedelta(days=270), r.exit_date
        s = stories[(stories["co_per_rol"] == r.co_per_rol) & (stories["day"] >= lo) & (stories["day"] <= hi)]["day"]
        q = p1[(p1["co_per_rol"] == r.co_per_rol) & (p1["first_news_rp"] >= r.exit_date - pd.Timedelta(days=365))
               & (p1["first_news_rp"] <= hi + pd.Timedelta(days=3))]
        used_p1 |= set(q["candidate_id"])
        first = min([r.exit_date] + list(s) + list(q["first_news_rp"]))
        rows.append({"case": r.case, "gvkey": r.gvkey, "co_per_rol": r.co_per_rol,
                     "first_news": first, "exit_date": r.exit_date, "source": "gentry", "rp_entity_id": r.rp_entity_id})
    for r in p1[~p1["candidate_id"].isin(used_p1)].itertuples():
        # Only a Gentry code-4 exit counts as an exit over the scandal; a later exit for another stated reason
        # (performance, retirement) is kept apart.
        scandal_exit = pd.notna(r.days_to_exit) and r.exit_code == 4
        exit_date = r.first_news_rp + pd.Timedelta(days=int(r.days_to_exit)) if scandal_exit else pd.NaT
        rows.append({"case": r.candidate_id, "gvkey": r.gvkey, "co_per_rol": r.co_per_rol, "first_news": r.first_news_rp,
                     "rp_entity_id": r.rp_entity_id,
                     "exit_date": exit_date, "source": "ravenpack", "other_exit": pd.notna(r.days_to_exit) and not scandal_exit,
                     "outcome_known": pd.notna(r.days_to_exit) or bool(r.ceo_next_year)})
    # Cases the headline search cannot find (events/personal_scandal_additions.csv), dated from the news.
    add = pd.read_csv(ROOT / "events/personal_scandal_additions.csv", dtype={"gvkey": str, "rp_entity_id": str},
                      parse_dates=["first_news", "exit_date"])
    for r in add.itertuples():
        rows.append({"case": r.case, "gvkey": r.gvkey, "co_per_rol": r.co_per_rol, "first_news": r.first_news,
                     "exit_date": r.exit_date, "source": "added by hand", "rp_entity_id": r.rp_entity_id,
                     "outcome_known": True})
    cases = pd.DataFrame(rows)
    cases["outcome_known"] = cases["outcome_known"].fillna(True).astype(bool)
    # Outcome for RavenPack cases outside Gentry's coverage: the first RavenPack CEO-departure story naming the
    # CEO within a year of the first news (executive firing, resignation or retirement).
    deps = pd.concat([pd.read_parquet(p, columns=["timestamp_utc", "rp_entity_id", "type", "position_name", "headline"])
                      for p in sorted((WRDS / "ravenpack").glob("dj_exec_*.parquet"))])
    deps = deps[deps["type"].isin(["executive-firing", "executive-resignation", "executive-retirement"])
                & deps["position_name"].fillna("").str.contains("Chief Executive", case=False)]
    deps = deps.assign(day=effective(deps["timestamp_utc"]))
    ent = bsc.entity_map().drop_duplicates("gvkey").set_index("gvkey")["rp_entity_id"]
    names = anncomp.drop_duplicates("co_per_rol").set_index("co_per_rol")["exec_fullname"].map(sample.surname)
    for i in cases.index[~cases["outcome_known"]]:
        c = cases.loc[i]
        d = deps[(deps["rp_entity_id"] == ent.get(c["gvkey"])) & (deps["day"] >= c["first_news"])
                 & (deps["day"] <= c["first_news"] + pd.Timedelta(days=365))]
        d = d[d["headline"].fillna("").str.contains(r"\b" + re.escape(str(names.get(c["co_per_rol"], "@@"))) + r"\b", case=False)]
        if len(d):
            cases.loc[i, ["exit_date", "outcome_known"]] = [d["day"].min(), True]
            cases.loc[i, "source"] = "ravenpack, exit from RavenPack"
    cases = cases[~cases["case"].isin(ov.loc[ov["action"].eq("drop"), "case"])].copy()
    fix = ov[ov["action"].eq("first_news")].set_index("case")["first_news"]
    hit = cases["case"].isin(fix.index)
    cases.loc[hit, "first_news"] = cases.loc[hit, "case"].map(fix)
    cases["disclosure"] = [scandals.classify_disclosure(f, e) for f, e in zip(cases["first_news"], cases["exit_date"])]
    cases.loc[~cases["outcome_known"], "disclosure"] = "outcome unknown"
    cases.loc[cases["other_exit"].fillna(False).astype(bool), "disclosure"] = "left later, other reason"
    cases["fyear"] = cases["first_news"].dt.year
    cases["ref_date"] = cases["first_news"]
    cases = sample.link_permno(cases, pd.read_csv(WRDS / "ccm_link_table.csv", dtype={"gvkey": str}))
    cases = power.ceo_power(cases, anncomp)
    funda = pd.read_csv(WRDS / "comp_funda_annual.csv", dtype={"gvkey": str}, usecols=["gvkey", "fyear", "at", "sale", "csho", "prcc_f", "sich"])
    cases = power.firm_power(cases, funda)
    founders = set(anncomp[anncomp["title"].fillna("").str.contains("founder", case=False)]["co_per_rol"])
    gentry = pd.read_excel(ROOT / "data/external/gentry/CEO Dismissal Database Posted to Web 10Oct24.xlsx",
                           usecols=["co_per_rol", "notes"])
    founders |= set(gentry[gentry["notes"].fillna("").str.contains("founder", case=False)]["co_per_rol"])
    cases["founder"] = cases["co_per_rol"].isin(founders).astype(float)
    manual = pd.read_csv(ROOT / "events/founder_manual.csv").set_index("case")["founder"]
    hit = cases["case"].isin(manual.index)
    cases.loc[hit, "founder"] = cases.loc[hit, "case"].map(manual).astype(float)
    cases["founder_source"] = np.where(hit, "hand check", "ExecuComp title or Gentry notes")
    return cases


def event_windows(panel: pd.DataFrame, case) -> list[dict]:
    col = str(int(case.permno))
    if col not in panel:
        return []
    days = panel.index[panel[col].notna()]
    p0 = days.searchsorted(case.first_news)
    if p0 >= len(days):
        return []
    k = 0
    if case.disclosure == "two-stage":
        k = int(days.searchsorted(case.exit_date) - p0)
    windows = [Window("pre", -20, -2), Window("first", -1, 1), Window("first_day0", 0, 0)]
    if k >= 4:
        windows += [Window("interim", 2, k - 2), Window("exit", k - 1, k + 1)]
    elif k > 1:
        windows += [Window("exit", 2, k + 1)]
    windows.append(Window("post", max(k, 0) + 2, max(k, 0) + 60))
    res = EventStudyEngine(panel, crsp.MARKET, RULE, windows).run_event(case.case, col, case.first_news)
    if res.status != "ok":
        return []
    out = res.windows.copy()
    beta = res.fit.beta_market
    # Main: market model with alpha set to zero (an alpha estimated a year earlier would otherwise be projected
    # over long windows). Robustness: the full market model, and market-adjusted returns (r - m).
    out["car_full"] = out["car"]
    out["car_main"] = out["car"] + out["alpha_part"]
    out["car_mktadj"] = out["sum_of_returns"] - out["market_part"] / beta if abs(beta) > 1e-6 else np.nan
    est = panel[col].loc[res.fit.residual_dates[0]:res.fit.residual_dates[-1]]
    out["alpha_daily"] = res.fit.alpha
    out["max_abs_return_estimation"] = float(est.abs().max())
    out["k"] = k
    return out.to_dict("records")


WINDOWS_SUM = ["pre", "first", "interim", "exit"]
VARIANTS = {"main": "car_main", "full": "car_full", "mktadj": "car_mktadj"}
BORDERLINE_SURVIVORS = {"C0253": "a divorce and the CEO as the target of attempted blackmail; no misconduct alleged"}


def entity_at_news(cases: pd.DataFrame) -> pd.Series:
    """RavenPack entity of each case's firm on the first-news date, through the CRSP CUSIP of that date. The
    Compustat link uses the current CUSIP, which points to a successor entity when a firm has since changed
    its name or merged (CBS to Paramount Skydance, WWE to TKO)."""
    names = pd.read_parquet(WRDS / "stage4/dsenames.parquet")
    names["namedt"], names["nameendt"] = pd.to_datetime(names["namedt"]), pd.to_datetime(names["nameendt"])
    rp = pd.read_parquet(WRDS / "ravenpack/company_mappings_hist.parquet").dropna(subset=["cusip"])
    rp["start"] = pd.to_datetime(rp["range_start"], errors="coerce")
    rp["end"] = pd.to_datetime(rp["range_end"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    rp["cusip8"] = rp["cusip"].str[:8]
    out = {}
    for c in cases.itertuples():
        n = names[(names["permno"] == c.permno) & (names["namedt"] <= c.first_news) & (names["nameendt"] >= c.first_news)]
        if n.empty:
            continue
        m = rp[(rp["cusip8"] == n["ncusip"].iloc[0]) & (rp["start"] <= c.first_news) & (rp["end"] >= c.first_news)]
        if len(m):
            out[c.Index] = m["rp_entity_id"].iloc[0]
    return pd.Series(out, dtype=object)


def coverage(cases: pd.DataFrame) -> pd.Series:
    """Dow Jones stories about the firm in RavenPack's stage-3 and executive-event pulls, from 365 to 30 days
    before the first news: a proxy for how closely the press follows the firm."""
    cols = ["rp_entity_id", "rp_story_id", "timestamp_utc"]
    parts = [pd.read_parquet(f, columns=cols) for f in sorted((WRDS / "stage3/ravenpack").glob("dj_channels_*.parquet"))]
    parts += [pd.read_parquet(f, columns=cols) for f in sorted((WRDS / "ravenpack").glob("dj_exec_*.parquet"))]
    news = pd.concat(parts).drop_duplicates(["rp_entity_id", "rp_story_id"])
    news["ts"] = pd.to_datetime(news["timestamp_utc"])
    covered = set(pd.concat(parts[: len(parts) - 27])["rp_entity_id"]) if len(parts) > 27 else set()
    by = {k: v["ts"].to_numpy() for k, v in news.groupby("rp_entity_id")}
    out = {}
    for c in cases.itertuples():
        e = c.rp_entity_id
        if e not in covered:
            out[c.Index] = np.nan
            continue
        t = by.get(e, np.array([], dtype="datetime64[ns]"))
        lo, hi = np.datetime64(c.first_news - pd.Timedelta(days=365)), np.datetime64(c.first_news - pd.Timedelta(days=30))
        out[c.Index] = float(((t >= lo) & (t < hi)).sum())
    return pd.Series(out)


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def main() -> None:
    cases = assemble()
    cases = cases[(cases["first_news"] <= CRSP_END - pd.Timedelta(days=10)) & cases["permno"].notna()].copy()
    dsf = crsp.add_delisting(crsp.load_dsf(WRDS / "stage4/dsf"), pd.read_parquet(WRDS / "stage4/dsedelist.parquet"))
    panel = crsp.returns_panel(dsf, pd.read_parquet(WRDS / "stage3/ff_factors_daily.parquet"))
    w = pd.DataFrame([r for c in cases.itertuples() for r in event_windows(panel, c)])
    d = cases.set_index("case")
    for v, col in VARIANTS.items():
        car = w.pivot_table(index="event_id", columns="window", values=col, aggfunc="first")
        car.columns = [f"{v}_{c}" for c in car.columns]
        d = d.join(car, how="inner" if v == "main" else "left")
        for c in ["interim", "exit"]:
            if f"{v}_{c}" not in d:
                d[f"{v}_{c}"] = np.nan
        d[f"{v}_total"] = d[[f"{v}_{c}" for c in WINDOWS_SUM]].sum(axis=1, min_count=2)
        d[f"{v}_before_exit"] = d[[f"{v}_{c}" for c in ["pre", "first", "interim"]]].sum(axis=1, min_count=2)
    est = w.drop_duplicates("event_id").set_index("event_id")[["alpha_daily", "max_abs_return_estimation"]]
    d = d.join(est)
    d["flag_alpha"] = d["alpha_daily"].abs() > 0.002
    d["flag_jump"] = d["max_abs_return_estimation"] > 0.5
    d["rp_entity_id"] = entity_at_news(d).reindex(d.index).fillna(d["rp_entity_id"])
    d["coverage"] = coverage(d)
    d.index.name = "case"
    d.reset_index().to_parquet(ROOT / "data/derived/personal_scandals.parquet", index=False)

    groups = ["one-stage", "two-stage", "survived"]
    rows = []
    for k in groups:
        g = d[d["disclosure"] == k]
        for v in VARIANTS:
            row = {"disclosure": k, "variant": v, "cases": len(g)}
            for c in ["pre", "first", "first_day0", "interim", "exit", "total", "post"]:
                x = g[f"{v}_{c}"].dropna()
                row[f"mean_{c}"] = x.mean() if len(x) else np.nan
                row[f"median_{c}"] = x.median() if len(x) else np.nan
                row[f"t_{c}"] = tstat(x)
                row[f"negative_{c}"] = int((x < 0).sum())
                row[f"n_{c}"] = len(x)
            if k == "two-stage":
                row["share_before_exit"] = g[f"{v}_before_exit"].mean() / g[f"{v}_total"].mean()
                row["median_days_first_news_to_exit"] = (g["exit_date"] - g["first_news"]).dt.days.median()
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "results/personal_scandal_summary.csv", index=False)

    # Leave-one-out ranges, and the survivors without the two borderline cases
    rob = []
    for k in ["two-stage", "survived"]:
        g = d[d["disclosure"] == k]
        for c in ["first", "first_day0", "total", "post"]:
            loo = [g.drop(i)[f"main_{c}"].mean() for i in g.index]
            rob.append({"disclosure": k, "sample": "all", "window": c, "cases": len(g), "mean": g[f"main_{c}"].mean(),
                        "loo_min": min(loo), "loo_max": max(loo), "negative": int((g[f"main_{c}"] < 0).sum())})
    core = d[(d["disclosure"] == "survived") & ~d.index.isin(BORDERLINE_SURVIVORS)]
    for c in ["first", "first_day0", "total", "post"]:
        rob.append({"disclosure": "survived", "sample": "core (without " + ", ".join(BORDERLINE_SURVIVORS) + ")",
                    "window": c, "cases": len(core), "mean": core[f"main_{c}"].mean(), "loo_min": np.nan,
                    "loo_max": np.nan, "negative": int((core[f"main_{c}"] < 0).sum())})
    pd.DataFrame(rob).to_csv(ROOT / "results/personal_scandal_robustness.csv", index=False)

    flags = d[d["flag_alpha"] | d["flag_jump"]][["disclosure", "alpha_daily", "max_abs_return_estimation",
                                                 "full_post", "main_post", "mktadj_post"]]
    flags.reset_index().to_csv(ROOT / "results/personal_scandal_flags.csv", index=False)

    d["size_tercile"] = pd.qcut(d["log_mv"], 3, labels=["small", "middle", "large"])
    d["public"] = d["disclosure"].isin(["two-stage", "survived"])
    cov = d[d["disclosure"].isin(groups)].groupby(["size_tercile", "public"], observed=True).agg(
        cases=("coverage", "size"), coverage_median=("coverage", "median"), coverage_mean=("coverage", "mean"),
        founder_share=("founder", "mean")).reset_index()
    cov.to_csv(ROOT / "results/personal_scandal_coverage.csv", index=False)

    rows = []
    for k in groups:
        g = d[d["disclosure"] == k]
        rows.append({"disclosure": k, "cases": len(g), "founder_share": g["founder"].mean(),
                     "duality_share": g["duality"].mean(), "tenure_median": g["tenure_years"].median(),
                     "market_value_median_bn": (np.exp(g["log_mv"]) / 1000).median(),
                     "coverage_median": g["coverage"].median(), "coverage_available": int(g["coverage"].notna().sum())})
    pd.DataFrame(rows).to_csv(ROOT / "results/personal_scandal_by_power.csv", index=False)

    pd.set_option("display.width", 250)
    print(cases["disclosure"].value_counts().to_string())
    print(summary[summary["variant"] == "main"].set_index("disclosure").round(4).T.to_string())
    print(summary.pivot(index="disclosure", columns="variant", values=["mean_first", "mean_total", "mean_post"]).round(4))
    print(pd.DataFrame(rob).round(4).to_string(index=False))
    print(flags.round(4).to_string())
    print(cov.round(2).to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
