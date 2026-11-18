"""Sanity checks on the departure event study: do the results behave as economics and common sense predict?

Each check states its prediction; the script prints the evidence and writes summary tables only:
    archive/results/departure_checks.csv          one row per check statistic
    archive/results/departure_daily_profile.csv   mean abnormal return by trading day -10..+10, by reason
    ~/.venvs/key-person-risk/bin/python scripts/check_departure_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from keyperson import crsp
from keyperson.engine import EventStudyEngine, Window

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_departure_study as study  # noqa: E402  (scripts/)

KEY = ["dismissed_performance", "dismissed_misconduct", "new_opportunity", "retired"]


def main() -> None:
    pd.set_option("display.width", 220)
    rows = []
    def note(check, group, stat, value, n=None):
        rows.append({"check": check, "group": group, "statistic": stat, "value": value, "n": n})

    ev = pd.read_parquet(ROOT / "data/derived/event_sample.parquet")
    ev = ev[ev["permno"].notna()].copy()
    ev["event_id"] = ev["dismissal_dataset_id"].astype(str)
    feats = pd.read_parquet(ROOT / "data/derived/event_features.parquet").assign(
        event_id=lambda d: d["dismissal_dataset_id"].astype(str))
    w = pd.read_parquet(ROOT / "data/derived/departure_windows.parquet")
    w = w[w["complete"].astype(bool)]
    wide = w.pivot_table(index="event_id", columns="window", values="car", aggfunc="first")
    d = wide.join(ev.set_index("event_id")[["reason", "date_source", "event_date"]]).join(
        feats.set_index("event_id")[["log_mv", "gvkey"]])

    # 1. Data: returns must lie above -100%; extreme values should be rare
    dsf = crsp.add_delisting(crsp.load_dsf(study.STAGE3 / "dsf"), pd.read_parquet(study.STAGE3 / "dsedelist.parquet"))
    note("1 data", "all", "min daily return", dsf["ret"].min(), len(dsf))
    note("1 data", "all", "share of |ret| > 50%", (dsf["ret"].abs() > 0.5).mean(), len(dsf))

    # 2. Model: betas near 1 on average, R2 of 0.1-0.4 for single stocks; negative alpha before forced exits
    est_path = ROOT / "data/derived/departure_estimation.parquet"
    panel = crsp.returns_panel(dsf, pd.read_parquet(study.STAGE3 / "ff_factors_daily.parquet"))
    days = tuple(Window(f"d{k:+d}", k, k) for k in range(-10, 11))
    engine = EventStudyEngine(panel, crsp.MARKET, study.RULE, days + (Window("anchor", -120, -120),))
    events = pd.DataFrame({"event_id": ev["event_id"], "ticker": ev["permno"].astype("int64").astype(str),
                           "day0": ev["event_date"]})
    daily, est, _ = engine.run(events)
    est = est[est["status"] == "ok"].merge(ev[["event_id", "reason"]], on="event_id")
    est.to_parquet(est_path, index=False)
    g = est.groupby("reason")
    for reason, s in g:
        note("2 model", reason, "mean beta", s["beta_market"].mean(), len(s))
        note("2 model", reason, "median R2", s["r_squared"].median(), len(s))
        note("2 model", reason, "mean alpha, % a year", s["alpha_daily"].mean() * 252 * 100, len(s))
    print(g[["beta_market", "r_squared", "alpha_daily"]].agg(["mean", "median"]).round(4).to_string())

    # 3. Timing: if day 0 is right, the reaction should spike on days 0 and +1, not spread over the window
    daily = daily[daily["window"] != "anchor"].merge(ev[["event_id", "reason"]], on="event_id")
    prof = daily.groupby(["reason", "rel_start"])["car"].agg(["mean", "count", "std"]).reset_index()
    prof["t"] = prof["mean"] / (prof["std"] / np.sqrt(prof["count"]))
    prof.rename(columns={"rel_start": "day"}).to_csv(ROOT / "archive/results/departure_daily_profile.csv", index=False)
    print(prof[prof["reason"].isin(KEY)].pivot(index="rel_start", columns="reason", values="mean").mul(100).round(2).to_string())

    # 4. Dating source: events dated only by the effective exit date should show weaker announcement effects
    for (reason, src), s in d[d["reason"].isin(KEY)].groupby(["reason", "date_source"]):
        x = s["ann_m1_1"].dropna()
        note("4 date source", f"{reason} | {src}", "mean ann CAR", x.mean(), len(x))
        note("4 date source", f"{reason} | {src}", "t", x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan, len(x))

    # 5. Robustness of the headline numbers: medians, winsorised means, no 2008-09, no small caps
    small = d["log_mv"] < d["log_mv"].quantile(0.25)
    crisis = d["event_date"].dt.year.isin([2008, 2009])
    for reason, s in d[d["reason"].isin(KEY)].groupby("reason"):
        x = s["ann_m1_1"].dropna()
        lo, hi = d["ann_m1_1"].quantile([0.01, 0.99])
        note("5 robust", reason, "median ann CAR", x.median(), len(x))
        note("5 robust", reason, "1%-winsorised mean ann CAR", x.clip(lo, hi).mean(), len(x))
        note("5 robust", reason, "mean ann CAR, excl. 2008-09", s.loc[~crisis.reindex(s.index), "ann_m1_1"].mean(), int((~crisis.reindex(s.index)).sum()))
        note("5 robust", reason, "mean ann CAR, excl. smallest quarter", s.loc[~small.reindex(s.index), "ann_m1_1"].mean(), int((~small.reindex(s.index)).sum()))

    # 6. Anticipation: the worse the run-up, the less there is left to learn at the announcement
    for reason in ["dismissed_performance", "dismissed_misconduct"]:
        s = d[d["reason"] == reason].dropna(subset=["ann_m1_1", "pre_120_2"])
        s = s.assign(pre=s["pre_120_2"].clip(*s["pre_120_2"].quantile([0.01, 0.99])),
                     ann=s["ann_m1_1"].clip(*s["ann_m1_1"].quantile([0.01, 0.99])))
        fit = smf.ols("ann ~ pre", data=s).fit(cov_type="cluster", cov_kwds={"groups": s["gvkey"]})
        note("6 anticipation", reason, "slope of ann CAR on [-120,-2] CAR", fit.params["pre"], int(fit.nobs))
        note("6 anticipation", reason, "p", fit.pvalues["pre"], int(fit.nobs))
        q = pd.qcut(s["pre"], 3, labels=["worst third", "middle", "best third"])
        for k, v in s.groupby(q, observed=True)["ann"]:
            note("6 anticipation", reason, f"mean ann CAR, run-up {k}", v.mean(), len(v))

    # 7. Priced-in share on well-dated events only (news-based dates: RavenPack or Gentry's announcement).
    # Events dated by the day the CEO left can have the real announcement inside the pre-event window,
    # which inflates the share priced in. Both alpha conventions, because forced exits follow long declines.
    from keyperson import measures
    rng = np.random.default_rng(20261117)
    for value in ["car", "car_noalpha"]:
        wv = w.pivot_table(index="event_id", columns="window", values=value, aggfunc="first").join(
            ev.set_index("event_id")[["reason", "date_source"]])
        for reason in ["dismissed_performance", "dismissed_misconduct", "new_opportunity"]:
            for sample, keep in [("news-dated", ["ravenpack", "gentry_announced"]), ("left-office-dated", ["left_office"])]:
                s = wv[(wv["reason"] == reason) & wv["date_source"].isin(keep)]
                for pre in ["pre_20_2", "pre_60_2"]:
                    x = s[[pre, "ann_m1_1"]].dropna()
                    a, b = x[pre].to_numpy(), x["ann_m1_1"].to_numpy()
                    share = measures.share_of_mean_move(a, b)
                    idx = rng.integers(0, len(x), size=(2000, len(x)))
                    bs = np.array([measures.share_of_mean_move(a[i], b[i]) for i in idx])
                    grp = f"{reason} | {sample} | {value}"
                    note("7 priced in, by dating", grp, f"mean {pre}", a.mean(), len(x))
                    note("7 priced in, by dating", grp, "mean ann_m1_1", b.mean(), len(x))
                    note("7 priced in, by dating", grp, f"share before, {pre}", share, len(x))
                    note("7 priced in, by dating", grp, f"share CI low, {pre}", np.nanpercentile(bs, 2.5), len(x))
                    note("7 priced in, by dating", grp, f"share CI high, {pre}", np.nanpercentile(bs, 97.5), len(x))

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "archive/results/departure_checks.csv", index=False)
    print(out.assign(value=out["value"].round(4)).to_string(index=False))


if __name__ == "__main__":
    main()
