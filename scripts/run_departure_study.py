"""Event study of CEO departures, 2000-2024: how much is priced in before, at and after the news, by reason.

Day 0 is the first trading day on which the news could move the close: the first RavenPack story naming the
CEO, else Gentry et al.'s announcement date, else the date the CEO left (scripts/build_event_sample.py).

Model: the market model on the Fama-French market return. The estimation window is 250 trading days
ending 141 days before day 0, i.e. [-390, -141], so that the run-up to a forced exit (poor performance
often lasts months) does not enter the benchmark. Because alpha estimated even that far back can carry
a firm's long decline, every CAR is also reported with alpha set to zero (`car_noalpha`).

Outputs (summary statistics only):
    results/departure_estimation_summary.csv  how many events ran, and why the rest did not
    results/departure_car_by_reason.csv       mean and median CARs, cross-sectional, BMP and sign tests
    results/departure_priced_in.csv           share of the mean move before the news (mean pre / mean (pre + ann))
                                              and mean post / mean ann, with bootstrap CIs
    results/departure_cross_section.csv       announcement CAR on reason, CEO power and firm power
Row-level CARs go to data/derived/departure_windows.parquet (gitignored).

    ~/.venvs/key-person-risk/bin/python scripts/run_departure_study.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from keyperson import crsp, measures
from keyperson.cross_section import summarise
from keyperson.engine import EstimationRule, EventStudyEngine, Window

ROOT = Path(__file__).resolve().parents[1]
STAGE3 = ROOT / "data/wrds/stage3"
WINDOWS = (
    Window("pre_120_2", -120, -2), Window("pre_60_2", -60, -2), Window("pre_20_2", -20, -2),
    Window("day_0", 0, 0), Window("ann_m1_1", -1, 1), Window("post_2_20", 2, 20), Window("post_2_60", 2, 60),
)
RULE = EstimationRule(length=250, gap=21, anchor=-120)
BOOT = 2000
SEED = 20261117


def run_engine(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dsf = crsp.add_delisting(crsp.load_dsf(STAGE3 / "dsf"), pd.read_parquet(STAGE3 / "dsedelist.parquet"))
    panel = crsp.returns_panel(dsf, pd.read_parquet(STAGE3 / "ff_factors_daily.parquet"))
    engine = EventStudyEngine(panel, crsp.MARKET, RULE, WINDOWS)
    events = pd.DataFrame({"event_id": ev["dismissal_dataset_id"].astype(str),
                           "ticker": ev["permno"].astype("int64").astype(str), "day0": ev["event_date"]})
    windows, estimation, _ = engine.run(events)
    windows["car_noalpha"] = windows["car"] + windows["alpha_part"]
    return windows, estimation


def by_reason(windows: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    w = windows[windows["complete"].astype(bool)].merge(ev[["event_id", "reason", "departure_code"]], on="event_id")
    rows = []
    for (code, reason, name), g in w.groupby(["departure_code", "reason", "window"]):
        row = {"departure_code": code, "reason": reason, "window": name}
        row |= summarise(g["car"], g["se"])
        row["mean_car_noalpha"] = g["car_noalpha"].mean()
        rows.append(row)
    order = {w.name: i for i, w in enumerate(WINDOWS)}
    return pd.DataFrame(rows).sort_values(["departure_code", "window"], key=lambda s: s.map(order) if s.name == "window" else s)


def wide_cars(windows: pd.DataFrame, value: str) -> pd.DataFrame:
    w = windows[windows["complete"].astype(bool)]
    return w.pivot_table(index="event_id", columns="window", values=value, aggfunc="first")


def priced_in(windows: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for value in ["car", "car_noalpha"]:
        wide = wide_cars(windows, value).join(ev.set_index("event_id")[["reason", "departure_code"]])
        for (code, reason), g in wide.groupby(["departure_code", "reason"]):
            for pre in ["pre_20_2", "pre_60_2", "pre_120_2"]:
                d = g[[pre, "ann_m1_1", "post_2_60"]].dropna()
                if len(d) < 10:
                    continue
                pre_v, ann_v, post_v = d[pre].to_numpy(), d["ann_m1_1"].to_numpy(), d["post_2_60"].to_numpy()
                share = measures.share_of_mean_move(pre_v, ann_v)
                post = measures.ratio_of_means(post_v, ann_v)
                idx = rng.integers(0, len(d), size=(BOOT, len(d)))
                bs = np.array([measures.share_of_mean_move(pre_v[i], ann_v[i]) for i in idx])
                bp = np.array([measures.ratio_of_means(post_v[i], ann_v[i]) for i in idx])
                sign_aligned = measures.aggregate_priced_in_share(pre_v, ann_v)
                rows.append({"value": value, "departure_code": code, "reason": reason, "pre_window": pre,
                             "n": len(d), "mean_pre": pre_v.mean(), "mean_ann": ann_v.mean(), "mean_post_2_60": post_v.mean(),
                             "priced_in_share": share, "priced_in_ci_low": np.nanpercentile(bs, 2.5),
                             "priced_in_ci_high": np.nanpercentile(bs, 97.5), "post_ratio": post,
                             "post_ratio_ci_low": np.nanpercentile(bp, 2.5), "post_ratio_ci_high": np.nanpercentile(bp, 97.5),
                             "sign_aligned_share_noise_prone": sign_aligned})
    return pd.DataFrame(rows)


def cross_section(windows: pd.DataFrame, ev: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    cars = wide_cars(windows, "car")[["pre_60_2", "ann_m1_1", "post_2_60"]]
    feats = features.assign(event_id=features["dismissal_dataset_id"].astype(str)).set_index("event_id")
    d = cars.join(feats, how="inner")
    d = d.assign(misconduct=(d["departure_code"] == 4).astype(float), performance=(d["departure_code"] == 3).astype(float),
                 death_illness=d["departure_code"].isin([1, 2]).astype(float),
                 own_pct_w=d["own_pct"].clip(upper=d["own_pct"].quantile(0.99)),
                 leaked=d["news_before_gentry"].fillna(0.0))
    for c in ["pre_60_2", "ann_m1_1", "post_2_60"]:
        lo, hi = d[c].quantile([0.005, 0.995])
        d[c] = d[c].clip(lo, hi)  # winsorise at 0.5% and 99.5%
    base = "misconduct + performance + death_illness"
    power = "duality + tenure_years + own_pct_w + pay_slice + log_mv + sales_share + prior_firms"
    specs = {
        "ann_reason_only": f"ann_m1_1 ~ {base} + C(year)",
        "ann_power": f"ann_m1_1 ~ {base} + {power} + C(year)",
        "ann_power_x_misconduct": (f"ann_m1_1 ~ {base} + {power} + misconduct:duality + misconduct:log_mv"
                                   " + misconduct:tenure_years + C(year)"),
        "pre_power": f"pre_60_2 ~ {base} + {power} + C(year)",
        "post_power": f"post_2_60 ~ {base} + {power} + C(year)",
    }
    rows = []
    for name, formula in specs.items():
        terms = [t.strip() for t in formula.split("~")[1].split("+") if not t.strip().startswith("C(")]
        needed = sorted({v for t in terms for v in t.split(":")} | {formula.split("~")[0].strip(), "year", "gvkey"})
        data = d.dropna(subset=needed)
        fit = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["gvkey"]})
        ci = fit.conf_int()
        for term in fit.params.index:
            if term.startswith("C(") or term == "Intercept":
                continue
            rows.append({"model": name, "term": term, "coef": fit.params[term], "se": fit.bse[term],
                         "p_value": fit.pvalues[term], "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1],
                         "n": int(fit.nobs), "r2": fit.rsquared})
    return pd.DataFrame(rows)


def main() -> None:
    ev = pd.read_parquet(ROOT / "data/derived/event_sample.parquet")
    ev = ev[ev["permno"].notna()].copy()
    ev["event_id"] = ev["dismissal_dataset_id"].astype(str)
    windows, estimation = run_engine(ev)
    windows.to_parquet(ROOT / "data/derived/departure_windows.parquet", index=False)

    status = estimation.merge(ev[["event_id", "reason"]], on="event_id").groupby(["reason", "status"]).size()
    status.rename("events").reset_index().to_csv(ROOT / "results/departure_estimation_summary.csv", index=False)
    t1 = by_reason(windows, ev)
    t1.to_csv(ROOT / "results/departure_car_by_reason.csv", index=False)
    t2 = priced_in(windows, ev)
    t2.to_csv(ROOT / "results/departure_priced_in.csv", index=False)
    t3 = cross_section(windows, ev, pd.read_parquet(ROOT / "data/derived/event_features.parquet"))
    t3.to_csv(ROOT / "results/departure_cross_section.csv", index=False)

    pd.set_option("display.width", 250)
    print(status.to_string())
    print(t1[["reason", "window", "n", "mean_car", "median_car", "t_cross_sectional", "p_bmp", "share_positive",
              "mean_car_noalpha"]].round(4).to_string(index=False))
    print(t2.round(3).to_string(index=False))
    print(t3.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
