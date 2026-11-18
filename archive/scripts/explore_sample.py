"""Exploratory tables on the CEO-departure sample (archived; not used in the paper).

Uses build() from scripts/build_event_features.py and writes:
    archive/results/explore_by_reason.csv     means by departure reason
    archive/results/explore_by_year.csv       departures per year by reason
    archive/results/explore_regressions.csv   two linear-probability models with firm-clustered errors

    ~/.venvs/key-person-risk/bin/python archive/scripts/explore_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from build_event_features import MEASURES, build  # noqa: E402


def by_reason(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby(["departure_code", "reason"])
    out = g[MEASURES].mean().round(3)
    out.insert(0, "departures", g.size())
    out["n_follow_up"] = g["later_exec_5y"].count()
    return out.reset_index()


def regressions(ev: pd.DataFrame) -> pd.DataFrame:
    d = ev.assign(misconduct=(ev["departure_code"] == 4).astype(float),
                  performance=(ev["departure_code"] == 3).astype(float),
                  dismissed=ev["departure_code"].isin([3, 4]).astype(float),
                  own_pct_w=ev["own_pct"].clip(upper=ev["own_pct"].quantile(0.99)))
    specs = {
        "dismissed_vs_voluntary": ("dismissed ~ tenure_years + age + own_pct_w + duality + pay_slice + log_mv"
                                   " + sales_share + prior_firms + C(year)", d[d["departure_code"].isin([3, 4, 5, 6])]),
        "later_exec_5y": ("later_exec_5y ~ misconduct + performance + tenure_years + age + own_pct_w + duality"
                          " + pay_slice + log_mv + prior_firms + C(year)", d[d["later_exec_5y"].notna()]),
    }
    rows = []
    for name, (formula, data) in specs.items():
        cols = [c.strip() for c in formula.replace("~", "+").split("+") if not c.strip().startswith("C(")]
        data = data.dropna(subset=cols + ["year"])
        fit = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["gvkey"]})
        for term in fit.params.index:
            if term.startswith("C(") or term == "Intercept":
                continue
            ci = fit.conf_int().loc[term]
            rows.append({"model": name, "term": term, "coef": fit.params[term], "se": fit.bse[term],
                         "p_value": fit.pvalues[term], "ci_low": ci[0], "ci_high": ci[1],
                         "n": int(fit.nobs), "mean_dependent": data[formula.split("~")[0].strip()].mean(),
                         "r2": fit.rsquared})
    return pd.DataFrame(rows)


def main() -> None:
    ev = build()
    out = ROOT / "archive/results"
    t1 = by_reason(ev)
    t1.to_csv(out / "explore_by_reason.csv", index=False)
    t2 = ev.pivot_table(index="year", columns="reason", values="gvkey", aggfunc="count", fill_value=0)
    t2.reset_index().to_csv(out / "explore_by_year.csv", index=False)
    t3 = regressions(ev)
    t3.to_csv(out / "explore_regressions.csv", index=False)
    pd.set_option("display.width", 250)
    print(t1.T.to_string()); print(t2.to_string()); print(t3.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
