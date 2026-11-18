"""Exploratory check, not used in the paper: are Nvidia's hourly abnormal returns more fat-tailed in thin
trading sessions and during the episode, as the herding model of Cont and Bouchaud (arXiv cond-mat/9712318)
suggests? In that model excess kurtosis falls with the number of orders and rises with herding.

Hourly bars with extended hours from Yahoo Finance (data/yahoo_sessions/, 3 August to 17 November 2026).
Bar return: log close-to-close between consecutive bars of the same day; the first bar of a day uses its own
open, so overnight gaps are left out. Abnormal: Nvidia minus its daily weights on SPY and on the equal-weighted
basket of the six other mega-caps (nvidia_sessions.betas). Sessions by bar start: pre 04:00-09:00, regular
09:30-15:30, after 16:00-19:00. "News bars" are the bars that contain the time of an item in
events/nvidia_timeline_sources.csv; they are dropped in the second set of rows. Extended-hours volume is not
reported by Yahoo, so the session stands in for order flow.

Writes summary statistics only, to archive/results/kurtosis_by_session.csv.

    ~/.venvs/key-person-risk/bin/python archive/scripts/kurtosis_by_session.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import nvidia_sessions as ns  # noqa: E402

EPISODE = ("2026-10-29", "2026-11-17")
NEWS_BARS = ["2026-11-05 16:00", "2026-10-29 05:00", "2026-11-11 16:00", "2026-11-11 17:00", "2026-11-12 08:00",
             "2026-11-16 12:30", "2026-11-17 10:30", "2026-11-17 16:00", "2026-11-17 17:00"]
rng = np.random.default_rng(20261118)


def bar_returns(symbol: str) -> pd.Series:
    h = pd.read_parquet(ROOT / f"data/yahoo_sessions/{symbol}_1h.parquet").dropna(subset=["open", "close"])
    h.index = pd.DatetimeIndex(h.index).tz_convert("America/New_York").tz_localize(None)
    day = h.index.normalize()
    prev = h["close"].shift(1).where(pd.Series(day, index=h.index).eq(pd.Series(day, index=h.index).shift(1)))
    return np.log(h["close"] / prev.fillna(h["open"]))


def session(ts: pd.DatetimeIndex) -> np.ndarray:
    hm = ts.hour * 100 + ts.minute
    return np.select([hm < 930, hm < 1600], ["pre", "regular"], "after")


def stats(x: pd.Series) -> dict:
    x = x.dropna()
    boot = [pd.Series(rng.choice(x.to_numpy(), len(x))).kurt() for _ in range(2000)]
    return {"bars": len(x), "sd_bp": 1e4 * x.std(), "excess_kurtosis": x.kurt(),
            "kurtosis_p05": np.percentile(boot, 5), "kurtosis_p95": np.percentile(boot, 95),
            "max_abs_in_sd": (x.abs() / x.std()).max()}


def main() -> None:
    b_spy, b_basket = ns.betas()
    r = pd.DataFrame({s: bar_returns(s) for s in ["NVDA", "SPY"] + ns.PEERS}).dropna()
    r["basket"] = r[ns.PEERS].mean(axis=1)
    r["abnormal"] = r["NVDA"] - b_spy * r["SPY"] - b_basket * r["basket"]
    r["session"] = session(r.index)
    r["period"] = np.where((r.index >= EPISODE[0]) & (r.index < pd.Timestamp(EPISODE[1]) + pd.Timedelta(days=1)),
                           "episode", np.where(r.index < EPISODE[0], "before", "after episode"))
    r["news"] = r.index.isin(pd.to_datetime(NEWS_BARS))
    rows = []
    for drop_news in [False, True]:
        for period in ["before", "episode"]:
            for s in ["pre", "regular", "after"]:
                g = r[(r["period"] == period) & (r["session"] == s) & ~(r["news"] & drop_news)]
                for series, label in [("abnormal", "Nvidia, abnormal"), ("basket", "peer basket")]:
                    rows.append({"news_bars": "dropped" if drop_news else "kept", "period": period, "session": s,
                                 "series": label, **stats(g[series])})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "archive/results/kurtosis_by_session.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
