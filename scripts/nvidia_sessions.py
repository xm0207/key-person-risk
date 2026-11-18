"""Nvidia, October-November 2026: in which trading session, and in which period, was information priced?

Each trading day t is split into three sessions, in log returns that add up to the day's total:
    pre-market  from the last after-hours price of day t-1 to the official open of day t
    regular     from the official open to the official close of day t
    after-hours from the official close to the last after-hours price of day t (no trade: zero)
Official open and close come from Yahoo daily bars; after-hours prices from Yahoo hourly bars with extended
hours (Yahoo reports no volume outside regular hours, so volume is compared by day only). Company-specific
session returns use the benchmark of the paper's Nvidia section: Nvidia's session return minus
beta_m x SPY's and beta_b x the equal-weighted return of the six other mega-caps (Apple, Microsoft, Alphabet,
Amazon, Meta, Tesla) in the same session, with the betas estimated on daily returns over 250 trading days
ending 81 trading days before 29 October 2026 (as in nvidia_timeline.py). The intercept is left out.

Windows, in trading days relative to 29 October 2026 (the first report):
    two months before [-42, -22]; one month before [-21, -1]; the episode [0, last day in the data].

Measures, per window, session and series (Nvidia, the peer basket, Nvidia minus the basket):
    weighted price contribution (Barclay and Warner 1993; Barclay and Hendershott 2003):
        WPC_s = sum_t sign(r_t) r_{s,t} / sum_t |r_t|, the share of the period's price discovery in session s;
    share of squared returns: sum_t r_{s,t}^2 / sum_t sum_s r_{s,t}^2;
    mean absolute return (basis points);
    and, per window, Nvidia's mean daily volume relative to two months before.
Key days: Nvidia's company-specific return in each session on the days with news.

Prices are for personal use: bars are cached in data/yahoo_sessions/ (not committed), and only the summary
tables above are written, to results/nvidia_sessions.csv and results/nvidia_session_days.csv.

    ~/.venvs/key-person-risk/bin/python scripts/nvidia_sessions.py [--refresh]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from keyperson.config import DATA_END

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/yahoo_sessions"
PEERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
DAY0 = pd.Timestamp("2026-10-29")
START = "2026-08-01"
WINDOWS = {"two months before": (-42, -22), "one month before": (-21, -1), "episode": (0, 10_000)}
SESSIONS = ["pre", "regular", "after"]
KEY_DAYS = {
    "2026-10-29": "first newspaper report (05:01)",
    "2026-11-02": "Monday after a Saturday report on officials' concerns",
    "2026-11-05": "House letter (story after the close); broad market fall",
    "2026-11-09": "Monday after the board's investigation and the CEO's email (Sunday)",
    "2026-11-11": "reports of a relationship with a direct report (after the close)",
    "2026-11-12": "resignation 8-K (08:52, before the open)",
    "2026-11-16": "report that the board will not take him back (12:38)",
    "2026-11-17": "offer to return (10:58); results (16:20) and call",
    "2026-11-18": "first session after the results",
}
HEADERS = {"User-Agent": "keyperson-research-script/0.1"}


def fetch(symbol: str, interval: str, refresh: bool) -> pd.DataFrame:
    path = CACHE / f"{symbol}_{interval}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    end = int((pd.Timestamp(DATA_END) + pd.Timedelta(1, unit="D")).tz_localize("America/New_York").timestamp())
    beg = int(pd.Timestamp(START).tz_localize("America/New_York").timestamp())
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", headers=HEADERS, timeout=60,
                     params={"interval": interval, "period1": beg, "period2": end, "includePrePost": "true"})
    r.raise_for_status()
    j = r.json()["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ["open", "close", "volume"]},
                      index=pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert("America/New_York"))
    df.index.name = "ts"
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def sessions(symbol: str, refresh: bool) -> pd.DataFrame:
    """Daily log returns by session for one symbol, indexed by trading date."""
    d = fetch(symbol, "1d", refresh).dropna(subset=["open", "close"])
    d.index = pd.to_datetime(d.index.date)
    d = d[d.index <= pd.Timestamp(DATA_END)]
    h = fetch(symbol, "1h", refresh).dropna(subset=["close"])
    post = h[(h.index.hour >= 16) & (h.index.hour < 20)]
    last_after = post.groupby(post.index.date)["close"].last()
    last_after.index = pd.to_datetime(last_after.index)
    d["after_last"] = last_after.reindex(d.index).fillna(d["close"])
    out = pd.DataFrame(index=d.index)
    out["pre"] = np.log(d["open"] / d["after_last"].shift(1))
    out["regular"] = np.log(d["close"] / d["open"])
    out["after"] = np.log(d["after_last"] / d["close"])
    out["volume"] = d["volume"]
    return out.iloc[1:]


def betas() -> tuple[float, float]:
    """Nvidia's daily betas on SPY and the peer basket over the estimation window of nvidia_timeline.py."""
    from keyperson.config import FIRMS, MIN_BASKET_MEMBERS
    from keyperson.data import ex_firm_basket, load_returns
    r = load_returns(ROOT / "data/keyperson.duckdb")
    r = r.assign(basket=ex_firm_basket(r, FIRMS, "NVDA", MIN_BASKET_MEMBERS))[["NVDA", "SPY", "basket"]].dropna()
    end = r.index.searchsorted(DAY0) - 81
    est = r.iloc[end - 250:end]
    x = np.column_stack([np.ones(len(est)), est["SPY"], est["basket"]])
    coef = np.linalg.lstsq(x, est["NVDA"].to_numpy(), rcond=None)[0]
    print(f"estimation {est.index[0].date()} to {est.index[-1].date()}: beta_SPY {coef[1]:.3f}, beta_basket {coef[2]:.3f}")
    return float(coef[1]), float(coef[2])


def measures(r: pd.DataFrame) -> dict:
    total = r[SESSIONS].sum(axis=1)
    sq = (r[SESSIONS] ** 2).sum()
    res = {}
    for s in SESSIONS:
        res[f"wpc_{s}"] = float((np.sign(total) * r[s]).sum() / total.abs().sum())
        res[f"sq_share_{s}"] = float(sq[s] / sq.sum())
        res[f"mean_abs_bp_{s}"] = float(1e4 * r[s].abs().mean())
    res["mean_abs_bp_day"] = float(1e4 * total.abs().mean())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="download the bars again")
    args = ap.parse_args()
    nv = sessions("NVDA", args.refresh)
    peers = {p: sessions(p, args.refresh) for p in PEERS}
    days = nv.index
    basket = pd.concat([peers[p][SESSIONS].reindex(days) for p in PEERS]).groupby(level=0).mean()
    spy = sessions("SPY", args.refresh)[SESSIONS].reindex(days)
    beta_m, beta_b = betas()
    excess = nv[SESSIONS] - beta_m * spy - beta_b * basket
    rel = pd.Series(np.arange(len(days)) - days.searchsorted(DAY0), index=days)

    rows = []
    vol_base = nv.loc[rel.between(*WINDOWS["two months before"]), "volume"].mean()
    for w, (a, b) in WINDOWS.items():
        idx = rel.index[rel.between(a, b)]
        for series, frame in [("Nvidia", nv), ("peer basket", basket), ("Nvidia, company-specific", excess)]:
            row = {"window": w, "first_day": idx.min().date(), "last_day": idx.max().date(), "days": len(idx),
                   "series": series, **measures(frame.loc[idx])}
            if series == "Nvidia":
                row["volume_vs_two_months_before"] = float(nv.loc[idx, "volume"].mean() / vol_base)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results/nvidia_sessions.csv", index=False)

    key = []
    for day, what in KEY_DAYS.items():
        t = pd.Timestamp(day)
        if t not in excess.index:
            continue
        e = excess.loc[t]
        key.append({"day": day, "news": what, "pre": e["pre"], "regular": e["regular"], "after": e["after"],
                    "total": e.sum(), "volume_vs_two_months_before": nv.loc[t, "volume"] / vol_base})
    pd.DataFrame(key).to_csv(ROOT / "results/nvidia_session_days.csv", index=False)

    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    print(pd.DataFrame(key).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
