"""Nvidia, 29 October - 18 November 2026: how much of the share-price fall came from the market, from the other
mega-caps, from other chip makers, and from Nvidia itself?

Three benchmarks, each estimated by OLS on daily returns over the window of nvidia_timeline.py (250 trading
days ending 81 trading days before 29 October 2026):
    A. SPY and an equal-weighted basket of the six other mega-caps (the paper's benchmark);
    B. A plus an equal-weighted basket of nine other chip makers (AMD, Broadcom, TSMC, Micron, Qualcomm,
       Intel, Applied Materials, Lam Research, Arm);
    C. SPY and the chip basket.
For each benchmark the sum of Nvidia's daily returns over the episode is split into the part explained by
each factor, the constant and the abnormal return, and the abnormal return is reported for the windows of
the news. Chip-maker prices: Yahoo Finance adjusted closes, cached in data/yahoo_sessions/ (personal use, not
committed). Only window sums are written, to results/nvidia_attribution.csv.

    ~/.venvs/key-person-risk/bin/python scripts/nvidia_attribution.py [--refresh]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from keyperson.config import DATA_END, FIRMS, MIN_BASKET_MEMBERS
from keyperson.data import ex_firm_basket, load_returns

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/yahoo_sessions/semis_daily_adjclose.parquet"
SEMIS = ["AMD", "AVGO", "TSM", "MU", "QCOM", "INTC", "AMAT", "LRCX", "ARM"]
DAY0 = pd.Timestamp("2026-10-29")
HEADERS = {"User-Agent": "keyperson-research-script/0.1"}
WINDOWS = {
    "29-30 Oct: first newspaper report": ("2026-10-29", "2026-10-30"),
    "2 Nov: after a Saturday report on officials' concerns": ("2026-11-02", "2026-11-02"),
    "3 Nov: election day": ("2026-11-03", "2026-11-03"),
    "4 Nov: day after the election": ("2026-11-04", "2026-11-04"),
    "5-6 Nov: House request; memo report": ("2026-11-05", "2026-11-06"),
    "9-10 Nov: board investigation; CEO's email": ("2026-11-09", "2026-11-10"),
    "11-12 Nov: evening reports; resignation": ("2026-11-11", "2026-11-12"),
    "13-17 Nov: after the resignation": ("2026-11-13", "2026-11-17"),
    "18 Nov: first session after the results": ("2026-11-18", "2026-11-18"),
    "29 Oct - 17 Nov: to the results": ("2026-10-29", "2026-11-17"),
    "29 Oct - 18 Nov: episode": ("2026-10-29", "2026-11-18"),
}
MODELS = {"A: SPY + mega-caps": ["SPY", "mega"], "B: SPY + mega-caps + chip makers": ["SPY", "mega", "chips"],
          "C: SPY + chip makers": ["SPY", "chips"]}


def chip_prices(refresh: bool) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    px = {}
    for t in SEMIS:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}", headers=HEADERS, timeout=60,
                         params={"interval": "1d", "range": "2y"})
        r.raise_for_status()
        j = r.json()["chart"]["result"][0]
        idx = pd.to_datetime(pd.to_datetime(j["timestamp"], unit="s").date)
        px[t] = pd.Series(j["indicators"]["adjclose"][0]["adjclose"], index=idx)
    out = pd.DataFrame(px).sort_index()
    out = out[out.index <= pd.Timestamp(DATA_END)]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    r = load_returns(ROOT / "data/keyperson.duckdb")
    r["mega"] = ex_firm_basket(r, FIRMS, "NVDA", MIN_BASKET_MEMBERS)
    r["chips"] = chip_prices(args.refresh).pct_change().mean(axis=1).reindex(r.index)
    d = r[["NVDA", "SPY", "mega", "chips"]].dropna()
    i0 = d.index.searchsorted(DAY0)
    est = d.iloc[i0 - 81 - 250:i0 - 81]
    ev = d.loc[DAY0:pd.Timestamp(DATA_END)]
    rows = []
    for name, cols in MODELS.items():
        x = np.column_stack([np.ones(len(est))] + [est[c] for c in cols])
        b = np.linalg.lstsq(x, est["NVDA"].to_numpy(), rcond=None)[0]
        parts = {c: b[k + 1] * ev[c] for k, c in enumerate(cols)}
        abnormal = ev["NVDA"] - b[0] - sum(parts.values())
        row = {"model": name, "betas": "; ".join(f"{c} {b[k + 1]:.2f}" for k, c in enumerate(cols)),
               "nvidia_sum": ev["NVDA"].sum(), "constant": b[0] * len(ev)}
        for c in ["SPY", "mega", "chips"]:
            row[f"part_{c}"] = parts[c].sum() if c in parts else np.nan
        for w, (a, z) in WINDOWS.items():
            row[w] = abnormal.loc[a:z].sum()
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/nvidia_attribution.csv", index=False)
    pd.set_option("display.width", 250)
    print((out.set_index("model").drop(columns="betas").T * 100).round(1).to_string())
    print(out[["model", "betas"]].to_string(index=False))


if __name__ == "__main__":
    main()
