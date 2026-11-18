"""Nvidia, October-November 2026: company-specific abnormal returns on each information date.

Dates (sources in events/nvidia_timeline_sources.csv; times checked against EDGAR and the publishers' pages on
17 Nov 2026): 29 Oct newspaper report, published 05:01 ET, on the CEO's relationship with an executive of a cloud
customer under US scrutiny; Sat 31 Oct report on intelligence officials' concerns about that executive (day 2 =
2 Nov); 5 Nov request from three House committee chairs for a closed-door
briefing, 6 Nov report on a US memo; Sun 8 Nov board investigation and the CEO's email to staff admitting he had
been unfaithful (day 0 = 9 Nov); 11 Nov report of a relationship with a direct report and that the board would meet
(the stock fell about 8% after hours); the board accepted his resignation on 11 Nov and the 8-K was accepted by
EDGAR at 08:52 ET on 12 Nov, before the open.

Main specification: market plus an equal-weight basket of the other six mega-caps (the sector ETF SMH holds Nvidia
and absorbs part of its own shock); robustness: market plus SMH, and market only. One estimation window for all
dates: 250 trading days ending 81 trading days before 29 October. Prices: Yahoo Finance to the close of 17 November
2026 (personal use; only these window sums are written).

    ~/.venvs/key-person-risk/bin/python scripts/nvidia_timeline.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from keyperson.config import FIRMS, MARKET, MIN_BASKET_MEMBERS
from keyperson.data import ex_firm_basket, load_returns
from keyperson.engine import EventStudyEngine, Window

ROOT = Path(__file__).resolve().parents[1]
DAY0 = pd.Timestamp("2026-10-29")
WINDOWS = (
    Window("before: 1-28 Oct", -20, -1),
    Window("29-30 Oct: first newspaper report", 0, 1),
    Window("2-4 Nov: report on officials' concerns about the customer executive", 2, 4),
    Window("5-6 Nov: House request; memo report", 5, 6),
    Window("9-10 Nov: board investigation; CEO's email", 7, 8),
    Window("11 Nov: before the evening reports", 9, 9),
    Window("12 Nov: evening reports of 11 Nov; resignation", 10, 10),
    Window("13 Nov", 11, 11),
    Window("16 Nov: report that the board is not considering a return", 12, 12),
    Window("17 Nov: former CEO offers to return (results after the close)", 13, 13),
    Window("anchor", -60, -60),
)
SPECS = {"basket": "basket_ex_NVDA", "smh": "SMH", "market_only": None}


def main() -> None:
    returns = load_returns(ROOT / "data/keyperson.duckdb")
    returns["basket_ex_NVDA"] = ex_firm_basket(returns, FIRMS, "NVDA", MIN_BASKET_MEMBERS)
    engine = EventStudyEngine(returns, MARKET, windows=WINDOWS)
    out = None
    for name, sector in SPECS.items():
        res = engine.run_event("NVDA-2026-10-29", "NVDA", DAY0, sector)
        w = res.windows[res.windows["window"] != "anchor"].set_index("window")
        cols = {"car": f"car_{name}", "t": f"t_{name}"}
        part = w[["date_start", "date_end", "days", "car", "t"]].rename(columns=cols)
        if out is None:
            out = part.rename(columns={"date_start": "start", "date_end": "end"})
            out.attrs["estimation"] = (res.fit.residual_dates[0].date(), res.fit.residual_dates[-1].date())
        else:
            out = out.join(part[[f"car_{name}", f"t_{name}"]])
    out = out.reset_index()
    steps = out.iloc[1:7]  # 29 Oct - 12 Nov
    for name in SPECS:
        before = steps.iloc[:5][f"car_{name}"].sum()
        total = steps[f"car_{name}"].sum()
        out.loc[len(out)] = {"window": f"share of 29 Oct-12 Nov before the resignation day ({name})",
                             f"car_{name}": before / total}
    out.to_csv(ROOT / "results/nvidia_timeline.csv", index=False)
    print("estimation window:", out.attrs.get("estimation"))
    pd.set_option("display.width", 220)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
