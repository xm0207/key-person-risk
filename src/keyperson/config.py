"""Tickers, sector proxies and the data cut-off used throughout the project."""

from __future__ import annotations

DATA_END = "2026-11-18"  # last close in the data (extended from 13 Nov on 18 Nov 2026 and to 18 Nov on 19 Nov 2026; mega7-decomposition stops at 10 November 2026)

FIRMS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
MARKET = "SPY"

# Primary sector proxy for each firm: an SPDR sector ETF (XLK technology, XLY consumer discretionary,
# XLC communication services, which trades from June 2018), except Nvidia, for which the semiconductor ETF
# SMH (VanEck) is used, as in mega7-decomposition. The sector ETFs hold the firms themselves, so part of a
# firm's own shock is counted as sector; the ex-firm mega-cap basket is the robustness check.
SECTOR_PROXY = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "GOOGL": "XLC",
    "AMZN": "XLY",
    "NVDA": "SMH",
    "META": "XLC",
    "TSLA": "XLY",
}
# Nvidia's alternative proxy, the SPDR technology ETF
ALTERNATIVE_SECTOR = {"NVDA": "XLK"}

SECTOR_ETFS = sorted(set(SECTOR_PROXY.values()) | set(ALTERNATIVE_SECTOR.values()))
ALL_TICKERS = FIRMS + [MARKET, "QQQ"] + SECTOR_ETFS

# The ex-firm mega-cap basket (robustness): equal weight of the other firms that have a return that day
MIN_BASKET_MEMBERS = 3

NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "Nvidia",
    "META": "Meta",
    "TSLA": "Tesla",
}
