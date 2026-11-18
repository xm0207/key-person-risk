"""Day 0: the first trading day on which an announcement could affect the closing price.

An announcement made before the open or during the session moves that day's close, so day 0 is the
announcement date itself. One made after the close, or on a weekend or holiday, first moves the next
session's close, so day 0 is the next trading day. When the time of day is unknown the announcement
date is used, and the [0, +1] window covers the case where it came after the close.
"""

from __future__ import annotations

import pandas as pd

SAME_DAY = {"before_open", "during_hours", "unknown"}
NEXT_DAY = {"after_close", "non_trading_day"}
TIMINGS = SAME_DAY | NEXT_DAY


def day_zero(trading_days: pd.DatetimeIndex, announcement_date, timing: str) -> pd.Timestamp:
    """The first trading day in `trading_days` on which the announcement could affect the close.

    `timing` is one of before_open, during_hours, after_close, non_trading_day or unknown. A date
    labelled non_trading_day that is in fact a trading day raises, because the label is then wrong.
    """
    if timing not in TIMINGS:
        raise ValueError(f"timing must be one of {sorted(TIMINGS)}, not {timing!r}")
    days = pd.DatetimeIndex(trading_days).sort_values()
    date = pd.Timestamp(announcement_date).normalize()
    is_trading_day = date in days
    if timing == "non_trading_day" and is_trading_day:
        raise ValueError(f"{date.date()} is labelled non_trading_day but is a trading day in the calendar")
    if timing in SAME_DAY and is_trading_day:
        return date
    position = days.searchsorted(date, side="right")
    if position >= len(days):
        raise ValueError(f"no trading day after {date.date()} in the calendar")
    return days[position]
