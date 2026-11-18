"""Returns for the engine, from the project's DuckDB price store (Yahoo Finance, personal use only).

The engine itself takes any DataFrame of daily simple returns, so a CRSP or other source can replace
this module without touching the engine.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mega7 import PriceStore
from mega7.event_study import equal_weight_basket


def load_returns(db: str | Path = "data/keyperson.duckdb") -> pd.DataFrame:
    """Daily simple returns of every series in the store, computed in SQL by mega7's PriceStore."""
    store = PriceStore(db)
    try:
        return store.returns("daily")
    finally:
        store.close()


def ex_firm_basket(returns: pd.DataFrame, members: list[str], exclude: str, min_members: int = 3) -> pd.Series:
    """Equal-weight basket of `members` other than `exclude`, rebalanced daily (mega7's equal_weight_basket);
    NaN on days with fewer than `min_members` returns."""
    others = [t for t in members if t != exclude and t in returns]
    return equal_weight_basket(returns[others], min_members).rename(f"basket_ex_{exclude}")
