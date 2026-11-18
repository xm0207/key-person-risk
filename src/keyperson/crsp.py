"""CRSP daily returns for the departure sample: delisting returns folded in, one wide panel for the engine.

Delisting: on a firm's delisting date the CRSP daily return is combined with the delisting return,
(1 + ret)(1 + dlret) - 1, or the delisting return alone when the daily return is missing. Without this, a
firm that is delisted after a scandal would simply drop out with its last loss unrecorded.

Market: the Fama-French market return, mktrf + rf (CRSP value-weighted, all NYSE, AMEX and Nasdaq firms).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MARKET = "MKT"


def load_dsf(folder: str | Path) -> pd.DataFrame:
    parts = [pd.read_parquet(p, columns=["permno", "date", "ret"]) for p in sorted(Path(folder).glob("dsf_*.parquet"))]
    d = pd.concat(parts, ignore_index=True)
    d["permno"] = d["permno"].astype("int64")
    d["date"] = pd.to_datetime(d["date"])
    d["ret"] = d["ret"].astype("float64")
    return d.drop_duplicates(["permno", "date"])


def add_delisting(dsf: pd.DataFrame, delist: pd.DataFrame) -> pd.DataFrame:
    dl = delist.dropna(subset=["dlret"]).copy()
    dl["permno"] = dl["permno"].astype("int64")
    dl["date"] = pd.to_datetime(dl["dlstdt"])
    dl["dlret"] = dl["dlret"].astype("float64")
    out = dsf.merge(dl[["permno", "date", "dlret"]], on=["permno", "date"], how="outer")
    both = out["ret"].notna() & out["dlret"].notna()
    out.loc[both, "ret"] = (1 + out.loc[both, "ret"]) * (1 + out.loc[both, "dlret"]) - 1
    only = out["ret"].isna() & out["dlret"].notna()
    out.loc[only, "ret"] = out.loc[only, "dlret"]
    return out.drop(columns="dlret").sort_values(["permno", "date"]).reset_index(drop=True)


def returns_panel(dsf: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Wide panel: one column per PERMNO (as a string) plus MKT, indexed by date, NaN where no return."""
    wide = dsf.pivot(index="date", columns="permno", values="ret")
    wide.columns = [str(c) for c in wide.columns]
    f = factors.copy()
    f["date"] = pd.to_datetime(f["date"])
    mkt = (f.set_index("date")["mktrf"].astype(float) + f.set_index("date")["rf"].astype(float)).rename(MARKET)
    return wide.join(mkt, how="inner").sort_index()
