"""Shared WRDS helpers: connection settings and a query wrapper that survives a dropped connection."""
from __future__ import annotations

import time

import pandas as pd
from wrds.sql import WRDS_CONNECT_ARGS

# RavenPack executive and board event types, and the columns kept from the yearly equities tables.
EVENT_TYPES = [
    "executive-scandal", "executive-firing", "executive-resignation", "executive-retirement",
    "executive-health", "executive-death", "executive-search", "executive-appointment",
    "board-member-firing", "board-member-resignation",
]

COLUMNS = [
    "rpa_date_utc", "rpa_time_utc", "timestamp_utc", "rp_story_id", "rp_entity_id", "entity_type",
    "entity_name", "country_code", "relevance", "event_sentiment_score", "event_relevance",
    "event_similarity_key", "event_similarity_days", '"group"', "type", "sub_type", "property",
    "category", "rp_position_id", "position_name", "event_text", "news_type", "source_name", "headline",
]

# Detect a dead connection in about 30 + 10 * 9 = 120 seconds rather than the OS default (hours).
CONNECT_ARGS = {**WRDS_CONNECT_ARGS, "keepalives": 1, "keepalives_idle": 30,
                "keepalives_interval": 10, "keepalives_count": 9}


def fetch(db, sql: str, label: str, tries: int = 2) -> pd.DataFrame:
    """Run a query; after a failure, roll back the broken transaction and retry once.

    SQLAlchemy refuses to reconnect until the failed transaction is rolled back, which is why
    one dropped connection used to fail every later query. After the rollback the next query
    opens a fresh connection with the stored login.
    """
    for attempt in range(1, tries + 1):
        try:
            return db.raw_sql(sql)
        except Exception as exc:
            print(f"{label}: attempt {attempt} failed ({type(exc).__name__}: {str(exc)[:150]})")
            try:
                db.connection.rollback()
            except Exception:
                pass
            if attempt == tries:
                raise
            print(f"{label}: reconnecting and retrying (approve the Duo push if one arrives)")
            time.sleep(5)
    raise AssertionError("unreachable")


# CRSP daily stock file, pulled only for each firm's window around its events.
DAYS_BEFORE, DAYS_AFTER = 600, 400


def firm_windows(sample: pd.DataFrame) -> pd.DataFrame:
    s = sample.dropna(subset=["permno"])
    w = s.groupby("permno")["event_date"].agg(["min", "max"]).reset_index()
    w["d0"] = (w["min"] - pd.Timedelta(days=DAYS_BEFORE)).dt.date
    w["d1"] = (w["max"] + pd.Timedelta(days=DAYS_AFTER)).dt.date
    return w[["permno", "d0", "d1"]]


def dsf_sql(windows: pd.DataFrame) -> str:
    values = ", ".join(f"({int(p)}, DATE '{a}', DATE '{b}')" for p, a, b in windows.itertuples(index=False))
    return ("SELECT a.permno, a.date, a.ret, a.retx, a.prc, a.shrout, a.vol "
            f"FROM crsp.dsf a JOIN (VALUES {values}) AS w(permno, d0, d1) "
            "ON a.permno = w.permno AND a.date BETWEEN w.d0 AND w.d1")
